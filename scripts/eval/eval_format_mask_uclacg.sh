#!/bin/bash
# Eval format-mask LoRA checkpoints on UCLACG.
# Usage: bash eval_format_mask_uclacg.sh [step1 step2 ...]
# Default steps: whatever is in checkpoints/format-mask-1.7b-math/step_*/
set -eo pipefail

cd /zhi_backup/ziheng/quick-distillation

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

EXP="format-mask-1.7b-math"
STUDENT="Qwen/Qwen2.5-Math-1.5B"

if [ "$#" -gt 0 ]; then
    STEPS=("$@")
else
    STEPS=()
    for d in checkpoints/${EXP}/step_*; do
        s=$(basename "$d" | sed 's/step_//')
        STEPS+=("$s")
    done
fi

echo "=== Evaluating ${EXP}, steps: ${STEPS[*]} ==="

for step in "${STEPS[@]}"; do
    CKPT="checkpoints/${EXP}/step_${step}"
    MERGED="${CKPT}/_eval_merged"
    EVAL_DIR="${CKPT}/eval"

    if [ ! -d "$CKPT" ]; then
        echo "Skipping step_${step} (no checkpoint)"
        continue
    fi

    if [ -f "${EVAL_DIR}/summary.json" ]; then
        echo "step_${step} already evaluated:"
        python3 -c "import json; d=json.load(open('${EVAL_DIR}/summary.json')); print(f'  avg@4={d[\"avg_accuracy\"]*100:.2f}%, pass@4={d[\"pass_accuracy\"]*100:.2f}%')"
        continue
    fi

    if [ ! -d "$MERGED" ]; then
        echo "=== Merging step_${step} on CPU ==="
        CUDA_VISIBLE_DEVICES="" python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
base = AutoModelForCausalLM.from_pretrained('${STUDENT}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${CKPT}')
model = model.merge_and_unload()
model.save_pretrained('${MERGED}')
AutoTokenizer.from_pretrained('${STUDENT}').save_pretrained('${MERGED}')
print('Merged to ${MERGED}')
"
    fi

    echo "=== Eval step_${step} on GPU 1 ==="
    CUDA_VISIBLE_DEVICES=1 python3 eval_math500.py \
        --model "${MERGED}" \
        --output_dir "${EVAL_DIR}" \
        --n_samples 4 --temperature 0.7 \
        --gpu_memory_utilization 0.85 \
        --max_model_len 4096 2>&1 | tail -30

    echo "--- step_${step} ---"
    python3 -c "import json; d=json.load(open('${EVAL_DIR}/summary.json')); print(f'avg@4={d[\"avg_accuracy\"]*100:.2f}%  pass@4={d[\"pass_accuracy\"]*100:.2f}%  maj@4={d[\"maj_accuracy\"]*100:.2f}%')"
done

echo "=== format-mask eval complete ==="
