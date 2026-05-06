#!/usr/bin/env bash
# Re-eval all Gemma coding (HE/HE+/MBPP/MBPP+): student baseline + every LoRA
# checkpoint, using the gemma3 venv (vLLM 0.11.0).
set -euo pipefail

PY=/zhi_backup/ziheng/venvs/gemma3/bin/python
STUDENT=/mnt/ziheng/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
ls "$STUDENT" >/dev/null

cd /zhi_backup/ziheng/quick-distillation
export TOKENIZERS_PARALLELISM=false
export GPU=${CUDA_VISIBLE_DEVICES:-0}

OUT_BASE=eval_results/gemma_coding_rerun
mkdir -p "$OUT_BASE"

run_eval () {
    local model_path="$1"
    local out_dir="$2"
    mkdir -p "$out_dir"
    if [[ -f "$out_dir/he_score.txt" && -f "$out_dir/mbpp_score.txt" ]]; then
        echo "[skip] $out_dir already scored"
        return 0
    fi
    for ds in humaneval mbpp; do
        echo "=== $(date) $ds gen: $model_path ==="
        CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/eval_humaneval.py \
            --model "$model_path" --dataset $ds --output_dir "$out_dir" \
            --n_samples 1 --temperature 0.0 --max_tokens 512 \
            --gpu_memory_utilization 0.50 --trust_remote_code
    done
    local he_jsonl="$out_dir/humaneval_${model_path//\//--}.jsonl"
    local mbpp_jsonl="$out_dir/mbpp_${model_path//\//--}.jsonl"
    echo "=== $(date) evalplus HumanEval ==="
    "$PY" -m evalplus.evaluate --dataset humaneval --samples "$he_jsonl" 2>&1 | tee "$out_dir/he_score.txt"
    echo "=== $(date) evalplus MBPP ==="
    "$PY" -m evalplus.evaluate --dataset mbpp --samples "$mbpp_jsonl" 2>&1 | tee "$out_dir/mbpp_score.txt"
}

merge_lora () {
    local lora_path="$1"
    local merged_path="$2"
    if [[ -d "$merged_path" ]]; then
        echo "[skip merge] $merged_path exists"
        return 0
    fi
    echo "[merge] $lora_path -> $merged_path"
    CUDA_VISIBLE_DEVICES="" "$PY" - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("$STUDENT", torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
m = PeftModel.from_pretrained(base, "$lora_path")
merged = m.merge_and_unload()
merged.save_pretrained("$merged_path")
AutoTokenizer.from_pretrained("$STUDENT", trust_remote_code=True).save_pretrained("$merged_path")
print("merged ok")
PY
}

# 1. Student baseline
echo "######## STUDENT BASELINE (gemma-2-2b-it) ########"
run_eval "$STUDENT" "$OUT_BASE/baseline"

# 2. LoRA checkpoints
for EXP in v1-gemma-pos50-coding v1-gemma-pos100-coding v1-gemma-fullseq-coding; do
    for STEP in 50 100 150 200; do
        LORA=checkpoints/$EXP/step_$STEP
        if [[ ! -f "$LORA/adapter_model.safetensors" ]]; then
            echo "[no checkpoint] $LORA"; continue
        fi
        MERGED=checkpoints/$EXP/_eval_merged_step_$STEP
        OUT=$OUT_BASE/$EXP/step_$STEP
        merge_lora "$LORA" "$MERGED"
        run_eval "$MERGED" "$OUT"
        rm -rf "$MERGED"
    done
done

echo "########  CODING DONE  ########"
echo "Summaries:"
for f in $(find $OUT_BASE -name "he_score.txt" | sort); do
    dir=$(dirname "$f")
    he=$(grep -A1 "humaneval (base tests)" "$f" | tail -1 | awk '{print $2}')
    hep=$(grep -A1 "humaneval+ (base + extra tests)" "$f" | tail -1 | awk '{print $2}')
    mb=$(grep -A1 "mbpp (base tests)" "$dir/mbpp_score.txt" 2>/dev/null | tail -1 | awk '{print $2}')
    mbp=$(grep -A1 "mbpp+ (base + extra tests)" "$dir/mbpp_score.txt" 2>/dev/null | tail -1 | awk '{print $2}')
    echo "  ${dir#$OUT_BASE/}  HE=$he  HE+=$hep  MBPP=$mb  MBPP+=$mbp"
done
