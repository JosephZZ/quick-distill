#!/bin/bash
# Top-KL token selection experiment using vLLM (dual-GPU)
# GPU 0: student (training + vLLM generation via offload)
# GPU 2: teacher (scoring, stays on GPU)
# Config: n1bs16 (3200 problems, K=100, 200 steps)
set -eo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

CONDA_ENV="/xuanwu-tank/east/antarachugh/envs/distill"
PYTHON="$CONDA_ENV/bin/python"
export PATH="$CONDA_ENV/bin:$PATH"
STUDENT="Qwen/Qwen2.5-Math-1.5B"
TEACHER="Qwen/Qwen3-1.7B"

K=100
LR=5e-5
LORA_R=32
LORA_ALPHA=64

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

echo "========== Top-KL Token Selection (K=$K, vLLM, dual-GPU) =========="
echo "Student: $STUDENT | Teacher: $TEACHER"
echo "GPUs: 0 (student+vLLM), 2 (teacher)"

# CUDA_VISIBLE_DEVICES="0,2" → logical 0=physical 0, logical 1=physical 2
# student_gpu=0: student training + vLLM (student offloads to CPU during vLLM)
# teacher_gpu=1: teacher stays on GPU 2, no offload needed
# vllm_gpu=0: vLLM uses same GPU as student (but student offloads first)
CUDA_VISIBLE_DEVICES="0,2" $PYTHON on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems 3200 \
    --bs 16 --n_samples 1 \
    --temperature 0.7 --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 1 --vllm_gpu 0 \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-token-select \
    --output_dir "checkpoints/topkl-k100-vllm" \
    --position_limit $K \
    --token_select_mode "top_kl" \
    --wandb_run_name "topkl-k100-vllm" \
    --use_vllm --max_new_tokens 2048 --teacher_micro_bs 4 \
    --vllm_gpu_util 0.85 \
    2>&1 | tee "logs/topkl-k100-vllm.log"

echo "=== Training done ==="

# Eval all checkpoints
echo "=== Evaluating checkpoints ==="
for STEP in 50 100 150 200; do
    LORA_PATH="checkpoints/topkl-k100-vllm/step_${STEP}"
    EVAL_DIR="checkpoints/topkl-k100-vllm/eval_step_${STEP}"

    [ ! -d "$LORA_PATH" ] && continue
    [ -f "$EVAL_DIR/summary.json" ] && echo "Step $STEP eval exists, skipping" && continue

    MERGED="checkpoints/topkl-k100-vllm/_eval_merged_step_${STEP}"
    echo "Merging step $STEP..."
    CUDA_VISIBLE_DEVICES="" $PYTHON -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
base = AutoModelForCausalLM.from_pretrained('$STUDENT', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '$LORA_PATH')
merged = model.merge_and_unload()
merged.save_pretrained('$MERGED')
AutoTokenizer.from_pretrained('$STUDENT', trust_remote_code=True).save_pretrained('$MERGED')
print('Merged.')
"

    echo "Evaluating step $STEP..."
    CUDA_VISIBLE_DEVICES="0" $PYTHON eval_math500.py \
        --model "$MERGED" --output_dir "$EVAL_DIR" \
        --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.85

    rm -rf "$MERGED"
    echo "Step $STEP eval done"
done

echo "=== All done ==="
