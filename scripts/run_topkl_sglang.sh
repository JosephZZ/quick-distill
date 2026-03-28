#!/bin/bash
# Top-KL token selection experiment using SGLang (dual-GPU)
# GPU 0: student (training only, full 46GB)
# GPU 2: teacher (scoring) + SGLang (generation, ~14GB)
# Config: n1bs16 (3200 problems, K=100, 200 steps)
set -eo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

CONDA_ENV="/xuanwu-tank/east/antarachugh/envs/distill"
PYTHON="$CONDA_ENV/bin/python"
export PATH="$CONDA_ENV/bin:$PATH"
export WANDB_MODE=disabled
STUDENT="Qwen/Qwen2.5-Math-1.5B"
TEACHER="Qwen/Qwen3-1.7B"

K=100
LR=5e-5
LORA_R=32
LORA_ALPHA=64

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

echo "========== Top-KL Token Selection (K=$K, SGLang, dual-GPU) =========="
echo "Student: $STUDENT | Teacher: $TEACHER"
echo "GPUs: 0 (student training), 2 (teacher + SGLang)"

# Kill any orphan SGLang processes first
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 2

# CUDA_VISIBLE_DEVICES="0,2" → logical 0=physical 0, logical 1=physical 2
# student_gpu=0: student training on GPU 0 (full 46GB)
# teacher_gpu=1: teacher scoring on GPU 2
# vllm_gpu=1: SGLang also on GPU 2 (coexists with teacher, ~14GB)
CUDA_VISIBLE_DEVICES="0,2" $PYTHON on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems 3200 \
    --bs 16 --n_samples 1 --mini_bs 4 \
    --temperature 0.7 --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 1 --vllm_gpu 1 \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-token-select \
    --output_dir "checkpoints/topkl-k100-sglang" \
    --position_limit $K \
    --token_select_mode "top_kl" \
    --wandb_run_name "topkl-k100-sglang" \
    --use_sglang --max_new_tokens 2048 --teacher_micro_bs 4 \
    --vllm_gpu_util 0.30 \
    2>&1 | tee "logs/topkl-k100-sglang.log"

echo "=== Training done ==="

# Kill SGLang server
pkill -f "sglang.launch_server" 2>/dev/null || true

# Eval all checkpoints
echo "=== Evaluating checkpoints ==="
for STEP in 50 100 150 200; do
    LORA_PATH="checkpoints/topkl-k100-sglang/step_${STEP}"
    EVAL_DIR="checkpoints/topkl-k100-sglang/eval_step_${STEP}"

    [ ! -d "$LORA_PATH" ] && continue
    [ -f "$EVAL_DIR/summary.json" ] && echo "Step $STEP eval exists, skipping" && continue

    MERGED="checkpoints/topkl-k100-sglang/_eval_merged_step_${STEP}"
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
