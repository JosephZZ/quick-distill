#!/bin/bash
# top_entropy_student K=200 ablation on UCLACG.
# Counterpart to existing topent-student-k100-fixed (61.35% avg@4).
# Tests whether doubling K helps the student-entropy selection method.
#
# Reference baselines (n1bs16, 3200 problems, 200 steps):
#   no-distill                           -> 50.95%
#   pos-100  LoRA                        -> 65.85%
#   top_entropy_student K=100 (existing) -> 61.35%

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

STUDENT="Qwen/Qwen2.5-Math-1.5B"
TEACHER="Qwen/Qwen3-1.7B"

LR=5e-5
LORA_R=32
LORA_ALPHA=64
NUM_PROBLEMS=3200
BS=16
N_SAMPLES=1
MINI_BS=2
K=200

GPU_ID="${1:-0}"

OUTDIR="checkpoints/topent-student-k${K}-1.7b-math"
RUN_NAME="topent-student-k${K}-1.7b-math"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

CUDA_VISIBLE_DEVICES="$GPU_ID" python on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems "$NUM_PROBLEMS" \
    --bs "$BS" --n_samples "$N_SAMPLES" --mini_bs "$MINI_BS" \
    --temperature 0.7 --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-token-select \
    --output_dir "$OUTDIR" \
    --token_select_mode top_entropy_student \
    --position_limit "$K" \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== top_entropy_student K=${K} training done ==="
