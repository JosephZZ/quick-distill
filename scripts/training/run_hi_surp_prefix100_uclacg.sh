#!/bin/bash
# prefix-100 + hi_surp (drop low-entropy within prefix) on UCLACG GPU 1.
#
# Tests whether removing the residual hiKL_loE format tokens from the
# first 100 positions improves over plain prefix-100 (65.85% avg@4).
# Prefix-100 contains ~3.1% hiKL_loE tokens (per docs/conceptual_framework_review.md);
# this run masks the bottom 50% of positions (by student surprise -s_lp)
# within the prefix region, keeping only ~50 high-surprise positions per
# trajectory.
#
# Hypothesis ladder:
#   plain prefix-100  -> 65.85%      (current best)
#   prefix-100 + hi_surp:
#     ≥ 65.85%  -> the 3.1% loE residue IS net-negative inside prefix.
#     <  65.85% -> within prefix, "more tokens" matters more than purity
#                  (gradient density wins over selection purity).
#
# Mirrors the n1bs16 LoRA recipe used for prefix-100 / hi_kl_hi_surp.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
STEPS=200
BS=16
N_SAMPLES=1
MINI_BS=2

OUTDIR="checkpoints/hi-surp-prefix100-1.7b-math"
RUN_NAME="hi-surp-prefix100-1.7b-math"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

# Single-GPU mode: student/teacher/vLLM take turns on GPU 1.
CUDA_VISIBLE_DEVICES=1 python on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems "$NUM_PROBLEMS" \
    --bs "$BS" --n_samples "$N_SAMPLES" --mini_bs "$MINI_BS" \
    --temperature 0.7 --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-hi-surp \
    --output_dir "$OUTDIR" \
    --token_select_mode hi_surp \
    --position_limit 100 \
    --hi_surp_quantile 0.5 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== hi_surp prefix-100 training done ==="
