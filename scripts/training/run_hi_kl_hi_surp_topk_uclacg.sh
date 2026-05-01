#!/bin/bash
# hi_kl_hi_surp_topk (K=100) ablation on UCLACG GPU 1.
# Budget-controlled counterpart to hi_kl_hi_surp: instead of per-batch
# P75 thresholds (variable count), this picks per-trajectory top-K
# positions ranked by joint score = KL * surprise.
#
# Slots directly into the K=100 token-selection family in paper Table 4
# (prefix-100, top-KL-100, ent-teacher-100, ent-student-100, random-100,
# middle-100, last-100) for an apples-to-apples comparison.
#
# Hypothesis: if prefix-100 wins because it concentrates hiKL & hiE
# tokens, then explicitly choosing the 100 hi-KL-hi-surp tokens (anywhere
# in the response) should match or exceed prefix-100 (65.85%).

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
STEPS=200
BS=16
N_SAMPLES=1
MINI_BS=2

# K is taken from --position_limit; the new mode reuses it as top-K.
K=100

OUTDIR="checkpoints/hi-kl-hi-surp-topk${K}-1.7b-math"
RUN_NAME="hi-kl-hi-surp-topk${K}-1.7b-math"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

CUDA_VISIBLE_DEVICES=1 python on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems "$NUM_PROBLEMS" \
    --bs "$BS" --n_samples "$N_SAMPLES" --mini_bs "$MINI_BS" \
    --temperature 0.7 --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-hi-kl-hi-surp \
    --output_dir "$OUTDIR" \
    --token_select_mode hi_kl_hi_surp_topk \
    --position_limit "$K" \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== hi_kl_hi_surp_topk K=${K} training done ==="
