#!/bin/bash
# hi_kl_hi_surp ablation on UCLACG GPU 1.
# Mirrors the n1bs16 LoRA fullseq baseline (Qwen2.5-Math-1.5B student,
# Qwen3-1.7B teacher) but with token_select_mode=hi_kl_hi_surp.
#
# This is the principled-selection counterpart to format_mask:
#   format_mask     : drops format tokens (negative selection by category)
#   hi_kl_hi_surp   : keeps only tokens where |s_lp - t_lp| > p75 AND
#                     -s_lp > p75 (positive selection by reasoning-pivot
#                     signal)
#
# Reference baselines (n1bs16, 3200 problems, 200 steps):
#   no-distill      -> avg@4 50.95%
#   pos-100  LoRA   -> avg@4 65.85%
#   fullseq  LoRA   -> ~65%
#
# Hypothesis: if prefix-100 wins because it concentrates hiKL_hiE tokens
# (per docs/position_x_bucket.md: prefix-100 is 2.22x enriched in
# hiKL_hiE vs the full-seq baseline), then directly selecting hiKL_hiE
# tokens regardless of position should match or exceed prefix-100.

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

OUTDIR="checkpoints/hi-kl-hi-surp-1.7b-math"
RUN_NAME="hi-kl-hi-surp-1.7b-math"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

# Single-GPU mode: student/teacher/vLLM take turns on GPU 1.
# Patched on_policy_distill_positional.py also offloads optimizer state
# to CPU before vLLM (single_gpu path), which lets vllm_gpu_util=0.50
# fit even at chunk_idx>0.
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
    --token_select_mode hi_kl_hi_surp \
    --hi_kl_quantile 0.75 \
    --hi_surp_quantile 0.75 \
    --position_limit 0 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== hi_kl_hi_surp training done ==="
