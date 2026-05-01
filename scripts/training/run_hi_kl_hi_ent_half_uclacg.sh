#!/bin/bash
# hi_kl_hi_ent half-length variant on UCLACG GPU 1.
#
# Per-trajectory top-K by score = KL * full-vocab entropy H(p_student).
# K = floor(response_len * 0.5) -- half the response length per trajectory.
#
# Differences vs hi_kl_hi_surp_topk K=100:
#   - score uses full-vocab entropy H(p)=-sum p log p instead of surprise -log p_y
#   - K scales with response length (typical ~400 tokens vs fixed 100)
#
# Hypothesis (per user):
#   - K=100 gives too few tokens; with longer response budget more learning signal
#   - full-vocab entropy is a stricter "uncertainty" measure than surprise
#     (surprise depends on the sampled token; entropy is over whole distribution)
#
# Reference baselines (n1bs16, 3200 problems, 200 steps):
#   no-distill        -> 50.95%
#   pos-100  LoRA     -> 65.85%
#   fullseq  LoRA     -> ~65%
#   hi_kl_hi_surp     -> 54.30% (P75 quantile intersection, full-seq style)

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

FRAC=0.5

OUTDIR="checkpoints/hi-kl-hi-ent-half-1.7b-math"
RUN_NAME="hi-kl-hi-ent-half-1.7b-math"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

CUDA_VISIBLE_DEVICES=1 python on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems "$NUM_PROBLEMS" \
    --bs "$BS" --n_samples "$N_SAMPLES" --mini_bs "$MINI_BS" \
    --temperature 0.7 --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-hi-kl-hi-ent \
    --output_dir "$OUTDIR" \
    --token_select_mode hi_kl_hi_ent_topk \
    --top_k_frac "$FRAC" \
    --position_limit 0 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== hi_kl_hi_ent_half (frac=${FRAC}) training done ==="
