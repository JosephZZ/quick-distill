#!/bin/bash
# full-seq + hi_ent (absolute H<=0.01) on UCLACG GPU 0 (after #16 finishes).
#
# Same selection rule as the prefix-100 hi_ent run, but applied to the WHOLE
# response. Estimated drop (from 100-traj offline analysis, docs/entropy_distribution_prefix/):
#   pos 0-49   :   9.2% dropped
#   pos 50-99  :  26.9% dropped
#   pos 100-199:  45.0% dropped
#   pos 200-299:  55.8% dropped
#   pos 300-399:  63.7% dropped
#   pos 400+   :  estimated 70%+ dropped (LaTeX/arith/newline plateau)
# -> overall ~50-60% of full-seq positions filtered out, mostly format
#    tokens late in the response.
#
# This is also the "format-token-mask + full-seq" baseline that paper R4
# reviewer asked for to disentangle position-vs-content confound.
#
# Hypothesis ladder:
#   plain full-seq baseline ~ 65%
#   hi_ent full-seq abs H<=0.01:
#     ≥ 65%  -> deterministic-format tokens are net-negative even in full-seq;
#               positional advantage was largely about avoiding them.
#     < 65%  -> the format tokens in tail still carry useful gradient
#               (e.g., for length calibration / format learning).

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
BS=16
N_SAMPLES=1
MINI_BS=2

OUTDIR="checkpoints/hi-ent-abs01-fullseq-1.7b-math"
RUN_NAME="hi-ent-abs01-fullseq-1.7b-math"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

CUDA_VISIBLE_DEVICES=0 python on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems "$NUM_PROBLEMS" \
    --bs "$BS" --n_samples "$N_SAMPLES" --mini_bs "$MINI_BS" \
    --temperature 0.7 --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-hi-ent \
    --output_dir "$OUTDIR" \
    --token_select_mode hi_ent \
    --position_limit 0 \
    --hi_ent_threshold 0.01 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== hi_ent full-seq (abs H<=0.01) training done ==="
