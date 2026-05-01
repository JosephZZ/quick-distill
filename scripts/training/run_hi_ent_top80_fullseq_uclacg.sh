#!/bin/bash
# top-80% high-entropy + full-seq on UCLACG.
#
# Threshold = ABSOLUTE H > 0.0004 — keep top 80% of positions by full-vocab
# entropy (drop only the most extreme "near-deterministic" tail). Computed
# from 100-traj offline distribution (docs/entropy_distribution_prefix/,
# positions 0-399):
#   - global p20(H) = 0.0004 -> keep H > 0.0004
#   - per-traj median: ~286 tokens kept (vs ~196 for H>0.01, ~68 for H>0.664)
#   - filters only the "absolutely no choice" tokens (top-1 prob > 99.96%):
#     pure punctuation, equation closers, repeated whitespace.
#
# Hypothesis ladder (full-seq):
#   plain full-seq baseline ~ 65%
#   hi_ent H>0.0004 (top-80%, ~286 tok/traj) -- THIS RUN
#   hi_ent H>0.01   (top-56%, ~196 tok/traj) -- already queued
#   hi_ent H>0.664  (top-20%, ~68  tok/traj) -- companion run
#
# If top-80% ≈ plain full-seq, then deterministic-tail tokens contribute
# essentially zero gradient (consistent with our story).
# If top-80% > plain full-seq, then even minimal filtering helps -- the
# argmax-confident positions add noise.

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

OUTDIR="checkpoints/hi-ent-top80-fullseq-1.7b-math"
RUN_NAME="hi-ent-top80-fullseq-1.7b-math"
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
    --hi_ent_threshold 0.0004 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== hi_ent top-80% (H>0.0004) full-seq training done ==="
