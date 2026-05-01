#!/bin/bash
# Format-mask ablation on UCLACG GPU 1.
# Mirrors the n1bs16 LoRA fullseq baseline (Qwen2.5-Math-1.5B student,
# Qwen3-1.7B teacher) but with token_select_mode=format_mask.
#
# Reference baselines from the paper (n1bs16, 3200 problems, 200 steps):
#   pos-100  LoRA -> avg@4 65.85%   (this is the "headline" prefix number)
#   fullseq  LoRA -> ~65%           (similar headline)
#   no-distill baseline -> 50.95%
#
# Hypothesis under test: if the prefix advantage is *only* about format-noise
# dilution, format_mask (full-seq with format tokens zeroed) should match or
# exceed prefix-100. If it falls between fullseq and prefix-100, format is
# part of the story but not all of it. If it doesn't beat fullseq, position
# carries independent signal.

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

OUTDIR="checkpoints/format-mask-1.7b-math"
RUN_NAME="format-mask-1.7b-math"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

# Single-GPU mode: student/teacher/sglang take turns on GPU 1
CUDA_VISIBLE_DEVICES=1 python on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems "$NUM_PROBLEMS" \
    --bs "$BS" --n_samples "$N_SAMPLES" --mini_bs "$MINI_BS" \
    --temperature 0.7 --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-format-mask \
    --output_dir "$OUTDIR" \
    --token_select_mode format_mask \
    --position_limit 0 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== format_mask training done ==="
