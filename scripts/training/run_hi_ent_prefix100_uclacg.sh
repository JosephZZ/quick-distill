#!/bin/bash
# prefix-100 + hi_ent (drop low full-vocab-entropy within prefix) on UCLACG GPU 1.
#
# Threshold = ABSOLUTE H <= 0.01 (top-1 prob >= 99%, deterministic format /
# structural / digit tokens). Picked from offline analysis of 100 undistilled
# trajectories (docs/entropy_distribution_prefix/):
#   - At H<=0.01 in prefix-100: drop 17.9% of positions (~17 tokens / traj median).
#   - Bottom-H tokens: '2','�','.',' ',' \\','{','1',' the',' to',' =',' of', ...
#     -> pure LaTeX/numeric/structural format. Same population the paper's
#     hiKL_loE bucket targets.
# Distribution is long-tailed with no bimodal valley, so absolute cutoff
# is justified by the top-1-prob interpretation, NOT by a knee.
#
# Hypothesis ladder:
#   plain prefix-100 (n1bs16 LoRA) -> 65.85%
#   hi_ent prefix-100, abs H<=0.01:
#     ≥ 65.85%  -> the format/structural residue inside prefix is net-negative.
#     < 65.85%  -> gradient density wins over selection purity inside prefix.
#
# Mirrors n1bs16 LoRA recipe (Qwen2.5-Math-1.5B → Qwen3-1.7B, 200 steps).

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

OUTDIR="checkpoints/hi-ent-abs01-prefix100-1.7b-math"
RUN_NAME="hi-ent-abs01-prefix100-1.7b-math"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

CUDA_VISIBLE_DEVICES=1 python on_policy_distill_positional.py \
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
    --position_limit 100 \
    --hi_ent_threshold 0.01 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 \
    --gen_batch_size 4 \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo "=== hi_ent prefix-100 (abs H<=0.01) training done ==="
