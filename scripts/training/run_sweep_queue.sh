#!/bin/bash
# Queue: top-KL K=200, then top-KL K=300, then top-entropy-student K=300.
# All on GPU 1 (sequential). GPU 0 has top-entropy-student K=200 already.
set -eo pipefail
cd /zhi_backup/ziheng/quick-distillation/quick-distillation
mkdir -p logs

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

run_one () {
    local MODE="$1"
    local K="$2"
    local NAME="$3"
    echo "=== launching $NAME (mode=$MODE, K=$K) at $(date) ==="
    CUDA_VISIBLE_DEVICES=1 python on_policy_distill_positional.py \
        --student_model Qwen/Qwen2.5-Math-1.5B --teacher_model Qwen/Qwen3-1.7B \
        --dataset AI-MO/NuminaMath-CoT --num_problems 3200 \
        --bs 16 --n_samples 1 --mini_bs 2 \
        --temperature 0.7 --lr 5e-5 --lora_r 32 --lora_alpha 64 \
        --save_steps 50 --log_steps 10 --eval_steps 0 \
        --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
        --system_prompt "$MATH_SYS" \
        --wandb_project dft-distill-token-select \
        --output_dir "checkpoints/$NAME" \
        --token_select_mode "$MODE" \
        --position_limit "$K" \
        --max_new_tokens 2048 \
        --teacher_micro_bs 4 --gen_batch_size 4 \
        --wandb_run_name "$NAME" \
        2>&1 | tee "logs/${NAME}.log"
    echo "=== $NAME done at $(date) ==="
}

run_one top_kl              200 topkl-k200-1.7b-math
run_one top_kl              300 topkl-k300-1.7b-math
run_one top_entropy_student 300 topent-student-k300-1.7b-math

echo "=== all 3 runs done at $(date) ==="
