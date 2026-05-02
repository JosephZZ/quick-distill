#!/bin/bash
# Wait for GPU 0 to free, then launch format_mask experiment masking only
# math_latex + structural categories (keeps numbers & operators).
# Expected: ~35% of tokens masked, ~44% of total KL removed.

set -uo pipefail

BASE="/zhi_backup/ziheng/quick-distillation/quick-distillation"
GPU=0
NAME="format-mask-latex-struct-1.7b-math"

cd "$BASE"
mkdir -p logs

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

# Wait for GPU 0 memory to drop below 5 GB used (effectively free).
echo "[launcher] $(date) -- waiting for GPU $GPU to free..."
while :; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
    if [ "$USED" -lt 5000 ]; then
        echo "[launcher] $(date) -- GPU $GPU is free (${USED} MiB used). Launching."
        break
    fi
    sleep 60
done

if pgrep -f "wandb_run_name $NAME" > /dev/null; then
    echo "[launcher] $NAME already running, abort."
    exit 0
fi

nohup env CUDA_VISIBLE_DEVICES=$GPU python on_policy_distill_positional.py \
    --student_model Qwen/Qwen2.5-Math-1.5B --teacher_model Qwen/Qwen3-1.7B \
    --dataset AI-MO/NuminaMath-CoT --num_problems 3200 \
    --bs 16 --n_samples 1 --mini_bs 2 \
    --temperature 0.7 --lr 5e-5 --lora_r 32 --lora_alpha 64 \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-token-select \
    --output_dir "checkpoints/$NAME" \
    --token_select_mode format_mask \
    --format_mask_cats "structural,math_latex" \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 --gen_batch_size 4 \
    --wandb_run_name "$NAME" \
    > "logs/${NAME}.log" 2>&1 &

PID=$!
echo "[launcher] launched PID=$PID name=$NAME"
sleep 30
echo "--- first 40 log lines ---"
head -40 "logs/${NAME}.log" || true
