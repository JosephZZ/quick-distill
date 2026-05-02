#!/bin/bash
# Wait for GPU 1 to free (Qwen3-4B teacher eval to finish), then launch
# hi-ent-abs01-fullseq: full-sequence training with H>0.01 absolute threshold.

set -uo pipefail

BASE="/zhi_backup/ziheng/quick-distillation/quick-distillation"
GPU=1
NAME="hi-ent-abs01-fullseq-1.7b-math"

cd "$BASE"
mkdir -p logs

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

# Wipe previous partial output to avoid resume confusion (killed run left empty config).
if [ -d "checkpoints/$NAME" ] && [ ! -d "checkpoints/$NAME/step_50" ]; then
    echo "[launcher] wiping empty prior $NAME dir"
    rm -rf "checkpoints/$NAME"
fi

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
    --wandb_project dft-distill-hi-ent \
    --output_dir "checkpoints/$NAME" \
    --token_select_mode hi_ent --hi_ent_threshold 0.01 --position_limit 0 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 --gen_batch_size 4 \
    --wandb_run_name "$NAME" \
    > "logs/${NAME}.log" 2>&1 &

PID=$!
echo "[launcher] launched PID=$PID name=$NAME"
sleep 30
echo "--- first 40 log lines ---"
head -40 "logs/${NAME}.log" || true
