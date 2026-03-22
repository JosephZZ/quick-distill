#!/bin/bash
# Function calling distillation experiments
# LoRA, n1bs16, 3200 problems → 200 steps, save every 50
# Configs: pos-50, pos-100, pos-200, full-seq
set -e

BASE_DIR="/CGLab/ziheng/projects/dft-distill"
cd "$BASE_DIR"
mkdir -p logs

STUDENT="Qwen/Qwen2.5-Math-1.5B"
TEACHER="Qwen/Qwen3-1.7B"
DATASET="data/funcall/train.jsonl"
SYS_PROMPT='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

run_train() {
    local POS=$1
    local OUTDIR=$2
    local RUN_NAME=$3
    local EXTRA_ARGS=$4

    if [ -d "$OUTDIR/step_200" ]; then
        echo "=== $RUN_NAME already done, skipping ==="
        return
    fi
    echo "=== Training $RUN_NAME ==="
    CUDA_VISIBLE_DEVICES=1 python on_policy_distill_positional.py \
        --student_model "$STUDENT" --teacher_model "$TEACHER" \
        --dataset "$DATASET" --num_problems 3200 \
        --bs 16 --n_samples 1 \
        --temperature 0.7 --lr 5e-5 --lora_r 32 --lora_alpha 64 \
        --save_steps 50 --log_steps 10 --eval_steps 0 \
        --student_gpu 0 --teacher_gpu 0 \
        --problem_field problem \
        --system_prompt "$SYS_PROMPT" \
        --wandb_project dft-distill-funcall \
        --output_dir "$OUTDIR" \
        --position_limit "$POS" \
        --wandb_run_name "$RUN_NAME" \
        $EXTRA_ARGS \
        2>&1 | tee "logs/${RUN_NAME}.log"
    echo "=== $RUN_NAME done ==="
}

# Positional variants
for POS in 50 100 200; do
    run_train $POS "checkpoints/funcall-pos${POS}-n1" "funcall-pos${POS}-n1bs16"
done

# Full-seq (position_limit=0, generate up to 512 tokens)
run_train 0 "checkpoints/funcall-fullseq-n1" "funcall-fullseq-n1bs16" "--max_new_tokens 512"

echo "=== All training complete ==="
