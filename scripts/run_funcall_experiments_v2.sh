#!/bin/bash
# Function calling distillation experiments - V2
# 1. LoRA pos-150 (补训)
# 2. FullFT: pos-50, pos-100, pos-150, pos-200, full-seq
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
        --temperature 0.7 --save_steps 50 --log_steps 10 --eval_steps 0 \
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

# 1. LoRA pos-150 (补训)
echo "========== LoRA pos-150 =========="
run_train 150 "checkpoints/funcall-pos150-n1" "funcall-pos150-n1bs16" \
    "--lr 5e-5 --lora_r 32 --lora_alpha 64"

# 2. FullFT experiments (lr=5e-6)
echo "========== FullFT experiments =========="
for POS in 50 100 150 200; do
    run_train $POS "checkpoints/funcall-pos${POS}-n1-fullft" "funcall-pos${POS}-n1bs16-fullft" \
        "--lr 5e-6 --full_finetune"
done

# FullFT full-seq
run_train 0 "checkpoints/funcall-fullseq-n1-fullft" "funcall-fullseq-n1bs16-fullft" \
    "--lr 5e-6 --full_finetune --max_new_tokens 512"

echo "=== All V2 training complete ==="
