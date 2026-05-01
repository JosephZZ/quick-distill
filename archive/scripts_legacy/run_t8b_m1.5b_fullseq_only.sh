#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

[ -f "${HOME}/.bashrc" ] && PS1=nonempty source "${HOME}/.bashrc" 2>/dev/null || true
source "$SCRIPT_DIR/hf_models_env.sh"

STUDENT="$MATH_STUDENT_15"
TEACHER="$QWEN3_8"
LR=5e-5
LORA_R=32
LORA_ALPHA=64

FULLSEQ_MATH_EXTRA="--max_new_tokens 3584 --teacher_micro_bs 1 --mini_bs 1"
FULLSEQ_CODE_EXTRA="--max_new_tokens 3584 --teacher_micro_bs 1 --mini_bs 1"
FULLSEQ_FUNCALL_EXTRA="--max_new_tokens 3584 --teacher_micro_bs 1 --mini_bs 1"

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

train_and_eval() {
    local TASK=$1
    local DATASET=$2
    local NUM_PROBLEMS=$3
    local SYS_PROMPT=$4
    local OUTDIR=$5
    local RUN_NAME=$6
    local EXTRA_ARGS=$7

    if [ -d "$OUTDIR/step_200" ]; then
        echo "=== $RUN_NAME training already done, skipping ==="
    else
        echo "=== Training $RUN_NAME ==="
        CUDA_VISIBLE_DEVICES=1 python on_policy_distill_positional.py \
            --student_model "$STUDENT" --teacher_model "$TEACHER" \
            --dataset "$DATASET" --num_problems "$NUM_PROBLEMS" \
            --bs 16 --n_samples 1 \
            --temperature 0.7 --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
            --save_steps 50 --log_steps 10 --eval_steps 0 \
            --student_gpu 0 --teacher_gpu 0 \
            --system_prompt "$SYS_PROMPT" \
            --wandb_project dft-distill-scaling \
            --output_dir "$OUTDIR" \
            --position_limit 0 \
            --wandb_run_name "$RUN_NAME" \
            $EXTRA_ARGS \
            2>&1 | tee "logs/${RUN_NAME}.log"
        echo "=== $RUN_NAME training done ==="
    fi
}

echo "========== Config B ONLY: Qwen2.5-Math-1.5B + Qwen3-8B (fullseq) =========="

train_and_eval "math" "AI-MO/NuminaMath-CoT" 3200 \
    "$MATH_SYS" "checkpoints/scale-m1.5b-t8b-math-fullseq" "scale-m1.5b-t8b-math-fullseq" \
    "$FULLSEQ_MATH_EXTRA"

train_and_eval "coding" "coseal/CodeUltraFeedback_binarized" 3200 \
    "$CODING_SYS" "checkpoints/scale-m1.5b-t8b-coding-fullseq" "scale-m1.5b-t8b-coding-fullseq" \
    "$FULLSEQ_CODE_EXTRA --problem_field instruction"

train_and_eval "funcall" "data/funcall/train.jsonl" 3200 \
    "$FUNCALL_SYS" "checkpoints/scale-m1.5b-t8b-funcall-fullseq" "scale-m1.5b-t8b-funcall-fullseq" \
    "$FULLSEQ_FUNCALL_EXTRA --problem_field problem"

echo "=== t8b + m1.5b fullseq done ==="
