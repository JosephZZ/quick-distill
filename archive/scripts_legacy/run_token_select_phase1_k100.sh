#!/bin/bash
# Phase-1 token-selection distillation experiments (K=100 only)
# student: Qwen2.5-Math-1.5B, teacher: Qwen3-1.7B
# Order:
#   1) top_kl on math
#   2) top_kl on funcall (agentic)
#   3) top_entropy_student on math
#   4) top_entropy_teacher on math
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

[ -f "${HOME}/.bashrc" ] && PS1=nonempty source "${HOME}/.bashrc" 2>/dev/null || true
source "$SCRIPT_DIR/hf_models_env.sh"

STUDENT="$MATH_STUDENT_15"
TEACHER="$QWEN3_17"

LR=5e-5
LORA_R=32
LORA_ALPHA=64
K=100

MATH_EXTRA="--use_vllm --max_new_tokens 2048 --teacher_micro_bs 4"
FUNCALL_EXTRA="--use_vllm --max_new_tokens 512 --teacher_micro_bs 4 --problem_field problem"

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

train_and_eval() {
    local TASK=$1
    local MODE=$2
    local DATASET=$3
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
            --dataset "$DATASET" --num_problems 3200 \
            --bs 16 --n_samples 1 \
            --temperature 0.7 --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
            --save_steps 50 --log_steps 10 --eval_steps 0 \
            --student_gpu 0 --teacher_gpu 0 \
            --system_prompt "$SYS_PROMPT" \
            --wandb_project dft-distill-token-select \
            --output_dir "$OUTDIR" \
            --position_limit $K \
            --token_select_mode "$MODE" \
            --wandb_run_name "$RUN_NAME" \
            $EXTRA_ARGS \
            2>&1 | tee "logs/${RUN_NAME}.log"
        echo "=== $RUN_NAME training done ==="
    fi

    eval_checkpoints "$TASK" "$OUTDIR" "$RUN_NAME"
}

eval_checkpoints() {
    local TASK=$1
    local OUTDIR=$2
    local RUN_NAME=$3

    for STEP in 50 100 150 200; do
        local LORA_PATH="$OUTDIR/step_${STEP}"
        local EVAL_DIR="$OUTDIR/eval_step_${STEP}"

        [ ! -d "$LORA_PATH" ] && continue
        [ -f "$EVAL_DIR/summary.json" ] && echo "=== $RUN_NAME step $STEP eval exists, skipping ===" && continue

        local MERGED_PATH="$OUTDIR/_eval_merged_step_${STEP}"
        echo "=== Merging $RUN_NAME step $STEP ==="
        CUDA_VISIBLE_DEVICES="" python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
base = AutoModelForCausalLM.from_pretrained('$STUDENT', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '$LORA_PATH')
merged = model.merge_and_unload()
merged.save_pretrained('$MERGED_PATH')
AutoTokenizer.from_pretrained('$STUDENT', trust_remote_code=True).save_pretrained('$MERGED_PATH')
print('Merged.')
"

        echo "=== Evaluating $RUN_NAME step $STEP ($TASK) ==="
        eval_task "$TASK" "$MERGED_PATH" "$EVAL_DIR"

        rm -rf "$MERGED_PATH"
        echo "=== $RUN_NAME step $STEP eval done ==="
    done
}

eval_task() {
    local TASK=$1
    local MODEL=$2
    local EVAL_DIR=$3

    if [ "$TASK" = "math" ]; then
        CUDA_VISIBLE_DEVICES=1 python eval_math500.py \
            --model "$MODEL" --output_dir "$EVAL_DIR" \
            --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.50
    elif [ "$TASK" = "funcall" ]; then
        CUDA_VISIBLE_DEVICES=1 python eval_funcall.py \
            --model "$MODEL" --output_dir "$EVAL_DIR" \
            --gpu_id 0 --gpu_memory_utilization 0.50 --categories "simple,multiple"
    fi
}

echo "========== Phase-1 Token Select (K=100) =========="

# 1) top-KL on math
train_and_eval \
    "math" "top_kl" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
    "checkpoints/token-select-k100-topkl-math" "token-select-k100-topkl-math" \
    "$MATH_EXTRA"

# 2) top-KL on funcall (agentic)
train_and_eval \
    "funcall" "top_kl" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
    "checkpoints/token-select-k100-topkl-funcall" "token-select-k100-topkl-funcall" \
    "$FUNCALL_EXTRA"

# 3) top-entropy (student) on math
train_and_eval \
    "math" "top_entropy_student" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
    "checkpoints/token-select-k100-topent-student-math" "token-select-k100-topent-student-math" \
    "$MATH_EXTRA"

# 4) top-entropy (teacher) on math
train_and_eval \
    "math" "top_entropy_teacher" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
    "checkpoints/token-select-k100-topent-teacher-math" "token-select-k100-topent-teacher-math" \
    "$MATH_EXTRA"

echo "=== Phase-1 K=100 experiments done ==="
