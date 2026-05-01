#!/bin/bash
# Phase-1 token-selection distillation (K=100), dual-GPU orchestration
# Order:
#   1) eval existing top_kl math checkpoints with vLLM
#   2) train remaining two experiments in parallel on GPU0/GPU1
#   3) train last remaining experiment
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

[ -f "${HOME}/.bashrc" ] && PS1=nonempty source "${HOME}/.bashrc" 2>/dev/null || true
source "$SCRIPT_DIR/hf_models_env.sh"

PYTHON_BIN="${PYTHON_BIN:-/sg-pvc/miniconda3/bin/python}"

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

eval_task() {
    local GPU=$1
    local TASK=$2
    local MODEL=$3
    local EVAL_DIR=$4

    if [ "$TASK" = "math" ]; then
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval_math500.py \
            --model "$MODEL" --output_dir "$EVAL_DIR" \
            --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.50
    elif [ "$TASK" = "funcall" ]; then
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" eval_funcall.py \
            --model "$MODEL" --output_dir "$EVAL_DIR" \
            --gpu_id 0 --gpu_memory_utilization 0.50 --categories "simple,multiple"
    fi
}

eval_checkpoints() {
    local GPU=$1
    local TASK=$2
    local OUTDIR=$3
    local RUN_NAME=$4

    for STEP in 50 100 150 200; do
        local LORA_PATH="$OUTDIR/step_${STEP}"
        local EVAL_DIR="$OUTDIR/eval_step_${STEP}"

        [ ! -d "$LORA_PATH" ] && continue
        [ -f "$EVAL_DIR/summary.json" ] && echo "=== $RUN_NAME step $STEP eval exists, skipping ===" && continue

        local MERGED_PATH="$OUTDIR/_eval_merged_step_${STEP}"
        echo "=== Merging $RUN_NAME step $STEP ==="
        CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -c "
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

        echo "=== Evaluating $RUN_NAME step $STEP ($TASK) on GPU $GPU ==="
        eval_task "$GPU" "$TASK" "$MERGED_PATH" "$EVAL_DIR"

        rm -rf "$MERGED_PATH"
        echo "=== $RUN_NAME step $STEP eval done ==="
    done
}

train_and_eval() {
    local GPU=$1
    local TASK=$2
    local MODE=$3
    local DATASET=$4
    local SYS_PROMPT=$5
    local OUTDIR=$6
    local RUN_NAME=$7
    local EXTRA_ARGS=$8

    if [ -d "$OUTDIR/step_200" ]; then
        echo "=== $RUN_NAME training already done, skipping ==="
    else
        echo "=== Training $RUN_NAME on GPU $GPU ==="
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" on_policy_distill_positional.py \
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

    eval_checkpoints "$GPU" "$TASK" "$OUTDIR" "$RUN_NAME"
}

echo "========== Phase-1 Token Select (K=100, dual-GPU) =========="

# 1) Eval first (explicit request)
eval_checkpoints 0 "math" "checkpoints/token-select-k100-topkl-math" "token-select-k100-topkl-math"

# 2) Train remaining in parallel on two GPUs
train_and_eval 0 \
    "funcall" "top_kl" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
    "checkpoints/token-select-k100-topkl-funcall" "token-select-k100-topkl-funcall" \
    "$FUNCALL_EXTRA" &
PID_A=$!

train_and_eval 1 \
    "math" "top_entropy_student" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
    "checkpoints/token-select-k100-topent-student-math" "token-select-k100-topent-student-math" \
    "$MATH_EXTRA" &
PID_B=$!

wait "$PID_A" "$PID_B"

# 3) Last remaining run
train_and_eval 0 \
    "math" "top_entropy_teacher" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
    "checkpoints/token-select-k100-topent-teacher-math" "token-select-k100-topent-teacher-math" \
    "$MATH_EXTRA"

echo "=== Phase-1 K=100 dual-GPU experiments done ==="
