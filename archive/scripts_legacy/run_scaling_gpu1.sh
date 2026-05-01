#!/bin/bash
# Model size scaling experiments — GPU 1 only (student+teacher on cuda:0 of this process = physical GPU 1)
# Teacher: Qwen3-8B
# Configs: D (Qwen3-1.7B student) + B (Math-1.5B student) + E (Qwen3-4B student)
# Tasks: Math, Coding, Funcall
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

[ -f "${HOME}/.bashrc" ] && PS1=nonempty source "${HOME}/.bashrc" 2>/dev/null || true
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hf_models_env.sh"

TEACHER="$QWEN3_8"
POS=100
LR=5e-5
LORA_R=32
LORA_ALPHA=64

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

# ─── Helper functions ───

train_and_eval() {
    local STUDENT=$1
    local TASK=$2
    local DATASET=$3
    local NUM_PROBLEMS=$4
    local STEPS=$5
    local SYS_PROMPT=$6
    local OUTDIR=$7
    local RUN_NAME=$8
    local EXTRA_ARGS=$9

    # Train
    if [ -d "$OUTDIR/step_${STEPS}" ]; then
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
            --position_limit $POS \
            --wandb_run_name "$RUN_NAME" \
            $EXTRA_ARGS \
            2>&1 | tee "logs/${RUN_NAME}.log"
        echo "=== $RUN_NAME training done ==="
    fi

    # Eval
    eval_checkpoints "$STUDENT" "$TASK" "$OUTDIR" "$RUN_NAME"
}

eval_checkpoints() {
    local STUDENT=$1
    local TASK=$2
    local OUTDIR=$3
    local RUN_NAME=$4

    for STEP in 50 100 150 200; do
        local LORA_PATH="$OUTDIR/step_${STEP}"
        local EVAL_DIR="$OUTDIR/eval_step_${STEP}"

        [ ! -d "$LORA_PATH" ] && continue
        [ -f "$EVAL_DIR/summary.json" ] && echo "=== $RUN_NAME step $STEP eval exists, skipping ===" && continue

        # Merge LoRA
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
    elif [ "$TASK" = "coding" ]; then
        mkdir -p "$EVAL_DIR"
        for DS in humaneval mbpp; do
            CUDA_VISIBLE_DEVICES=1 python scripts/eval_humaneval.py \
                --model "$MODEL" --dataset $DS \
                --output_dir "$EVAL_DIR" \
                --gpu_memory_utilization 0.50 --trust_remote_code
        done
        echo '{"status": "done"}' > "$EVAL_DIR/summary.json"
    elif [ "$TASK" = "funcall" ]; then
        CUDA_VISIBLE_DEVICES=1 python eval_funcall.py \
            --model "$MODEL" --output_dir "$EVAL_DIR" \
            --gpu_id 0 --gpu_memory_utilization 0.50 --categories "simple,multiple"
    fi
}

# Baseline eval (Qwen3-8B, Qwen3-1.7B) — run separately if needed; omitted here.

# ─── Config D: Qwen3-1.7B student + Qwen3-8B teacher ───
echo "========== Config D: Qwen3-1.7B + Qwen3-8B =========="
D_STUDENT="$QWEN3_17"

train_and_eval "$D_STUDENT" "math" "AI-MO/NuminaMath-CoT" 3200 200 \
    "$MATH_SYS" "checkpoints/scale-q1.7b-t8b-math-pos100" "scale-q1.7b-t8b-math"

train_and_eval "$D_STUDENT" "coding" "coseal/CodeUltraFeedback_binarized" 3200 200 \
    "$CODING_SYS" "checkpoints/scale-q1.7b-t8b-coding-pos100" "scale-q1.7b-t8b-coding" \
    "--problem_field instruction --mini_bs 4"

train_and_eval "$D_STUDENT" "funcall" "data/funcall/train.jsonl" 3200 200 \
    "$FUNCALL_SYS" "checkpoints/scale-q1.7b-t8b-funcall-pos100" "scale-q1.7b-t8b-funcall" \
    "--problem_field problem"

# ─── Config B: Math-1.5B student + Qwen3-8B teacher ───
echo "========== Config B: Math-1.5B + Qwen3-8B =========="
B_STUDENT="$MATH_STUDENT_15"

train_and_eval "$B_STUDENT" "math" "AI-MO/NuminaMath-CoT" 3200 200 \
    "$MATH_SYS" "checkpoints/scale-m1.5b-t8b-math-pos100" "scale-m1.5b-t8b-math"

train_and_eval "$B_STUDENT" "coding" "coseal/CodeUltraFeedback_binarized" 3200 200 \
    "$CODING_SYS" "checkpoints/scale-m1.5b-t8b-coding-pos100" "scale-m1.5b-t8b-coding" \
    "--problem_field instruction --mini_bs 4"

train_and_eval "$B_STUDENT" "funcall" "data/funcall/train.jsonl" 3200 200 \
    "$FUNCALL_SYS" "checkpoints/scale-m1.5b-t8b-funcall-pos100" "scale-m1.5b-t8b-funcall" \
    "--problem_field problem"

# ─── Config E: Qwen3-4B student + Qwen3-8B teacher ───
echo "========== Config E: Qwen3-4B + Qwen3-8B =========="
E_STUDENT="$QWEN3_4"

train_and_eval "$E_STUDENT" "math" "AI-MO/NuminaMath-CoT" 3200 200 \
    "$MATH_SYS" "checkpoints/scale-q4b-t8b-math-pos100" "scale-q4b-t8b-math" \
    "--mini_bs 4"

train_and_eval "$E_STUDENT" "coding" "coseal/CodeUltraFeedback_binarized" 3200 200 \
    "$CODING_SYS" "checkpoints/scale-q4b-t8b-coding-pos100" "scale-q4b-t8b-coding" \
    "--problem_field instruction --mini_bs 4"

train_and_eval "$E_STUDENT" "funcall" "data/funcall/train.jsonl" 3200 200 \
    "$FUNCALL_SYS" "checkpoints/scale-q4b-t8b-funcall-pos100" "scale-q4b-t8b-funcall" \
    "--problem_field problem --mini_bs 4"

echo "=== GPU 1 ALL DONE ==="
