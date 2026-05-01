#!/bin/bash
# Gemma fullseq LoRA distillation: math + coding + agentic(funcall)
# Same settings as previous fullseq experiments, only model pair changed.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

[ -f "${HOME}/.bashrc" ] && PS1=nonempty source "${HOME}/.bashrc" 2>/dev/null || true

TEACHER="google/gemma-3-4b-it"
STUDENT="google/gemma-2-2b-it"

LR=5e-5
LORA_R=32
LORA_ALPHA=64

FULLSEQ_MATH_EXTRA="--use_vllm --max_new_tokens 2048 --teacher_micro_bs 4"
FULLSEQ_CODE_EXTRA="--use_vllm --max_new_tokens 512 --teacher_micro_bs 4"
FULLSEQ_AGENTIC_EXTRA="--use_vllm --max_new_tokens 512 --teacher_micro_bs 4"

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
AGENTIC_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

train_and_eval() {
    local TASK=$1
    local DATASET=$2
    local NUM_PROBLEMS=$3
    local STEPS=$4
    local SYS_PROMPT=$5
    local OUTDIR=$6
    local RUN_NAME=$7
    local EXTRA_ARGS=$8

    if [ -d "$OUTDIR/step_${STEPS}" ]; then
        echo "=== $RUN_NAME training already done, skipping ==="
    else
        echo "=== Training $RUN_NAME ==="
        CUDA_VISIBLE_DEVICES=0 python on_policy_distill_positional.py \
            --student_model "$STUDENT" --teacher_model "$TEACHER" \
            --dataset "$DATASET" --num_problems "$NUM_PROBLEMS" \
            --bs 16 --n_samples 1 \
            --temperature 0.7 --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
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
        CUDA_VISIBLE_DEVICES=0 python eval_math500.py \
            --model "$MODEL" --output_dir "$EVAL_DIR" \
            --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.50
    elif [ "$TASK" = "coding" ]; then
        mkdir -p "$EVAL_DIR"
        for DS in humaneval mbpp; do
            CUDA_VISIBLE_DEVICES=0 python scripts/eval_humaneval.py \
                --model "$MODEL" --dataset "$DS" \
                --output_dir "$EVAL_DIR" \
                --gpu_memory_utilization 0.50 --trust_remote_code
        done
        echo '{"status": "done"}' > "$EVAL_DIR/summary.json"
    elif [ "$TASK" = "agentic" ]; then
        CUDA_VISIBLE_DEVICES=0 python eval_funcall.py \
            --model "$MODEL" --output_dir "$EVAL_DIR" \
            --gpu_id 0 --gpu_memory_utilization 0.50 --categories "simple,multiple"
    fi
}

echo "========== Gemma fullseq: 2-2b-it student + 3-4b-it teacher =========="

train_and_eval "math" "AI-MO/NuminaMath-CoT" 3200 200 \
    "$MATH_SYS" "checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq" "scale-gemma2-2b-tgemma3-4b-math-fullseq" \
    "$FULLSEQ_MATH_EXTRA"

train_and_eval "coding" "coseal/CodeUltraFeedback_binarized" 3200 200 \
    "$CODING_SYS" "checkpoints/scale-gemma2-2b-tgemma3-4b-coding-fullseq" "scale-gemma2-2b-tgemma3-4b-coding-fullseq" \
    "$FULLSEQ_CODE_EXTRA --problem_field instruction --mini_bs 4"

# agentic task uses the existing function-calling dataset/eval pipeline used by previous fullseq runs
train_and_eval "agentic" "data/funcall/train.jsonl" 3200 200 \
    "$AGENTIC_SYS" "checkpoints/scale-gemma2-2b-tgemma3-4b-agentic-fullseq" "scale-gemma2-2b-tgemma3-4b-agentic-fullseq" \
    "$FULLSEQ_AGENTIC_EXTRA --problem_field problem"

echo "=== Gemma fullseq all done ==="
