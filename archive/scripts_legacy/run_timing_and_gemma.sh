#!/bin/bash
# Timing measurements (HF generate for both fullseq and pos-100) + Gemma pos-100
set -e
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/ziheng/quick-distillation
mkdir -p logs checkpoints logs/timing

LR=5e-5; LORA_R=32; LORA_ALPHA=64
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

S="Qwen/Qwen2.5-Math-1.5B"

# =============================================================================
# PART 1: Timing measurements — ALL use HF generate (fair comparison)
# =============================================================================
echo "========== Timing Measurements (all HF generate) =========="

for TEACHER_INFO in "Qwen/Qwen3-1.7B:t1.7b" "Qwen/Qwen3-4B:t4b" "Qwen/Qwen3-8B:t8b"; do
    IFS=: read -r TEACHER TAG <<< "$TEACHER_INFO"

    # Full-seq timing (HF generate, max_new_tokens=3584)
    NAME="timing-fullseq-${TAG}"
    if [ ! -f "logs/timing/${NAME}.done" ]; then
        echo "=== Timing: $NAME (HF generate, 3584 tokens) ==="
        python on_policy_distill_positional.py \
            --student_model "$S" --teacher_model "$TEACHER" \
            --dataset "AI-MO/NuminaMath-CoT" --num_problems 160 \
            --bs 16 --n_samples 1 --temperature 0.7 \
            --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
            --save_steps 999 --log_steps 1 --eval_steps 0 \
            --student_gpu 0 --teacher_gpu 0 \
            --system_prompt "$MATH_SYS" \
            --wandb_project dft-distill-timing \
            --output_dir "checkpoints/_timing_${NAME}" \
            --position_limit 0 \
            --max_new_tokens 3584 \
            --wandb_run_name "$NAME" \
            2>&1 | tee "logs/timing/${NAME}.log"
        touch "logs/timing/${NAME}.done"
        rm -rf "checkpoints/_timing_${NAME}"
    fi

    # Pos-100 timing (HF generate, max_new_tokens=100)
    NAME="timing-pos100-${TAG}"
    if [ ! -f "logs/timing/${NAME}.done" ]; then
        echo "=== Timing: $NAME (HF generate, 100 tokens) ==="
        python on_policy_distill_positional.py \
            --student_model "$S" --teacher_model "$TEACHER" \
            --dataset "AI-MO/NuminaMath-CoT" --num_problems 160 \
            --bs 16 --n_samples 1 --temperature 0.7 \
            --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
            --save_steps 999 --log_steps 1 --eval_steps 0 \
            --student_gpu 0 --teacher_gpu 0 \
            --system_prompt "$MATH_SYS" \
            --wandb_project dft-distill-timing \
            --output_dir "checkpoints/_timing_${NAME}" \
            --position_limit 100 \
            --wandb_run_name "$NAME" \
            2>&1 | tee "logs/timing/${NAME}.log"
        touch "logs/timing/${NAME}.done"
        rm -rf "checkpoints/_timing_${NAME}"
    fi
done

echo "========== Timing Done =========="

# =============================================================================
# PART 2: Gemma pos-100 (HF generate)
# =============================================================================
echo "========== Gemma Pos-100 =========="

GS="google/gemma-2-2b-it"
GT="google/gemma-3-4b-it"

train_run() {
    local SMODEL=$1 TEACHER=$2 DATASET=$3 SYS_PROMPT=$4 OUTDIR=$5 RUN_NAME=$6 EXTRA=$7
    [ -d "$OUTDIR/step_200" ] && echo "=== $RUN_NAME done, skip ===" && return
    echo "=== Training $RUN_NAME ==="
    python on_policy_distill_positional.py \
        --student_model "$SMODEL" --teacher_model "$TEACHER" \
        --dataset "$DATASET" --num_problems 3200 \
        --bs 16 --n_samples 1 --temperature 0.7 \
        --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
        --save_steps 50 --log_steps 10 --eval_steps 0 \
        --student_gpu 0 --teacher_gpu 0 \
        --system_prompt "$SYS_PROMPT" \
        --wandb_project dft-distill-gemma \
        --output_dir "$OUTDIR" \
        --wandb_run_name "$RUN_NAME" \
        $EXTRA \
        2>&1 | tee "logs/${RUN_NAME}.log"
    echo "=== $RUN_NAME done ==="
}

eval_checkpoints() {
    local SMODEL=$1 TASK=$2 OUTDIR=$3 RUN_NAME=$4
    for STEP in 50 100 150 200; do
        local LP="$OUTDIR/step_${STEP}" ED="$OUTDIR/eval_step_${STEP}"
        [ ! -d "$LP" ] && continue
        [ -f "$ED/summary.json" ] && echo "=== $RUN_NAME s$STEP eval exists ===" && continue
        local MP="$OUTDIR/_eval_merged_step_${STEP}"
        echo "=== Merge+eval $RUN_NAME s$STEP ==="
        CUDA_VISIBLE_DEVICES="" python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer; from peft import PeftModel; import torch
b=AutoModelForCausalLM.from_pretrained('$SMODEL',torch_dtype=torch.bfloat16)
m=PeftModel.from_pretrained(b,'$LP').merge_and_unload()
m.save_pretrained('$MP'); AutoTokenizer.from_pretrained('$SMODEL',trust_remote_code=True).save_pretrained('$MP'); print('Merged')
"
        eval_task "$TASK" "$MP" "$ED"
        rm -rf "$MP"
    done
}

eval_task() {
    local TASK=$1 MODEL=$2 EVAL_DIR=$3
    if [ "$TASK" = "math" ]; then
        python eval_math500.py --model "$MODEL" --output_dir "$EVAL_DIR" \
            --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.70
    elif [ "$TASK" = "coding" ]; then
        mkdir -p "$EVAL_DIR"
        for DS in humaneval mbpp; do
            python scripts/eval_humaneval.py --model "$MODEL" --dataset $DS \
                --output_dir "$EVAL_DIR" --gpu_memory_utilization 0.70 --trust_remote_code
        done
        echo '{"status":"done"}' > "$EVAL_DIR/summary.json"
    elif [ "$TASK" = "funcall" ]; then
        python eval_funcall.py --model "$MODEL" --output_dir "$EVAL_DIR" \
            --gpu_id 0 --gpu_memory_utilization 0.70 --categories "simple,multiple"
    fi
}

eval_baseline() {
    local MODEL=$1 NAME=$2 BASE="checkpoints/${NAME}-baseline"
    for TASK in math coding funcall; do
        ED="$BASE/eval_${TASK}"
        [ -f "$ED/summary.json" ] && continue
        mkdir -p "$ED"
        echo "=== Baseline $NAME $TASK ==="
        eval_task "$TASK" "$MODEL" "$ED"
    done
}

# Gemma baseline eval
eval_baseline "$GS" "gemma"

# Gemma pos-100: math, coding, funcall
train_run "$GS" "$GT" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
    "checkpoints/gemma-pos100-math" "gemma-pos100-math" "--position_limit 100"
eval_checkpoints "$GS" "math" "checkpoints/gemma-pos100-math" "gemma-pos100-math"

train_run "$GS" "$GT" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
    "checkpoints/gemma-pos100-coding" "gemma-pos100-coding" "--position_limit 100 --problem_field instruction"
eval_checkpoints "$GS" "coding" "checkpoints/gemma-pos100-coding" "gemma-pos100-coding"

train_run "$GS" "$GT" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
    "checkpoints/gemma-pos100-funcall" "gemma-pos100-funcall" "--position_limit 100 --problem_field problem"
eval_checkpoints "$GS" "funcall" "checkpoints/gemma-pos100-funcall" "gemma-pos100-funcall"

echo "=== ALL DONE ==="
