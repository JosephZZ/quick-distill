#!/bin/bash
# =============================================================================
# All remaining experiments for the paper
# Designed to run on a fresh machine with 2 GPUs (48GB each)
# Uses relative paths — run from the repo root directory
#
# Requirements:
#   pip install torch transformers peft datasets wandb vllm==0.6.6.post1 \
#               evalplus latex2sympy2 antlr4-python3-runtime==4.7.2
#   (or use sglang[all] instead of vllm for faster full-seq generation)
#
# Usage:
#   cd <repo_root>
#   bash scripts/run_all_experiments.sh
# =============================================================================
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

# ─── Config ───
LR=5e-5; LORA_R=32; LORA_ALPHA=64
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

# ─── Inference backend: use sglang if available, else vllm ───
FULLSEQ_BACKEND="--use_vllm --vllm_gpu_util 0.50"
python -c "from sglang import Engine; import sgl_kernel" 2>/dev/null && \
    FULLSEQ_BACKEND="--use_sglang --vllm_gpu_util 0.30" && echo "Using SGLang backend" || \
    echo "Using vLLM backend"

# =============================================================================
# Helper functions
# =============================================================================

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
        --wandb_project dft-distill \
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

# =============================================================================
# GROUP 1: Qwen full-seq baselines (fill Table 1 full-seq rows)
# Student: Qwen2.5-Math-1.5B, LoRA
# These run on GPU 0 and GPU 1 in parallel via the _gpu0/_gpu1 wrapper scripts
# =============================================================================

run_qwen_fullseq() {
    local GPU=$1 TEACHER=$2 TAG=$3
    export CUDA_VISIBLE_DEVICES=$GPU
    local S="Qwen/Qwen2.5-Math-1.5B"

    # Math
    train_run "$S" "$TEACHER" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
        "checkpoints/fullseq-m1.5b-${TAG}-math" "fullseq-m1.5b-${TAG}-math" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1"
    eval_checkpoints "$S" "math" "checkpoints/fullseq-m1.5b-${TAG}-math" "fullseq-m1.5b-${TAG}-math"

    # Coding
    train_run "$S" "$TEACHER" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
        "checkpoints/fullseq-m1.5b-${TAG}-coding" "fullseq-m1.5b-${TAG}-coding" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 --problem_field instruction"
    eval_checkpoints "$S" "coding" "checkpoints/fullseq-m1.5b-${TAG}-coding" "fullseq-m1.5b-${TAG}-coding"

    # Funcall
    train_run "$S" "$TEACHER" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
        "checkpoints/fullseq-m1.5b-${TAG}-funcall" "fullseq-m1.5b-${TAG}-funcall" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 --problem_field problem"
    eval_checkpoints "$S" "funcall" "checkpoints/fullseq-m1.5b-${TAG}-funcall" "fullseq-m1.5b-${TAG}-funcall"
}

# =============================================================================
# GROUP 2: Gemma cross-family (fill Table 1 Gemma row)
# Student: Gemma-2-2b, Teacher: Gemma-3-4b-pt
# =============================================================================

run_gemma() {
    local GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU
    local GS="google/gemma-2-2b-it"
    local GT="google/gemma-2-9b-it"

    # Baseline eval
    eval_baseline "$GS" "gemma"

    # Full-seq (3 tasks)
    for TASK_INFO in "math:AI-MO/NuminaMath-CoT:$MATH_SYS:problem" \
                     "coding:coseal/CodeUltraFeedback_binarized:$CODING_SYS:instruction" \
                     "funcall:data/funcall/train.jsonl:$FUNCALL_SYS:problem"; do
        IFS=: read -r TASK DATASET SYS FIELD <<< "$TASK_INFO"
        local EXTRA="$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1"
        [ "$FIELD" != "problem" ] && EXTRA="$EXTRA --problem_field $FIELD"
        train_run "$GS" "$GT" "$DATASET" "$SYS" \
            "checkpoints/gemma-fullseq-${TASK}" "gemma-fullseq-${TASK}" "$EXTRA"
        eval_checkpoints "$GS" "$TASK" "checkpoints/gemma-fullseq-${TASK}" "gemma-fullseq-${TASK}"
    done

    # Pos-100 (3 tasks) — uses HF generate, no vLLM/sglang needed
    for TASK_INFO in "math:AI-MO/NuminaMath-CoT:$MATH_SYS:problem" \
                     "coding:coseal/CodeUltraFeedback_binarized:$CODING_SYS:instruction" \
                     "funcall:data/funcall/train.jsonl:$FUNCALL_SYS:problem"; do
        IFS=: read -r TASK DATASET SYS FIELD <<< "$TASK_INFO"
        local EXTRA="--position_limit 100"
        [ "$FIELD" != "problem" ] && EXTRA="$EXTRA --problem_field $FIELD"
        train_run "$GS" "$GT" "$DATASET" "$SYS" \
            "checkpoints/gemma-pos100-${TASK}" "gemma-pos100-${TASK}" "$EXTRA"
        eval_checkpoints "$GS" "$TASK" "checkpoints/gemma-pos100-${TASK}" "gemma-pos100-${TASK}"
    done
}

# =============================================================================
# GROUP 3: Timing measurements (fill Table 1 Time/Memory columns)
# 10-step timed runs for each config on math task
# =============================================================================

run_timing() {
    local GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU
    local S="Qwen/Qwen2.5-Math-1.5B"
    mkdir -p logs/timing

    for TEACHER_INFO in "Qwen/Qwen3-1.7B:t1.7b" "Qwen/Qwen3-4B:t4b" "Qwen/Qwen3-8B:t8b"; do
        IFS=: read -r TEACHER TAG <<< "$TEACHER_INFO"

        # Full-seq timing (10 steps)
        local NAME="timing-fullseq-${TAG}"
        if [ ! -f "logs/timing/${NAME}.done" ]; then
            echo "=== Timing: $NAME ==="
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
                $FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 \
                --wandb_run_name "$NAME" \
                2>&1 | tee "logs/timing/${NAME}.log"
            touch "logs/timing/${NAME}.done"
            rm -rf "checkpoints/_timing_${NAME}"
        fi

        # Pos-100 timing (10 steps, HF generate)
        NAME="timing-pos100-${TAG}"
        if [ ! -f "logs/timing/${NAME}.done" ]; then
            echo "=== Timing: $NAME ==="
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
}

# =============================================================================
# MAIN: Launch GPU 0 and GPU 1 in parallel
# =============================================================================

echo "=========================================="
echo "Starting all experiments (2 GPUs parallel)"
echo "=========================================="

# GPU 0: 8B full-seq + timing
(
    run_qwen_fullseq 0 "Qwen/Qwen3-8B" "t8b"
    run_timing 0
    echo "=== GPU 0 ALL DONE ==="
) > logs/gpu0_master.log 2>&1 &
PID0=$!

# GPU 1: 4B full-seq + Gemma
(
    run_qwen_fullseq 1 "Qwen/Qwen3-4B" "t4b"
    run_gemma 1
    echo "=== GPU 1 ALL DONE ==="
) > logs/gpu1_master.log 2>&1 &
PID1=$!

echo "GPU 0 PID: $PID0 (8B fullseq + timing)"
echo "GPU 1 PID: $PID1 (4B fullseq + Gemma)"
echo "Logs: logs/gpu0_master.log, logs/gpu1_master.log"
echo "Monitor: tail -f logs/gpu0_master.log logs/gpu1_master.log"

# Wait for both
wait $PID0
echo "GPU 0 finished (exit code: $?)"
wait $PID1
echo "GPU 1 finished (exit code: $?)"

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
