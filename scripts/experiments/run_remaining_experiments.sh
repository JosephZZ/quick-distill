#!/bin/bash
# =============================================================================
# Remaining experiments NOT yet running on the local machine
# Local machine is running: 4B/8B math fullseq (will finish there)
# This script covers: 4B/8B coding+funcall fullseq + all Gemma experiments
#
# Designed for a remote server with 2 GPUs (48GB+ each)
# Uses relative paths — run from the repo root directory
#
# Requirements:
#   pip install torch transformers peft datasets wandb evalplus \
#               latex2sympy2 antlr4-python3-runtime==4.7.2
#   pip install sglang[all]  # preferred for full-seq (faster than vllm)
#   # OR: pip install vllm==0.6.6.post1  # fallback
#
# Usage:
#   cd <repo_root>
#   bash scripts/run_remaining_experiments.sh
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

# ─── Auto-detect inference backend ───
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
# GPU 0: Qwen3-8B full-seq coding + funcall
# GPU 1: Qwen3-4B full-seq coding + funcall + all Gemma
# =============================================================================

echo "=========================================="
echo "Starting remaining experiments (2 GPUs)"
echo "=========================================="

# --- GPU 0: 8B math + coding + funcall ---
(
    export CUDA_VISIBLE_DEVICES=0
    S="Qwen/Qwen2.5-Math-1.5B"
    T="Qwen/Qwen3-8B"

    # Math (re-run needed — previous run had too many OOM skips, only step_50 saved)
    train_run "$S" "$T" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
        "checkpoints/fullseq-m1.5b-t8b-math" "fullseq-m1.5b-t8b-math" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1"
    eval_checkpoints "$S" "math" "checkpoints/fullseq-m1.5b-t8b-math" "fullseq-m1.5b-t8b-math"

    # Coding
    train_run "$S" "$T" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
        "checkpoints/fullseq-m1.5b-t8b-coding" "fullseq-m1.5b-t8b-coding" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 --problem_field instruction"
    eval_checkpoints "$S" "coding" "checkpoints/fullseq-m1.5b-t8b-coding" "fullseq-m1.5b-t8b-coding"

    # Funcall
    train_run "$S" "$T" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
        "checkpoints/fullseq-m1.5b-t8b-funcall" "fullseq-m1.5b-t8b-funcall" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 --problem_field problem"
    eval_checkpoints "$S" "funcall" "checkpoints/fullseq-m1.5b-t8b-funcall" "fullseq-m1.5b-t8b-funcall"

    echo "=== GPU 0 DONE ==="
) > logs/gpu0_remaining.log 2>&1 &
PID0=$!

# --- GPU 1: 4B coding + funcall + Gemma ---
(
    export CUDA_VISIBLE_DEVICES=1
    S="Qwen/Qwen2.5-Math-1.5B"
    T="Qwen/Qwen3-4B"

    # 4B Coding
    train_run "$S" "$T" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
        "checkpoints/fullseq-m1.5b-t4b-coding" "fullseq-m1.5b-t4b-coding" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 --problem_field instruction"
    eval_checkpoints "$S" "coding" "checkpoints/fullseq-m1.5b-t4b-coding" "fullseq-m1.5b-t4b-coding"

    # 4B Funcall
    train_run "$S" "$T" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
        "checkpoints/fullseq-m1.5b-t4b-funcall" "fullseq-m1.5b-t4b-funcall" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 --problem_field problem"
    eval_checkpoints "$S" "funcall" "checkpoints/fullseq-m1.5b-t4b-funcall" "fullseq-m1.5b-t4b-funcall"

    # ─── Gemma cross-family ───
    GS="google/gemma-2-2b"
    GT="google/gemma-3-4b-pt"

    # Gemma baseline eval
    eval_baseline "$GS" "gemma"

    # Gemma full-seq (3 tasks)
    train_run "$GS" "$GT" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
        "checkpoints/gemma-fullseq-math" "gemma-fullseq-math" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1"
    eval_checkpoints "$GS" "math" "checkpoints/gemma-fullseq-math" "gemma-fullseq-math"

    train_run "$GS" "$GT" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
        "checkpoints/gemma-fullseq-coding" "gemma-fullseq-coding" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 --problem_field instruction"
    eval_checkpoints "$GS" "coding" "checkpoints/gemma-fullseq-coding" "gemma-fullseq-coding"

    train_run "$GS" "$GT" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
        "checkpoints/gemma-fullseq-funcall" "gemma-fullseq-funcall" \
        "$FULLSEQ_BACKEND --max_new_tokens 3584 --mini_bs 1 --problem_field problem"
    eval_checkpoints "$GS" "funcall" "checkpoints/gemma-fullseq-funcall" "gemma-fullseq-funcall"

    # Gemma pos-100 (3 tasks) — HF generate, no vLLM/sglang
    train_run "$GS" "$GT" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
        "checkpoints/gemma-pos100-math" "gemma-pos100-math" \
        "--position_limit 100"
    eval_checkpoints "$GS" "math" "checkpoints/gemma-pos100-math" "gemma-pos100-math"

    train_run "$GS" "$GT" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
        "checkpoints/gemma-pos100-coding" "gemma-pos100-coding" \
        "--position_limit 100 --problem_field instruction"
    eval_checkpoints "$GS" "coding" "checkpoints/gemma-pos100-coding" "gemma-pos100-coding"

    train_run "$GS" "$GT" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
        "checkpoints/gemma-pos100-funcall" "gemma-pos100-funcall" \
        "--position_limit 100 --problem_field problem"
    eval_checkpoints "$GS" "funcall" "checkpoints/gemma-pos100-funcall" "gemma-pos100-funcall"

    echo "=== GPU 1 DONE ==="
) > logs/gpu1_remaining.log 2>&1 &
PID1=$!

echo "GPU 0 PID: $PID0 (8B coding + funcall)"
echo "GPU 1 PID: $PID1 (4B coding + funcall + Gemma)"
echo "Logs: logs/gpu0_remaining.log, logs/gpu1_remaining.log"

wait $PID0; echo "GPU 0 finished (exit $?)"
wait $PID1; echo "GPU 1 finished (exit $?)"

echo "=========================================="
echo "ALL REMAINING EXPERIMENTS COMPLETE"
echo "=========================================="
