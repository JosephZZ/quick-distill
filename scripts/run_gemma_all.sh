#!/bin/bash
# Gemma cross-family: gemma-2-2b-it → gemma-3-4b-it (with vocab mapping)
# Runs on both GPUs: pos-100 on GPU 0, fullseq on GPU 1
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/ziheng/quick-distillation
mkdir -p logs checkpoints

LR=5e-5; LORA_R=32; LORA_ALPHA=64
GS="google/gemma-2-2b-it"
GT="google/gemma-3-4b-it"

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

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
        --mini_bs 1 \
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

# ============================================================
# GPU 0: Gemma pos-100 (math + coding, funcall already done)
# ============================================================
(
    export CUDA_VISIBLE_DEVICES=0

    # Baseline eval
    eval_baseline "$GS" "gemma"

    # Pos-100 math (re-run with mini_bs 1)
    train_run "$GS" "$GT" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
        "checkpoints/gemma-pos100-math" "gemma-pos100-math" "--position_limit 100"
    eval_checkpoints "$GS" "math" "checkpoints/gemma-pos100-math" "gemma-pos100-math"

    # Pos-100 coding (re-run with mini_bs 1)
    train_run "$GS" "$GT" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
        "checkpoints/gemma-pos100-coding" "gemma-pos100-coding" "--position_limit 100 --problem_field instruction"
    eval_checkpoints "$GS" "coding" "checkpoints/gemma-pos100-coding" "gemma-pos100-coding"

    # Pos-100 funcall eval (training already done)
    eval_checkpoints "$GS" "funcall" "checkpoints/gemma-pos100-funcall" "gemma-pos100-funcall"

    echo "=== GPU 0 DONE ==="
) > logs/gemma_gpu0.log 2>&1 &
PID0=$!

# ============================================================
# GPU 1: Gemma full-seq
# ============================================================
(
    export CUDA_VISIBLE_DEVICES=1

    # Full-seq math
    train_run "$GS" "$GT" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
        "checkpoints/gemma-fullseq-math" "gemma-fullseq-math" \
        "--position_limit 0 --use_vllm --max_new_tokens 3584 --vllm_gpu_util 0.50"
    eval_checkpoints "$GS" "math" "checkpoints/gemma-fullseq-math" "gemma-fullseq-math"

    # Full-seq coding
    train_run "$GS" "$GT" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
        "checkpoints/gemma-fullseq-coding" "gemma-fullseq-coding" \
        "--position_limit 0 --use_vllm --max_new_tokens 3584 --vllm_gpu_util 0.50 --problem_field instruction"
    eval_checkpoints "$GS" "coding" "checkpoints/gemma-fullseq-coding" "gemma-fullseq-coding"

    # Full-seq funcall
    train_run "$GS" "$GT" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
        "checkpoints/gemma-fullseq-funcall" "gemma-fullseq-funcall" \
        "--position_limit 0 --use_vllm --max_new_tokens 3584 --vllm_gpu_util 0.50 --problem_field problem"
    eval_checkpoints "$GS" "funcall" "checkpoints/gemma-fullseq-funcall" "gemma-fullseq-funcall"

    echo "=== GPU 1 DONE ==="
) > logs/gemma_gpu1.log 2>&1 &
PID1=$!

echo "GPU 0 PID: $PID0 (pos-100 math+coding + evals)"
echo "GPU 1 PID: $PID1 (fullseq math+coding+funcall)"
echo "Logs: logs/gemma_gpu0.log, logs/gemma_gpu1.log"

wait $PID0; echo "GPU 0 finished"
wait $PID1; echo "GPU 1 finished"
echo "=== ALL GEMMA DONE ==="
