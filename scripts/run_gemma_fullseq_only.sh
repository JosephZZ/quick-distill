#!/bin/bash
# Run ONLY the fullseq phase (pos-100 already complete)
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

cd /mnt/ziheng/quick-distillation
mkdir -p logs checkpoints

LR=5e-5; LORA_R=32; LORA_ALPHA=64
GS="google/gemma-2-2b-it"
GT="google/gemma-3-4b-it"

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

train_run() {
    local SMODEL=$1 TEACHER=$2 DATASET=$3 SYS_PROMPT=$4 OUTDIR=$5 RUN_NAME=$6 SGPU=$7 TGPU=$8 EXTRA=$9
    [ -d "$OUTDIR/step_200" ] && echo "=== $RUN_NAME done, skip ===" && return
    echo "=== Training $RUN_NAME (student=GPU${SGPU}, teacher=GPU${TGPU}) ==="
    python on_policy_distill_positional.py \
        --student_model "$SMODEL" --teacher_model "$TEACHER" \
        --dataset "$DATASET" --num_problems 3200 \
        --bs 16 --n_samples 1 --temperature 0.7 \
        --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
        --save_steps 25 --log_steps 10 --eval_steps 0 \
        --student_gpu $SGPU --teacher_gpu $TGPU \
        --system_prompt "$SYS_PROMPT" \
        --wandb_project dft-distill-gemma \
        --output_dir "$OUTDIR" \
        --wandb_run_name "$RUN_NAME" \
        --mini_bs 1 \
        $EXTRA \
        2>&1 | tee "logs/${RUN_NAME}.log"
    echo "=== $RUN_NAME done ==="
}

eval_task() {
    local TASK=$1 MODEL=$2 EVAL_DIR=$3 EVAL_GPU=$4
    if [ "$TASK" = "math" ]; then
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python eval_math500.py --model "$MODEL" --output_dir "$EVAL_DIR" \
            --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.70
    elif [ "$TASK" = "coding" ]; then
        mkdir -p "$EVAL_DIR"
        for DS in humaneval mbpp; do
            CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/eval_humaneval.py --model "$MODEL" --dataset $DS \
                --output_dir "$EVAL_DIR" --gpu_memory_utilization 0.70 --trust_remote_code
        done
        echo '{"status":"done"}' > "$EVAL_DIR/summary.json"
    elif [ "$TASK" = "funcall" ]; then
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python eval_funcall.py --model "$MODEL" --output_dir "$EVAL_DIR" \
            --gpu_id 0 --gpu_memory_utilization 0.70 --categories "simple,multiple"
    fi
}

eval_checkpoints() {
    local SMODEL=$1 TASK=$2 OUTDIR=$3 RUN_NAME=$4 EVAL_GPU=$5
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
        eval_task "$TASK" "$MP" "$ED" "$EVAL_GPU"
        rm -rf "$MP"
    done
}

echo "=== PHASE 2: fullseq (student=GPU1, teacher=GPU0) ==="

# Full-seq math
train_run "$GS" "$GT" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
    "checkpoints/gemma-fullseq-math" "gemma-fullseq-math" 1 0 \
    "--position_limit 0 --use_vllm --max_new_tokens 3584 --vllm_gpu_util 0.70"
eval_checkpoints "$GS" "math" "checkpoints/gemma-fullseq-math" "gemma-fullseq-math" 1

# Full-seq coding
train_run "$GS" "$GT" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
    "checkpoints/gemma-fullseq-coding" "gemma-fullseq-coding" 1 0 \
    "--position_limit 0 --use_vllm --max_new_tokens 3584 --vllm_gpu_util 0.70 --problem_field instruction"
eval_checkpoints "$GS" "coding" "checkpoints/gemma-fullseq-coding" "gemma-fullseq-coding" 1

# Full-seq funcall
train_run "$GS" "$GT" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
    "checkpoints/gemma-fullseq-funcall" "gemma-fullseq-funcall" 1 0 \
    "--position_limit 0 --use_vllm --max_new_tokens 3584 --vllm_gpu_util 0.70 --problem_field problem"
eval_checkpoints "$GS" "funcall" "checkpoints/gemma-fullseq-funcall" "gemma-fullseq-funcall" 1

echo "=== ALL GEMMA FULLSEQ DONE ==="
