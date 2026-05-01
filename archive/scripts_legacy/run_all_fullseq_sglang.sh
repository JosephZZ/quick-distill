#!/bin/bash
# Run ALL remaining fullseq experiments with SGLang
# Uses conda Python 3.10 env with sglang 0.5.9
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/usr/local/cuda-12.6/bin:/mnt/ziheng/conda-sglang/bin:$PATH
export CC=/mnt/ziheng/conda-sglang/bin/gcc
export CXX=/mnt/ziheng/conda-sglang/bin/g++
export FLASHINFER_CACHE_DIR=/mnt/ziheng/.cache/flashinfer
export TMPDIR=/mnt/ziheng/tmp
export PATH=/usr/local/cuda-12.6/bin:/mnt/ziheng/conda-sglang/bin:/home/ziheng/.local/bin:$PATH
export HF_HOME=/mnt/ziheng/.cache/huggingface

PY=/mnt/ziheng/conda-sglang/bin/python
cd /mnt/ziheng/quick-distillation
mkdir -p logs checkpoints

SM="Qwen/Qwen2.5-Math-1.5B"
LR=5e-5; LORA_R=32; LORA_ALPHA=64

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

train_fullseq() {
    local TEACHER=$1 DATASET=$2 SYS_PROMPT=$3 OUTDIR=$4 RUN_NAME=$5 EXTRA=$6
    [ -d "$OUTDIR/step_200" ] && echo "=== $RUN_NAME done, skip ===" && return
    echo "=== Training $RUN_NAME (student=GPU1, teacher=GPU0, sglang=GPU0) ==="
    $PY on_policy_distill_positional.py \
        --student_model "$SM" --teacher_model "$TEACHER" \
        --dataset "$DATASET" --num_problems 3200 \
        --bs 16 --n_samples 1 --temperature 0.7 \
        --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
        --save_steps 25 --log_steps 10 --eval_steps 0 \
        --student_gpu 1 --teacher_gpu 0 \
        --system_prompt "$SYS_PROMPT" \
        --wandb_project dft-distill-scaling \
        --output_dir "$OUTDIR" \
        --wandb_run_name "$RUN_NAME" \
        --mini_bs 1 \
        --position_limit 0 --use_sglang --sglang_gpu 0 --vllm_gpu_util 0.50 --max_new_tokens 3584 \
        $EXTRA \
        2>&1 | tee "logs/${RUN_NAME}.log"
    echo "=== $RUN_NAME done ==="
}

eval_math() {
    local OUTDIR=$1 RUN_NAME=$2
    for STEP in 25 50 75 100 125 150 175 200; do
        local LP="$OUTDIR/step_${STEP}" ED="$OUTDIR/eval_step_${STEP}"
        [ ! -d "$LP" ] && continue
        [ -f "$ED/summary.json" ] && echo "=== $RUN_NAME s$STEP eval exists ===" && continue
        local MP="$OUTDIR/_eval_merged_step_${STEP}"
        echo "=== Merge+eval math $RUN_NAME s$STEP ==="
        CUDA_VISIBLE_DEVICES="" $PY -c "
from transformers import AutoModelForCausalLM, AutoTokenizer; from peft import PeftModel; import torch
b=AutoModelForCausalLM.from_pretrained('$SM',torch_dtype=torch.bfloat16)
m=PeftModel.from_pretrained(b,'$LP').merge_and_unload()
m.save_pretrained('$MP'); AutoTokenizer.from_pretrained('$SM',trust_remote_code=True).save_pretrained('$MP'); print('Merged')
"
        CUDA_VISIBLE_DEVICES=1 $PY eval_math500.py --model "$MP" --output_dir "$ED" \
            --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.70
        rm -rf "$MP"
    done
}

eval_coding() {
    local OUTDIR=$1 RUN_NAME=$2
    for STEP in 25 50 75 100 125 150 175 200; do
        local LP="$OUTDIR/step_${STEP}" ED="$OUTDIR/eval_step_${STEP}"
        [ ! -d "$LP" ] && continue
        [ -f "$ED/summary.json" ] && continue
        local MP="$OUTDIR/_eval_merged_step_${STEP}"
        echo "=== Merge+eval coding $RUN_NAME s$STEP ==="
        CUDA_VISIBLE_DEVICES="" $PY -c "
from transformers import AutoModelForCausalLM, AutoTokenizer; from peft import PeftModel; import torch
b=AutoModelForCausalLM.from_pretrained('$SM',torch_dtype=torch.bfloat16)
m=PeftModel.from_pretrained(b,'$LP').merge_and_unload()
m.save_pretrained('$MP'); AutoTokenizer.from_pretrained('$SM',trust_remote_code=True).save_pretrained('$MP'); print('Merged')
"
        mkdir -p "$ED"
        for DS in humaneval mbpp; do
            CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_humaneval.py --model "$MP" --dataset $DS \
                --output_dir "$ED" --gpu_memory_utilization 0.70 --trust_remote_code
        done
        echo '{"status":"done"}' > "$ED/summary.json"
        rm -rf "$MP"
    done
}

eval_funcall() {
    local OUTDIR=$1 RUN_NAME=$2
    for STEP in 25 50 75 100 125 150 175 200; do
        local LP="$OUTDIR/step_${STEP}" ED="$OUTDIR/eval_step_${STEP}"
        [ ! -d "$LP" ] && continue
        [ -f "$ED/summary.json" ] && continue
        local MP="$OUTDIR/_eval_merged_step_${STEP}"
        echo "=== Merge+eval funcall $RUN_NAME s$STEP ==="
        CUDA_VISIBLE_DEVICES="" $PY -c "
from transformers import AutoModelForCausalLM, AutoTokenizer; from peft import PeftModel; import torch
b=AutoModelForCausalLM.from_pretrained('$SM',torch_dtype=torch.bfloat16)
m=PeftModel.from_pretrained(b,'$LP').merge_and_unload()
m.save_pretrained('$MP'); AutoTokenizer.from_pretrained('$SM',trust_remote_code=True).save_pretrained('$MP'); print('Merged')
"
        CUDA_VISIBLE_DEVICES=1 $PY eval_funcall.py --model "$MP" --output_dir "$ED" \
            --gpu_id 0 --gpu_memory_utilization 0.70 --categories "simple,multiple"
        rm -rf "$MP"
    done
}

# ============================================================
# Qwen scaling fullseq (sglang colocate mode)
# ============================================================
echo "=== PHASE 1: Qwen 4B Fullseq ==="

# SGLang server on GPU 0 (dedicated), student+teacher on GPU 1
# 4B: student(3GB)+teacher(8GB)+optim(3GB) = 14GB on GPU 1 ← fits
# 8B: student(3GB)+teacher(16GB)+optim(3GB) = 22GB on GPU 1 ← fits

# 4B Funcall
train_fullseq "Qwen/Qwen3-4B" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
    "checkpoints/fullseq-m1.5b-t4b-funcall" "fullseq-m1.5b-t4b-funcall" "--problem_field problem"
eval_funcall "checkpoints/fullseq-m1.5b-t4b-funcall" "fullseq-m1.5b-t4b-funcall"

# 4B Coding (continue from step_50)
train_fullseq "Qwen/Qwen3-4B" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
    "checkpoints/fullseq-m1.5b-t4b-coding" "fullseq-m1.5b-t4b-coding" "--problem_field instruction"
eval_coding "checkpoints/fullseq-m1.5b-t4b-coding" "fullseq-m1.5b-t4b-coding"

echo "=== PHASE 2: Qwen 8B Fullseq ==="

# 8B Math (continue from step_50)
train_fullseq "Qwen/Qwen3-8B" "AI-MO/NuminaMath-CoT" "$MATH_SYS" \
    "checkpoints/fullseq-m1.5b-t8b-math" "fullseq-m1.5b-t8b-math" ""
eval_math "checkpoints/fullseq-m1.5b-t8b-math" "fullseq-m1.5b-t8b-math"

# 8B Coding
train_fullseq "Qwen/Qwen3-8B" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" \
    "checkpoints/fullseq-m1.5b-t8b-coding" "fullseq-m1.5b-t8b-coding" "--problem_field instruction"
eval_coding "checkpoints/fullseq-m1.5b-t8b-coding" "fullseq-m1.5b-t8b-coding"

# 8B Funcall
train_fullseq "Qwen/Qwen3-8B" "data/funcall/train.jsonl" "$FUNCALL_SYS" \
    "checkpoints/fullseq-m1.5b-t8b-funcall" "fullseq-m1.5b-t8b-funcall" "--problem_field problem"
eval_funcall "checkpoints/fullseq-m1.5b-t8b-funcall" "fullseq-m1.5b-t8b-funcall"

echo "=== ALL FULLSEQ DONE ==="
