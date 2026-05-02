#!/bin/bash
# Pipeline: (1) eval hi-ent-abs01-prefix100 step 50/100/150/200,
# then (2) launch new hi-ent-abs01-fullseq training (full sequence, abs H>0.01).

set -uo pipefail

BASE=/zhi_backup/ziheng/quick-distillation/quick-distillation
cd "$BASE"
mkdir -p logs

GPU=1
STUDENT="Qwen/Qwen2.5-Math-1.5B"
TEACHER="Qwen/Qwen3-1.7B"
PYTHON=python

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

EVAL_DIR_BASE="checkpoints/hi-ent-abs01-prefix100-1.7b-math"
LOG_PIPE=logs/hi_ent_abs01_pipeline.log
echo "[$(date)] pipeline start" | tee -a "$LOG_PIPE"

# === STEP 1: Eval prefix-100 abs01 checkpoints ===
for STEP in 50 100 150 200; do
    LORA_PATH="$EVAL_DIR_BASE/step_${STEP}"
    EVAL_DIR="$EVAL_DIR_BASE/eval_step_${STEP}"
    [ ! -d "$LORA_PATH" ] && { echo "[$(date)] step $STEP missing, skip" | tee -a "$LOG_PIPE"; continue; }
    [ -f "$EVAL_DIR/summary.json" ] && { echo "[$(date)] step $STEP already evaled, skip" | tee -a "$LOG_PIPE"; continue; }

    MERGED="$EVAL_DIR_BASE/_eval_merged_step_${STEP}"
    echo "[$(date)] merging step $STEP (CPU)" | tee -a "$LOG_PIPE"
    CUDA_VISIBLE_DEVICES="" $PYTHON -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
base = AutoModelForCausalLM.from_pretrained(\"$STUDENT\", torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, \"$LORA_PATH\")
merged = m.merge_and_unload()
merged.save_pretrained(\"$MERGED\")
AutoTokenizer.from_pretrained(\"$STUDENT\", trust_remote_code=True).save_pretrained(\"$MERGED\")
print(\"merged ok\")
" >> "$LOG_PIPE" 2>&1

    echo "[$(date)] eval step $STEP on GPU $GPU" | tee -a "$LOG_PIPE"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON eval_math500.py \
        --model "$MERGED" --output_dir "$EVAL_DIR" \
        --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.85 \
        >> "$LOG_PIPE" 2>&1

    rm -rf "$MERGED"
    if [ -f "$EVAL_DIR/summary.json" ]; then
        $PYTHON -c "import json; d=json.load(open(\"$EVAL_DIR/summary.json\")); print(f\"  step $STEP: avg={d.get(\"avg_accuracy\",0)*100:.2f} maj={d.get(\"maj_accuracy\",0)*100:.2f} pass={d.get(\"pass_accuracy\",d.get(\"accuracy\",0))*100:.2f}\")" | tee -a "$LOG_PIPE"
    else
        echo "[$(date)] WARN: step $STEP eval failed (no summary.json)" | tee -a "$LOG_PIPE"
    fi
done

echo "[$(date)] === eval phase done, launching fullseq abs01 training ===" | tee -a "$LOG_PIPE"

# === STEP 2: Launch hi-ent-abs01-fullseq training ===
NAME=hi-ent-abs01-fullseq-1.7b-math
MATH_SYS="Please reason step by step, and put your final answer within \\\\boxed{}."

CUDA_VISIBLE_DEVICES=$GPU $PYTHON on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset AI-MO/NuminaMath-CoT --num_problems 3200 \
    --bs 16 --n_samples 1 --mini_bs 2 \
    --temperature 0.7 --lr 5e-5 --lora_r 32 --lora_alpha 64 \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-hi-ent \
    --output_dir "checkpoints/$NAME" \
    --token_select_mode hi_ent \
    --hi_ent_threshold 0.01 \
    --position_limit 0 \
    --max_new_tokens 2048 \
    --teacher_micro_bs 4 --gen_batch_size 4 \
    --wandb_run_name "$NAME" \
    > "logs/${NAME}.log" 2>&1
echo "[$(date)] training $NAME exited" | tee -a "$LOG_PIPE"
