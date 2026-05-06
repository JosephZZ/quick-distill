#!/bin/bash
# Evaluate scale-m7b-t14b-math-fullseq checkpoints sequentially.
# Waits for training to finish before starting (to avoid GPU collision).
# Merges LoRA on CPU, runs eval_math500.py on GPU 1, writes eval_step_N/summary.json.
set -eo pipefail
cd /zhi_backup/ziheng/quick-distillation

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

EXP="scale-m7b-t14b-math-fullseq"
STUDENT="Qwen/Qwen2.5-Math-7B"
LOG=/zhi_backup/ziheng/quick-distillation/quick-distillation/logs/eval_${EXP}.log
LOCK=/tmp/eval_${EXP}.lock

# Single-instance lock (allow cron to fire without piling up)
exec 200>"$LOCK"
flock -n 200 || { echo "[$(date +%F\ %T)] already running, skip" >> "$LOG"; exit 0; }

mkdir -p "$(dirname "$LOG")"
echo "[$(date +%F\ %T)] tick" >> "$LOG"

# Wait guard: if training process still alive, defer.
if pgrep -f "on_policy_distill_positional.py.*Qwen2.5-Math-7B.*scale-m7b-t14b-math-fullseq" >/dev/null 2>&1; then
  echo "[$(date +%F\ %T)] training still running, defer" >> "$LOG"
  exit 0
fi

for step in 50 100 150 200; do
  CKPT="checkpoints/${EXP}/step_${step}"
  MERGED="${CKPT}/_eval_merged"
  EVAL_DIR="${CKPT}/../eval_step_${step}"
  # Note: prefix50 convention was eval_step_N as sibling of step_N. Match it.
  EVAL_DIR="checkpoints/${EXP}/eval_step_${step}"

  if [ ! -d "$CKPT" ]; then
    echo "[$(date +%F\ %T)] step_${step} not yet on disk" >> "$LOG"
    continue
  fi
  if [ -f "${EVAL_DIR}/summary.json" ]; then
    echo "[$(date +%F\ %T)] step_${step} already evaluated" >> "$LOG"
    continue
  fi

  if [ ! -d "$MERGED" ] || [ ! -f "${MERGED}/config.json" ]; then
    echo "[$(date +%F\ %T)] merging step_${step} on CPU..." >> "$LOG"
    CUDA_VISIBLE_DEVICES="" python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
base = AutoModelForCausalLM.from_pretrained(\"${STUDENT}\", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, \"${CKPT}\")
model = model.merge_and_unload()
model.save_pretrained(\"${MERGED}\")
AutoTokenizer.from_pretrained(\"${STUDENT}\").save_pretrained(\"${MERGED}\")
print(\"merged -> ${MERGED}\")
" >> "$LOG" 2>&1
  fi

  echo "[$(date +%F\ %T)] eval step_${step} on GPU 1..." >> "$LOG"
  CUDA_VISIBLE_DEVICES=1 python3 eval_math500.py \
    --model "${MERGED}" \
    --output_dir "${EVAL_DIR}" \
    --n_samples 4 --temperature 0.7 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 4096 >> "$LOG" 2>&1

  if [ -f "${EVAL_DIR}/summary.json" ]; then
    echo "[$(date +%F\ %T)] step_${step} done:" >> "$LOG"
    python3 -c "import json; d=json.load(open(\"${EVAL_DIR}/summary.json\")); print(f\"  avg@4={d[\\\"avg_accuracy\\\"]*100:.2f}%  pass@4={d[\\\"pass_accuracy\\\"]*100:.2f}%  maj@4={d[\\\"maj_accuracy\\\"]*100:.2f}%\")" >> "$LOG" 2>&1
    # Free disk: drop merged dir (~14GB each for 7B) after successful eval
    rm -rf "${MERGED}"
    echo "[$(date +%F\ %T)] cleaned up ${MERGED}" >> "$LOG"
  else
    echo "[$(date +%F\ %T)] step_${step} FAILED — summary.json missing" >> "$LOG"
  fi
done

echo "[$(date +%F\ %T)] eval pass complete" >> "$LOG"
