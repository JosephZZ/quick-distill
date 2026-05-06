#!/bin/bash
# Background eval for scale-m7b-t14b-math-fullseq on GPU 0 (teacher GPU) while training runs.
# Uses only ~17GB of GPU 0 (teacher uses 28.5GB, leaves ~3GB buffer).
# Aborts if teacher GPU peaks > 44GB at start (out of 48GB total) — too risky.
# Runs step_50 -> 100 -> 150 serially. step_200 will be done by the post-training cron.
set -eo pipefail
cd /zhi_backup/ziheng/quick-distillation

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

EXP="scale-m7b-t14b-math-fullseq"
STUDENT="Qwen/Qwen2.5-Math-7B"
LOG=/zhi_backup/ziheng/quick-distillation/quick-distillation/logs/eval_${EXP}_gpu0.log
LOCK=/tmp/eval_${EXP}_gpu0.lock

exec 200>"$LOCK"
flock -n 200 || { echo "[$(date +%F\ %T)] already running, skip" >> "$LOG"; exit 0; }

mkdir -p "$(dirname "$LOG")"
echo "" >> "$LOG"
echo "[$(date +%F\ %T)] === GPU0 eval pass start ===" >> "$LOG"

# Safety check: GPU 0 memory used must be < 32GB (teacher is 28.5GB; if higher, abort)
GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d "[:space:]")
echo "[$(date +%F\ %T)] GPU0 used: ${GPU0_USED} MiB" >> "$LOG"
if [ "${GPU0_USED}" -gt 32000 ]; then
  echo "[$(date +%F\ %T)] ABORT: GPU0 too hot (${GPU0_USED} MiB > 32000 MiB)" >> "$LOG"
  exit 0
fi

# Also verify training process still alive (if dead, let the post-training cron handle it)
if ! pgrep -f "on_policy_distill_positional.py.*Qwen2.5-Math-7B.*scale-m7b-t14b-math-fullseq" >/dev/null 2>&1; then
  echo "[$(date +%F\ %T)] training no longer running — yielding to post-training cron" >> "$LOG"
  exit 0
fi

for step in 50 100 150; do
  CKPT="checkpoints/${EXP}/step_${step}"
  MERGED="${CKPT}_merged"
  EVAL_DIR="checkpoints/${EXP}/eval_step_${step}"

  if [ ! -d "$CKPT" ]; then
    echo "[$(date +%F\ %T)] step_${step} not on disk, skip" >> "$LOG"
    continue
  fi
  if [ -f "${EVAL_DIR}/summary.json" ]; then
    echo "[$(date +%F\ %T)] step_${step} already evaluated, skip" >> "$LOG"
    continue
  fi

  # Re-check GPU 0 before each eval
  GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d "[:space:]")
  if [ "${GPU0_USED}" -gt 32000 ]; then
    echo "[$(date +%F\ %T)] ABORT mid-loop: GPU0 hot (${GPU0_USED} MiB)" >> "$LOG"
    exit 0
  fi

  if [ ! -f "${MERGED}/config.json" ]; then
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

  echo "[$(date +%F\ %T)] eval step_${step} on GPU 0 with mem_util=0.34 ..." >> "$LOG"
  # gpu_memory_utilization=0.34 * 49140 = 16.7 GB cap for vLLM
  CUDA_VISIBLE_DEVICES=0 python3 eval_math500.py \
    --model "${MERGED}" \
    --output_dir "${EVAL_DIR}" \
    --n_samples 4 --temperature 0.7 \
    --gpu_memory_utilization 0.34 \
    --max_model_len 4096 >> "$LOG" 2>&1 || {
      echo "[$(date +%F\ %T)] EVAL FAILED for step_${step} — likely OOM. Stopping GPU0 loop." >> "$LOG"
      rm -rf "${MERGED}"
      exit 1
    }

  if [ -f "${EVAL_DIR}/summary.json" ]; then
    echo "[$(date +%F\ %T)] step_${step} done:" >> "$LOG"
    python3 -c "import json; d=json.load(open(\"${EVAL_DIR}/summary.json\")); print(f\"  avg@4={d[\\\"avg_accuracy\\\"]*100:.2f}%  pass@4={d[\\\"pass_accuracy\\\"]*100:.2f}%  maj@4={d[\\\"maj_accuracy\\\"]*100:.2f}%\")" >> "$LOG" 2>&1
    rm -rf "${MERGED}"
    echo "[$(date +%F\ %T)] cleaned up ${MERGED}" >> "$LOG"
  else
    echo "[$(date +%F\ %T)] step_${step} FAILED — no summary.json" >> "$LOG"
    exit 1
  fi
done

echo "[$(date +%F\ %T)] === GPU0 eval pass done ===" >> "$LOG"
