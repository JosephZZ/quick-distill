#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR" || exit 1
mkdir -p logs

LOCK_FILE="/tmp/eval_m1.5b_t8b_all_steps_gpu1.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date -Is) another eval lane(gpu1) is running, exit." >> "$REPO_DIR/logs/eval_m1.5b_t8b_all_steps_gpu1.log"
  exit 0
fi

[ -f "${HOME}/.bashrc" ] && PS1=nonempty source "${HOME}/.bashrc" 2>/dev/null || true
source "$SCRIPT_DIR/hf_models_env.sh"

STUDENT="$MATH_STUDENT_15"
GPU=1
MATH_MEM=0.70
OTHER_MEM=0.50
STEPS=(50 100 150 200)
MASTER_LOG="$REPO_DIR/logs/eval_m1.5b_t8b_all_steps_gpu1.log"

ts() { date -Is; }
log() { echo "$(ts) $*" | tee -a "$MASTER_LOG"; }

need_eval_math() {
  local outdir="$1"
  [[ ! -f "$outdir/summary.json" ]]
}

need_eval_coding() {
  local outdir="$1"
  [[ ! -f "$outdir/humaneval_"*".jsonl" || ! -f "$outdir/mbpp_"*".jsonl" ]] 2>/dev/null
}

need_eval_funcall() {
  local outdir="$1"
  [[ ! -f "$outdir/summary.json" ]]
}

merge_lora() {
  local lora_path="$1"
  local merged_path="$2"
  CUDA_VISIBLE_DEVICES="" python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
base = AutoModelForCausalLM.from_pretrained("$STUDENT", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "$lora_path")
merged = model.merge_and_unload()
merged.save_pretrained("$merged_path")
AutoTokenizer.from_pretrained("$STUDENT", trust_remote_code=True).save_pretrained("$merged_path")
print("Merged to $merged_path")
PY
}

eval_math_step() {
  local exp="$1" step="$2"
  local ckpt="$REPO_DIR/checkpoints/$exp/step_$step"
  local eval_dir="$REPO_DIR/checkpoints/$exp/eval_step_$step"
  local merged="$REPO_DIR/checkpoints/$exp/_eval_merged_step_$step"
  local run_log="$REPO_DIR/logs/${exp}.eval.log"
  [[ -d "$ckpt" ]] || { log "SKIP $exp step=$step (missing ckpt)"; return 0; }
  mkdir -p "$eval_dir"
  if need_eval_math "$eval_dir"; then
    log "START eval math $exp step=$step"
    merge_lora "$ckpt" "$merged" >> "$run_log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU python eval_math500.py \
      --model "$merged" \
      --output_dir "$eval_dir" \
      --n_samples 4 \
      --temperature 0.7 \
      --gpu_memory_utilization "$MATH_MEM" \
      >> "$run_log" 2>&1
    rm -rf "$merged"
    log "DONE eval math $exp step=$step"
  else
    log "SKIP eval math $exp step=$step (already done)"
  fi
}

eval_coding_step() {
  local exp="$1" step="$2"
  local ckpt="$REPO_DIR/checkpoints/$exp/step_$step"
  local eval_dir="$REPO_DIR/checkpoints/$exp/eval_step_$step"
  local merged="$REPO_DIR/checkpoints/$exp/_eval_merged_step_$step"
  local run_log="$REPO_DIR/logs/${exp}.eval.log"
  [[ -d "$ckpt" ]] || { log "SKIP $exp step=$step (missing ckpt)"; return 0; }
  mkdir -p "$eval_dir"
  if need_eval_coding "$eval_dir"; then
    log "START eval coding $exp step=$step"
    merge_lora "$ckpt" "$merged" >> "$run_log" 2>&1
    for ds in humaneval mbpp; do
      CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_humaneval.py \
        --model "$merged" \
        --dataset "$ds" \
        --output_dir "$eval_dir" \
        --gpu_memory_utilization "$OTHER_MEM" \
        --trust_remote_code \
        >> "$run_log" 2>&1
    done
    echo '{"status":"done"}' > "$eval_dir/summary.json"
    rm -rf "$merged"
    log "DONE eval coding $exp step=$step"
  else
    log "SKIP eval coding $exp step=$step (already done)"
  fi
}

eval_funcall_step() {
  local exp="$1" step="$2"
  local ckpt="$REPO_DIR/checkpoints/$exp/step_$step"
  local eval_dir="$REPO_DIR/checkpoints/$exp/eval_step_$step"
  local merged="$REPO_DIR/checkpoints/$exp/_eval_merged_step_$step"
  local run_log="$REPO_DIR/logs/${exp}.eval.log"
  [[ -d "$ckpt" ]] || { log "SKIP $exp step=$step (missing ckpt)"; return 0; }
  mkdir -p "$eval_dir"
  if need_eval_funcall "$eval_dir"; then
    log "START eval funcall $exp step=$step"
    merge_lora "$ckpt" "$merged" >> "$run_log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU python eval_funcall.py \
      --model "$merged" \
      --eval_data data/funcall/eval_bfcl.jsonl \
      --output_dir "$eval_dir" \
      --gpu_id 0 \
      --gpu_memory_utilization "$OTHER_MEM" \
      --categories "simple,multiple" \
      >> "$run_log" 2>&1
    rm -rf "$merged"
    log "DONE eval funcall $exp step=$step"
  else
    log "SKIP eval funcall $exp step=$step (already done)"
  fi
}

log "EVAL LANE START gpu1 (t8b)"
for s in "${STEPS[@]}"; do eval_math_step "scale-m1.5b-t8b-math-fullseq" "$s"; done
for s in "${STEPS[@]}"; do eval_coding_step "scale-m1.5b-t8b-coding-fullseq" "$s"; done
for s in "${STEPS[@]}"; do eval_funcall_step "scale-m1.5b-t8b-funcall-fullseq" "$s"; done
log "EVAL LANE DONE gpu1 (t8b)"

