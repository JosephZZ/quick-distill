#!/usr/bin/env bash
set -u
set -o pipefail

REPO="/sg-pvc/quick-distillation"
cd "$REPO" || exit 1
mkdir -p logs

WLOG="$REPO/logs/eval_m1.5b_t4b_t8b.watchdog.log"
Q0="$REPO/scripts/eval_m1.5b_t4b_all_steps_gpu0.sh"
Q1="$REPO/scripts/eval_m1.5b_t8b_all_steps_gpu1.sh"

ts() { date -Is; }
log() { echo "$(ts) $*" | tee -a "$WLOG"; }

done_math=1
for s in 50 100 150 200; do
  d="$REPO/checkpoints/scale-m1.5b-t8b-math-fullseq/eval_step_$s/summary.json"
  [[ -f "$d" ]] || done_math=0
done

done_t8_coding=1
for s in 50 100 150 200; do
  d="$REPO/checkpoints/scale-m1.5b-t8b-coding-fullseq/eval_step_$s"
  ls "$d"/humaneval_*.jsonl "$d"/mbpp_*.jsonl >/dev/null 2>&1 || done_t8_coding=0
done

done_t8_funcall=1
for s in 50 100 150 200; do
  d="$REPO/checkpoints/scale-m1.5b-t8b-funcall-fullseq/eval_step_$s/summary.json"
  [[ -f "$d" ]] || done_t8_funcall=0
done

done_t4_coding=1
for s in 50 100 150 200; do
  d="$REPO/checkpoints/scale-m1.5b-t4b-coding-fullseq/eval_step_$s"
  ls "$d"/humaneval_*.jsonl "$d"/mbpp_*.jsonl >/dev/null 2>&1 || done_t4_coding=0
done

done_t4_funcall=1
for s in 50 100 150 200; do
  d="$REPO/checkpoints/scale-m1.5b-t4b-funcall-fullseq/eval_step_$s/summary.json"
  [[ -f "$d" ]] || done_t4_funcall=0
done

log "status t8_math=$done_math t8_coding=$done_t8_coding t8_funcall=$done_t8_funcall t4_coding=$done_t4_coding t4_funcall=$done_t4_funcall"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$WLOG" 2>&1 || true

t8_all_done=0
[[ $done_math -eq 1 && $done_t8_coding -eq 1 && $done_t8_funcall -eq 1 ]] && t8_all_done=1
t4_all_done=0
[[ $done_t4_coding -eq 1 && $done_t4_funcall -eq 1 ]] && t4_all_done=1

if [[ $t8_all_done -eq 0 ]]; then
  if pgrep -f 'eval_m1\.5b_t8b_all_steps_gpu1\.sh|scale-m1\.5b-t8b-.*eval' >/dev/null 2>&1; then
    log "t8 eval lane running"
  else
    log "restart t8 eval lane"
    nohup bash "$Q1" >> "$REPO/logs/eval_m1.5b_t8b_all_steps_gpu1.nohup.log" 2>&1 &
    log "started t8 pid=$!"
  fi
else
  log "t8 eval lane done"
fi

if [[ $t4_all_done -eq 0 ]]; then
  if pgrep -f 'eval_m1\.5b_t4b_all_steps_gpu0\.sh|scale-m1\.5b-t4b-.*eval' >/dev/null 2>&1; then
    log "t4 eval lane running"
  else
    log "restart t4 eval lane"
    nohup bash "$Q0" >> "$REPO/logs/eval_m1.5b_t4b_all_steps_gpu0.nohup.log" 2>&1 &
    log "started t4 pid=$!"
  fi
else
  log "t4 eval lane done"
fi

if [[ $t8_all_done -eq 1 && $t4_all_done -eq 1 ]]; then
  log "ALL EVAL DONE"
fi

