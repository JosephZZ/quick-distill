#!/usr/bin/env bash
set -u
set -o pipefail

REPO="/sg-pvc/quick-distillation"
cd "$REPO" || exit 1

LOG_DIR="$REPO/logs"
STATE_DIR="$LOG_DIR/.m1.5b_t4b_t8b_status_state"
REPORT_LOG="$LOG_DIR/m1.5b_t4b_t8b_halfhour_status.log"
mkdir -p "$LOG_DIR" "$STATE_DIR"

EXPS=(
  "scale-m1.5b-t8b-math-fullseq"
  "scale-m1.5b-t8b-coding-fullseq"
  "scale-m1.5b-t8b-funcall-fullseq"
  "scale-m1.5b-t4b-coding-fullseq"
  "scale-m1.5b-t4b-funcall-fullseq"
)

QUEUE_LOGS=(
  "$LOG_DIR/m1.5b_t8b_fullseq_gpu1.queue.log"
  "$LOG_DIR/m1.5b_t4b_fullseq_gpu0.queue.log"
)

_ts() { date -Is; }

max_step() {
  local outdir="$1"
  if [[ ! -d "$outdir" ]]; then
    echo 0
    return
  fi
  local m
  m=$(find "$outdir" -maxdepth 1 -type d -name 'step_*' -printf '%f\n' 2>/dev/null \
      | sed -n 's/^step_\([0-9]\+\)$/\1/p' | sort -n | tail -n 1)
  if [[ -n "$m" ]]; then echo "$m"; else echo 0; fi
}

latest_log_step() {
  local exp="$1"
  local logf="$LOG_DIR/${exp}.log"
  if [[ ! -f "$logf" ]]; then
    echo 0
    return
  fi
  local s
  s=$(rg -o 'step=[0-9]+' "$logf" 2>/dev/null | sed -n 's/step=\([0-9]\+\)/\1/p' | tail -n 1)
  if [[ -n "$s" ]]; then echo "$s"; else echo 0; fi
}

progress_step() {
  local exp="$1"
  local outdir="$REPO/checkpoints/$exp"
  local dstep lstep
  dstep=$(max_step "$outdir")
  lstep=$(latest_log_step "$exp")
  if (( lstep > dstep )); then
    echo "$lstep"
  else
    echo "$dstep"
  fi
}

eval_done_list() {
  local outdir="$1"
  local done=()
  local s
  for s in 50 100 150 200; do
    if [[ -d "$outdir/eval_step_${s}" ]]; then
      done+=("$s")
    fi
  done
  if [[ ${#done[@]} -eq 0 ]]; then
    echo "none"
  else
    local IFS=,
    echo "${done[*]}"
  fi
}

init_state() {
  local l
  for l in "${QUEUE_LOGS[@]}"; do
    local st="$STATE_DIR/$(basename "$l").line"
    if [[ -f "$l" ]]; then
      wc -l < "$l" > "$st"
    else
      echo 0 > "$st"
    fi
  done
  for e in "${EXPS[@]}"; do
    local elog="$LOG_DIR/${e}.log"
    local st="$STATE_DIR/$(basename "$elog").line"
    if [[ -f "$elog" ]]; then
      wc -l < "$elog" > "$st"
    else
      echo 0 > "$st"
    fi
  done
}

if [[ "${1:-}" == "--init-state" ]]; then
  init_state
  exit 0
fi

if [[ ! -f "$STATE_DIR/.initialized" ]]; then
  init_state
  touch "$STATE_DIR/.initialized"
fi

declare -A RUNNING
while IFS= read -r line; do
  outdir=$(echo "$line" | sed -n 's/.*--output_dir \([^ ]\+\).*/\1/p')
  if [[ -n "$outdir" ]]; then
    exp=$(basename "$outdir")
    RUNNING["$exp"]=1
  fi
done < <(pgrep -af 'on_policy_distill_positional\.py.*scale-m1\.5b-t(4|8)b-.*fullseq' || true)

issues=()
scan_log_for_new_issues() {
  local l="$1"
  [[ -f "$l" ]] || return 0
  local st="$STATE_DIR/$(basename "$l").line"
  local prev=0
  if [[ -f "$st" ]]; then prev=$(cat "$st" 2>/dev/null || echo 0); fi
  local curr
  curr=$(wc -l < "$l")
  if (( curr > prev )); then
    local from=$((prev + 1))
    while IFS= read -r ln; do
      issues+=("[$(basename "$l")] $ln")
    done < <(sed -n "${from},${curr}p" "$l" \
      | rg -N 'RETRY|RESTART|rc=[1-9]|Traceback|Error|OOM|Killed|another instance is running' || true)
  fi
  echo "$curr" > "$st"
}

for l in "${QUEUE_LOGS[@]}"; do
  scan_log_for_new_issues "$l"
done
for e in "${EXPS[@]}"; do
  scan_log_for_new_issues "$LOG_DIR/${e}.log"
done

{
  echo "[$(_ts)] half-hour report"
  echo "1) running experiments:"
  local_running=0
  for e in "${EXPS[@]}"; do
    if [[ -n "${RUNNING[$e]:-}" ]]; then
      echo "   - $e"
      local_running=1
    fi
  done
  if [[ "$local_running" -eq 0 ]]; then
    echo "   - none"
  fi

  echo "2) progress (max step):"
  for e in "${EXPS[@]}"; do
    outdir="$REPO/checkpoints/$e"
    step="$(progress_step "$e")"
    if [[ -d "$outdir/step_200" ]]; then
      state="DONE"
    elif [[ -n "${RUNNING[$e]:-}" ]]; then
      state="RUNNING"
    else
      state="PENDING"
    fi
    echo "   - $e: step=$step, state=$state"
  done

  echo "3) eval_step_{50,100,150,200} completed:"
  for e in "${EXPS[@]}"; do
    outdir="$REPO/checkpoints/$e"
    edone="$(eval_done_list "$outdir")"
    echo "   - $e: $edone"
  done

  if [[ ${#issues[@]} -eq 0 ]]; then
    echo "4) restart/errors since last report: no"
  else
    echo "4) restart/errors since last report: yes"
    for i in "${issues[@]}"; do
      echo "   - $i"
    done
  fi

  echo
} | tee -a "$REPORT_LOG"
