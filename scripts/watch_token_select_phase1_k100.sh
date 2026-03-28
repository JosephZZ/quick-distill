#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

WATCH_LOG="logs/watch_token_select_phase1_k100.log"
RUN_LOG="logs/run_token_select_phase1_k100.nohup.log"

D1="checkpoints/token-select-k100-topkl-math"
D2="checkpoints/token-select-k100-topkl-funcall"
D3="checkpoints/token-select-k100-topent-student-math"
D4="checkpoints/token-select-k100-topent-teacher-math"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$WATCH_LOG"; }

is_done() {
  local d="$1"
  [ -d "$d/step_200" ] && [ -f "$d/eval_step_200/summary.json" ]
}

all_done() {
  is_done "$D1" && is_done "$D2" && is_done "$D3" && is_done "$D4"
}

runner_alive() {
  pgrep -f "run_token_select_phase1_k100\.sh|token-select-k100" >/dev/null 2>&1
}

start_runner() {
  log "Runner not alive; starting run_token_select_phase1_k100.sh"
  nohup bash "$BASE_DIR/scripts/run_token_select_phase1_k100.sh" >>"$RUN_LOG" 2>&1 &
  sleep 2
  if runner_alive; then
    log "Runner started"
  else
    log "WARNING: start issued, process not detected"
  fi
}

log "watchdog started (5-minute interval)"

while true; do
  s1=no; s2=no; s3=no; s4=no
  if is_done "$D1"; then s1=yes; fi
  if is_done "$D2"; then s2=yes; fi
  if is_done "$D3"; then s3=yes; fi
  if is_done "$D4"; then s4=yes; fi
  log "status: topkl-math=$s1 topkl-funcall=$s2 topent-student-math=$s3 topent-teacher-math=$s4"

  if all_done; then
    log "all phase-1 K=100 experiments done"
    exit 0
  fi

  if runner_alive; then
    log "runner alive"
  else
    start_runner
  fi

  sleep 300
 done
