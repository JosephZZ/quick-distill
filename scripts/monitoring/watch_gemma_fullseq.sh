#!/bin/bash
# 5-minute watchdog for Gemma fullseq runs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

WATCH_LOG="logs/watch_gemma_fullseq.log"
RUN_LOG="logs/run_gemma_fullseq.nohup.log"

MATH_DIR="checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq"
CODING_DIR="checkpoints/scale-gemma2-2b-tgemma3-4b-coding-fullseq"
AGENTIC_DIR="checkpoints/scale-gemma2-2b-tgemma3-4b-agentic-fullseq"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$WATCH_LOG"; }

is_task_done() {
  local d="$1"
  [ -d "$d/step_200" ] && [ -f "$d/eval_step_200/summary.json" ]
}

all_done() {
  if is_task_done "$MATH_DIR" && is_task_done "$CODING_DIR" && is_task_done "$AGENTIC_DIR"; then
    return 0
  fi
  return 1
}

runner_alive() {
  pgrep -f "bash .*scripts/run_gemma_fullseq\.sh|scripts/run_gemma_fullseq\.sh" >/dev/null 2>&1
}

start_runner() {
  log "Runner not alive; starting scripts/run_gemma_fullseq.sh via nohup"
  nohup bash "$BASE_DIR/scripts/run_gemma_fullseq.sh" >>"$RUN_LOG" 2>&1 &
  sleep 2
  if runner_alive; then
    log "Runner started successfully"
  else
    log "WARNING: runner start command issued, but process not detected yet"
  fi
}

log "Watchdog started (5-minute interval)"

while true; do
  math_done="no"; coding_done="no"; agentic_done="no"
  if is_task_done "$MATH_DIR"; then math_done="yes"; fi
  if is_task_done "$CODING_DIR"; then coding_done="yes"; fi
  if is_task_done "$AGENTIC_DIR"; then agentic_done="yes"; fi

  log "status: math=$math_done coding=$coding_done agentic=$agentic_done"

  if all_done; then
    log "All tasks completed. Watchdog exiting."
    exit 0
  fi

  if runner_alive; then
    log "runner alive"
  else
    start_runner
  fi

  sleep 300
 done
