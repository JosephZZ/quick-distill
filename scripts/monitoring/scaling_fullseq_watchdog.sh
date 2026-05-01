#!/usr/bin/env bash
# Keep scaling fullseq drivers running: if work remains and the GPU0/GPU1 bash (or matching
# on_policy_distill) is gone, restart via nohup. Touch logs/scaling_fullseq_watchdog.disable to pause.
set -u
REPO="/sg-pvc/quick-distillation"
cd "$REPO" || exit 1
mkdir -p "$REPO/logs"

DISABLE="$REPO/logs/scaling_fullseq_watchdog.disable"
WLOG="$REPO/logs/scaling_fullseq_watchdog.log"

ts() { date -Is; }

log() { echo "$(ts) $*" >>"$WLOG"; echo "$(ts) $*"; }

if [[ -f "$DISABLE" ]]; then
  log "DISABLE file present — skip."
  exit 0
fi

# Expected checkpoint dirs (must contain step_200 when that experiment is finished)
GPU0_EXPS=(
  scale-q1.7b-t4b-math-fullseq
  scale-q1.7b-t4b-coding-fullseq
  scale-q1.7b-t4b-funcall-fullseq
  scale-m1.5b-t4b-math-fullseq
  scale-m1.5b-t4b-coding-fullseq
  scale-m1.5b-t4b-funcall-fullseq
)
GPU1_EXPS=(
  scale-q1.7b-t8b-math-fullseq
  scale-q1.7b-t8b-coding-fullseq
  scale-q1.7b-t8b-funcall-fullseq
  scale-m1.5b-t8b-math-fullseq
  scale-m1.5b-t8b-coding-fullseq
  scale-m1.5b-t8b-funcall-fullseq
  scale-q4b-t8b-math-fullseq
  scale-q4b-t8b-coding-fullseq
  scale-q4b-t8b-funcall-fullseq
)

work_remains_gpu0() {
  local d
  for d in "${GPU0_EXPS[@]}"; do
    [[ -d "$REPO/checkpoints/$d/step_200" ]] || return 0
  done
  return 1
}

work_remains_gpu1() {
  local d
  for d in "${GPU1_EXPS[@]}"; do
    [[ -d "$REPO/checkpoints/$d/step_200" ]] || return 0
  done
  return 1
}

bash_gpu0_running() { pgrep -f 'run_scaling_gpu0_fullseq\.sh' >/dev/null 2>&1; }
bash_gpu1_running() { pgrep -f 'run_scaling_gpu1_fullseq\.sh' >/dev/null 2>&1; }

# Distill child: output_dir in argv contains t4b/t8b and fullseq
py_gpu0_running() {
  pgrep -af 'on_policy_distill_positional\.py' 2>/dev/null | grep -qE 't4b[^[:space:]]*fullseq|/scale-[^[:space:]]*t4b[^[:space:]]*fullseq' || return 1
  return 0
}
py_gpu1_running() {
  pgrep -af 'on_policy_distill_positional\.py' 2>/dev/null | grep -qE 't8b[^[:space:]]*fullseq|/scale-[^[:space:]]*t8b[^[:space:]]*fullseq' || return 1
  return 0
}

start_gpu0() {
  log "Starting run_scaling_gpu0_fullseq.sh (nohup)"
  nohup bash "$REPO/scripts/run_scaling_gpu0_fullseq.sh" >>"$REPO/logs/scaling_gpu0_fullseq.nohup.log" 2>&1 &
}

start_gpu1() {
  log "Starting run_scaling_gpu1_fullseq.sh (nohup)"
  nohup bash "$REPO/scripts/run_scaling_gpu1_fullseq.sh" >>"$REPO/logs/scaling_gpu1_fullseq.nohup.log" 2>&1 &
}

# GPU 0 lane
if work_remains_gpu0; then
  if bash_gpu0_running || py_gpu0_running; then
    log "GPU0 fullseq: work remains, driver or train process OK"
  else
    log "GPU0 fullseq: work remains but nothing running — restarting"
    start_gpu0
  fi
else
  log "GPU0 fullseq: all 6 experiments have step_200 — idle"
fi

# GPU 1 lane
if work_remains_gpu1; then
  if bash_gpu1_running || py_gpu1_running; then
    log "GPU1 fullseq: work remains, driver or train process OK"
  else
    log "GPU1 fullseq: work remains but nothing running — restarting"
    start_gpu1
  fi
else
  log "GPU1 fullseq: all 9 experiments have step_200 — idle"
fi

exit 0
