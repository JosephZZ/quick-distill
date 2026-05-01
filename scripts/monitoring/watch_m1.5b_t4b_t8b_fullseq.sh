#!/usr/bin/env bash
set -u

REPO="/sg-pvc/quick-distillation"
cd "$REPO" || exit 1
mkdir -p logs

WLOG="$REPO/logs/m1.5b_t4b_t8b_fullseq.watchdog.log"
DISABLE="$REPO/logs/m1.5b_t4b_t8b_fullseq.watchdog.disable"
QUEUE_SCRIPT="$REPO/scripts/run_m1.5b_t4b_t8b_fullseq_queue.sh"
QUEUE_NOHUP_LOG="$REPO/logs/m1.5b_t4b_t8b_fullseq.queue.nohup.log"

EXPS=(
  scale-m1.5b-t8b-math-fullseq
  scale-m1.5b-t8b-coding-fullseq
  scale-m1.5b-t8b-funcall-fullseq
  scale-m1.5b-t4b-coding-fullseq
  scale-m1.5b-t4b-funcall-fullseq
)

ts() { date -Is; }
log() { echo "$(ts) $*" | tee -a "$WLOG"; }

if [[ -f "$DISABLE" ]]; then
  log "DISABLE file present, skip check."
  exit 0
fi

work_remains=0
for exp in "${EXPS[@]}"; do
  if [[ -d "$REPO/checkpoints/$exp/step_200" ]]; then
    log "DONE $exp"
  else
    log "TODO $exp"
    work_remains=1
  fi
done

# Frequent status snapshots for post-mortem debugging.
{
  echo "$(ts) process snapshot:" 
  pgrep -af 'run_m1\.5b_t4b_t8b_fullseq_queue\.sh|on_policy_distill_positional\.py.*scale-m1\.5b-t(4|8)b-.*fullseq' || true
  echo "$(ts) gpu snapshot:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
} >> "$WLOG" 2>&1

if [[ $work_remains -eq 0 ]]; then
  log "ALL DONE: all 5 target experiments have step_200."
  exit 0
fi

if pgrep -f 'run_m1\.5b_t4b_t8b_fullseq_queue\.sh' >/dev/null 2>&1 || \
   pgrep -af 'on_policy_distill_positional\.py' | grep -qE 'scale-m1\.5b-t(4|8)b-[^[:space:]]*fullseq'; then
  log "RUNNING: queue or training process is alive."
  exit 0
fi

log "RESTART: work remains but no process alive. starting queue via nohup"
nohup bash "$QUEUE_SCRIPT" >> "$QUEUE_NOHUP_LOG" 2>&1 &
log "STARTED queue pid=$!"
exit 0
