#!/usr/bin/env bash
set -u
REPO="/sg-pvc/quick-distillation"
cd "$REPO" || exit 1
mkdir -p logs

WLOG="$REPO/logs/m1.5b_t4b_t8b_split.watchdog.log"
DISABLE="$REPO/logs/m1.5b_t4b_t8b_split.watchdog.disable"
Q8="$REPO/scripts/run_m1.5b_t8b_fullseq_queue_gpu1.sh"
Q4="$REPO/scripts/run_m1.5b_t4b_fullseq_queue_gpu0.sh"

# 5 target experiments
T8=(scale-m1.5b-t8b-math-fullseq scale-m1.5b-t8b-coding-fullseq scale-m1.5b-t8b-funcall-fullseq)
T4=(scale-m1.5b-t4b-coding-fullseq scale-m1.5b-t4b-funcall-fullseq)

ts() { date -Is; }
log() { echo "$(ts) $*" | tee -a "$WLOG"; }

if [[ -f "$DISABLE" ]]; then
  log "DISABLE file present, skip."
  exit 0
fi

done_t8=1
for e in "${T8[@]}"; do
  if [[ -d "$REPO/checkpoints/$e/step_200" ]]; then log "DONE $e"; else log "TODO $e"; done_t8=0; fi
done

done_t4=1
for e in "${T4[@]}"; do
  if [[ -d "$REPO/checkpoints/$e/step_200" ]]; then log "DONE $e"; else log "TODO $e"; done_t4=0; fi
done

{
  echo "$(ts) process snapshot:"
  pgrep -af 'run_m1\.5b_t8b_fullseq_queue_gpu1\.sh|run_m1\.5b_t4b_fullseq_queue_gpu0\.sh|on_policy_distill_positional\.py.*scale-m1\.5b-t(4|8)b-.*fullseq' || true
  echo "$(ts) gpu snapshot:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
} >> "$WLOG" 2>&1

if [[ $done_t8 -eq 0 ]]; then
  if pgrep -f 'run_m1\.5b_t8b_fullseq_queue_gpu1\.sh' >/dev/null 2>&1 || \
     pgrep -af 'on_policy_distill_positional\.py' | grep -q 'scale-m1.5b-t8b-'; then
    log "T8 lane running"
  else
    log "RESTART T8 lane"
    nohup bash "$Q8" >> "$REPO/logs/m1.5b_t8b_fullseq_gpu1.queue.nohup.log" 2>&1 &
    log "STARTED T8 pid=$!"
  fi
else
  log "T8 lane all done"
fi

if [[ $done_t4 -eq 0 ]]; then
  if pgrep -f 'run_m1\.5b_t4b_fullseq_queue_gpu0\.sh' >/dev/null 2>&1 || \
     pgrep -af 'on_policy_distill_positional\.py' | grep -q 'scale-m1.5b-t4b-'; then
    log "T4 lane running"
  else
    log "RESTART T4 lane"
    nohup bash "$Q4" >> "$REPO/logs/m1.5b_t4b_fullseq_gpu0.queue.nohup.log" 2>&1 &
    log "STARTED T4 pid=$!"
  fi
else
  log "T4 lane all done"
fi

if [[ $done_t8 -eq 1 && $done_t4 -eq 1 ]]; then
  log "ALL DONE: 5 target experiments complete."
fi
