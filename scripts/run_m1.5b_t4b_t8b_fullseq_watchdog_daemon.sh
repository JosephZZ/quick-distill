#!/usr/bin/env bash
set -u
REPO="/sg-pvc/quick-distillation"
INTERVAL_SEC=300
mkdir -p "$REPO/logs"
echo "$(date -Is) watchdog daemon started, interval=${INTERVAL_SEC}s" >> "$REPO/logs/m1.5b_t4b_t8b_fullseq.watchdog.daemon.log"
while true; do
  flock -n /tmp/m1.5b_t4b_t8b_fullseq.watchdog.lock bash "$REPO/scripts/watch_m1.5b_t4b_t8b_fullseq.sh" >> "$REPO/logs/m1.5b_t4b_t8b_fullseq.watchdog.daemon.log" 2>&1 || true
  sleep "$INTERVAL_SEC"
done
