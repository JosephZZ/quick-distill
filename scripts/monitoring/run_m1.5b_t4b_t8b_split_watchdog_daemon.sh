#!/usr/bin/env bash
set -u
REPO="/sg-pvc/quick-distillation"
INTERVAL_SEC=300
mkdir -p "$REPO/logs"
LOCK_FILE="/tmp/m1.5b_t4b_t8b_split_watchdog_daemon.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date -Is) split watchdog daemon already running, exit." >> "$REPO/logs/m1.5b_t4b_t8b_split.watchdog.daemon.log"
  exit 0
fi
echo "$(date -Is) split watchdog daemon started, interval=${INTERVAL_SEC}s" >> "$REPO/logs/m1.5b_t4b_t8b_split.watchdog.daemon.log"
while true; do
  flock -n /tmp/m1.5b_t4b_t8b_split.watchdog.lock bash "$REPO/scripts/watch_m1.5b_t4b_t8b_split.sh" >> "$REPO/logs/m1.5b_t4b_t8b_split.watchdog.daemon.log" 2>&1 || true
  sleep "$INTERVAL_SEC"
done
