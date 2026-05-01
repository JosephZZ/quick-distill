#!/usr/bin/env bash
# Use when cron is not available: runs the same check every 10 minutes forever.
# Start: nohup bash scripts/run_scaling_fullseq_watchdog_daemon.sh >> logs/scaling_fullseq_watchdog.daemon.log 2>&1 &
set -euo pipefail
REPO="/sg-pvc/quick-distillation"
INTERVAL_SEC="${WATCHDOG_INTERVAL_SEC:-300}"
cd "$REPO"
mkdir -p "$REPO/logs"
echo "$(date -Is) scaling_fullseq watchdog daemon started, interval=${INTERVAL_SEC}s"
while true; do
  flock -n /tmp/scaling_fullseq_watchdog.lock bash "$REPO/scripts/scaling_fullseq_watchdog.sh" >>"$REPO/logs/scaling_fullseq_watchdog.cron.log" 2>&1 || true
  sleep "$INTERVAL_SEC"
done
