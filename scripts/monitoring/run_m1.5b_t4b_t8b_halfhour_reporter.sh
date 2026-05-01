#!/usr/bin/env bash
set -u
set -o pipefail

REPO="/sg-pvc/quick-distillation"
cd "$REPO" || exit 1
mkdir -p logs

LOCK_FILE="/tmp/m1.5b_t4b_t8b_halfhour_reporter.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date -Is) half-hour reporter already running, exit." >> "$REPO/logs/m1.5b_t4b_t8b_halfhour_status.log"
  exit 0
fi

bash "$REPO/scripts/report_m1.5b_t4b_t8b_status.sh" --init-state

while true; do
  bash "$REPO/scripts/report_m1.5b_t4b_t8b_status.sh"
  sleep 1800
done

