#!/usr/bin/env bash
# =============================================================================
# 一键启动全序列评估监控系统
# =============================================================================

set -u
REPO="/sg-pvc/quick-distillation"
cd "$REPO" || exit 1

echo "🚀 Starting Full-Sequence Evaluation Monitoring System..."

# 确保脚本可执行
chmod +x scripts/*.sh

# 启动 robust watchdog (主监控)
if ! pgrep -f "robust_eval_watchdog.sh" > /dev/null; then
    echo "Starting robust watchdog..."
    nohup ./scripts/robust_eval_watchdog.sh > logs/robust_watchdog.nohup.log 2>&1 &
    echo "   Robust watchdog PID: $!"
else
    echo "✅ Robust watchdog already running"
fi

# 启动旧的 watchdog (作为备份)
if ! pgrep -f "watch_eval_m1.5b_t4b_t8b_all_steps.sh" > /dev/null; then
    echo "Starting backup watchdog..."
    nohup ./scripts/watch_eval_m1.5b_t4b_t8b_all_steps.sh > logs/eval_watchdog_backup.log 2>&1 &
    echo "   Backup watchdog PID: $!"
else
    echo "✅ Backup watchdog already running"
fi

echo ""
echo "📊 Monitoring commands:"
echo "   tail -f logs/robust_eval_watchdog.log"
echo "   tail -f logs/eval_m1.5b_t4b_t8b.watchdog.log"
echo "   nvidia-smi -l 10"
echo ""
echo "✅ All monitoring systems started!"
echo "   This will ensure all missing evaluations (funcall + coding) keep running."
