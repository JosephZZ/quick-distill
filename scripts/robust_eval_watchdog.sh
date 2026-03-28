#!/usr/bin/env bash
# =============================================================================
# Robust Evaluation Watchdog for Full-Sequence Scaling Experiments
# 确保所有 missing 的 funcall 和 coding 评估持续运行
# =============================================================================

set -u
set -o pipefail

REPO="/sg-pvc/quick-distillation"
cd "$REPO" || exit 1

mkdir -p logs
WATCHDOG_LOG="$REPO/logs/robust_eval_watchdog.log"
EVAL_LOG="$REPO/logs/eval_robust.log"

ts() { date -Is; }
log() {
    echo "[$(ts)] $*" | tee -a "$WATCHDOG_LOG"
}

log "=== Robust Eval Watchdog Started ==="

# 要监控的实验
EXPERIMENTS=(
    "scale-m1.5b-t4b-funcall-fullseq"
    "scale-m1.5b-t8b-funcall-fullseq"
    "scale-m1.5b-t4b-coding-fullseq"
    "scale-m1.5b-t8b-coding-fullseq"
)

# 评估脚本
EVAL_T4B_SCRIPT="$REPO/scripts/eval_m1.5b_t4b_all_steps_gpu0.sh"
EVAL_T8B_SCRIPT="$REPO/scripts/eval_m1.5b_t8b_all_steps_gpu1.sh"

# 检查实验是否需要评估
needs_eval() {
    local exp=$1
    local has_missing=0

    for step in 50 100 150 200; do
        local eval_dir="$REPO/checkpoints/$exp/eval_step_$step"
        
        if [[ $exp == *funcall* ]]; then
            # funcall 需要 summary.json
            if [[ ! -f "$eval_dir/summary.json" ]]; then
                has_missing=1
            fi
        else
            # coding 需要 humaneval 和 mbpp 的 jsonl 文件
            if [[ ! -f "$eval_dir/humaneval_results.jsonl" && ! -f "$eval_dir/humaneval_*.jsonl" ]] || \
               [[ ! -f "$eval_dir/mbpp_results.jsonl" && ! -f "$eval_dir/mbpp_*.jsonl" ]]; then
                has_missing=1
            fi
        fi
    done

    echo $has_missing
}

# 检查进程是否在运行
is_running() {
    local pattern=$1
    pgrep -f "$pattern" >/dev/null 2>&1
    echo $?
}

# 主监控循环
counter=0
while true; do
    counter=$((counter + 1))
    log "=== Watchdog Cycle $counter ==="

    # 1. 检查 GPU 状态
    log "GPU Status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$WATCHDOG_LOG"

    # 2. 检查每个实验的状态
    t4b_needs=0
    t8b_needs=0

    for exp in "${EXPERIMENTS[@]}"; do
        need=$(needs_eval "$exp")
        status="DONE"
        
        if [[ $need -eq 1 ]]; then
            status="NEEDS_EVAL"
            if [[ $exp == *t4b* ]]; then
                t4b_needs=1
            else
                t8b_needs=1
            fi
        fi
        
        log "  $exp: $status"
    done

    # 3. 确保评估进程在运行
    if [[ $t8b_needs -eq 1 ]]; then
        if [[ $(is_running "eval_m1.5b_t8b_all_steps_gpu1.sh|scale-m1.5b-t8b.*eval") -eq 0 ]]; then
            log "✅ t8b eval lane is running"
        else
            log "⚠️  Restarting t8b eval lane..."
            nohup bash "$EVAL_T8B_SCRIPT" >> "$EVAL_LOG" 2>&1 &
            log "   Started t8b PID=$!"
        fi
    else
        log "✅ t8b eval lane completed"
    fi

    if [[ $t4b_needs -eq 1 ]]; then
        if [[ $(is_running "eval_m1.5b_t4b_all_steps_gpu0.sh|scale-m1.5b-t4b.*eval") -eq 0 ]]; then
            log "✅ t4b eval lane is running"
        else
            log "⚠️  Restarting t4b eval lane..."
            nohup bash "$EVAL_T4B_SCRIPT" >> "$EVAL_LOG" 2>&1 &
            log "   Started t4b PID=$!"
        fi
    else
        log "✅ t4b eval lane completed"
    fi

    # 4. 检查是否全部完成
    all_done=1
    for exp in "${EXPERIMENTS[@]}"; do
        if [[ $(needs_eval "$exp") -eq 1 ]]; then
            all_done=0
            break
        fi
    done

    if [[ $all_done -eq 1 ]]; then
        log "🎉 ALL EVALUATIONS COMPLETED!"
        log "Watchdog will continue monitoring for 2 more hours then exit."
        sleep 7200  # 2小时
        log "Watchdog shutting down after all tasks completed."
        exit 0
    fi

    # 每 5 分钟检查一次
    log "Next check in 5 minutes. (Cycle $counter)"
    echo "----------------------------------------" >> "$WATCHDOG_LOG"
    sleep 300
done
