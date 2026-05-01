#!/bin/bash
# Watchdog for Qwen3-8B training - checks every 5 minutes

LOG_FILE="/sg-pvc/quick-distillation/logs/watchdog_t8b.log"
PID_FILE="/tmp/watchdog_t8b.pid"

# Write PID
echo $$ > $PID_FILE

echo "$(date '+%Y-%m-%d %H:%M:%S') Watchdog started" >> $LOG_FILE

while true; do
    echo "" >> $LOG_FILE
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> $LOG_FILE
    
    # Check running processes
    echo "Running processes:" >> $LOG_FILE
    ps aux | grep "on_policy_distill" | grep "t8b" | grep -v grep | grep -oE "scale-[a-z0-9.-]+" | sort | uniq -c >> $LOG_FILE 2>&1
    
    # Check GPU
    echo "GPU memory:" >> $LOG_FILE
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >> $LOG_FILE 2>&1
    
    # Check log progress for each experiment
    for exp in scale-m1.5b-t8b-math-fullseq scale-q1.7b-t8b-math-fullseq scale-q4b-t8b-math-fullseq; do
        LOG="/sg-pvc/quick-distillation/logs/${exp}.log"
        if [ -f "$LOG" ]; then
            LAST_STEP=$(grep "step=" "$LOG" 2>/dev/null | tail -1)
            if [ -n "$LAST_STEP" ]; then
                echo "${exp}: ${LAST_STEP:0:60}" >> $LOG_FILE
            else
                echo "${exp}: No training steps yet" >> $LOG_FILE
            fi
            
            # Check for OOM
            OOM_COUNT=$(grep -c "OutOfMemoryError\|OOM" "$LOG" 2>/dev/null || echo "0")
            if [ "$OOM_COUNT" -gt 0 ]; then
                echo "  WARNING: ${OOM_COUNT} OOM errors found!" >> $LOG_FILE
            fi
        else
            echo "${exp}: No log file" >> $LOG_FILE
        fi
    done
    
    # Check checkpoints
    echo "Checkpoints:" >> $LOG_FILE
    for exp in scale-m1.5b-t8b-math-fullseq scale-q1.7b-t8b-math-fullseq scale-q4b-t8b-math-fullseq; do
        COUNT=$(ls /sg-pvc/quick-distillation/checkpoints/${exp}/step_* 2>/dev/null | wc -l)
        LATEST=$(ls -d /sg-pvc/quick-distillation/checkpoints/${exp}/step_* 2>/dev/null | sort -V | tail -1 | xargs basename 2>/dev/null || echo "none")
        echo "  ${exp}: ${COUNT} checkpoints, latest: ${LATEST}" >> $LOG_FILE
    done
    
    echo "---" >> $LOG_FILE
    
    # Sleep 5 minutes
    sleep 300
done
