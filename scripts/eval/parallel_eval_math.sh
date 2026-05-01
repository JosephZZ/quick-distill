#!/bin/bash
# Parallel eval for math checkpoints - runs when GPU is free
set -e

REPO="/sg-pvc/quick-distillation"
cd "$REPO"

# Config A evals (GPU 1)
echo "=== Starting Config A evals on GPU 1 ==="
for STEP in 50 100 150 200; do
  MERGED="checkpoints/scale-m1.5b-t4b-math-fullseq/_eval_merged_step_${STEP}"
  EVAL_DIR="checkpoints/scale-m1.5b-t4b-math-fullseq/eval_step_${STEP}"
  
  if [ -d "$MERGED" ] && [ ! -f "$EVAL_DIR/summary.json" ]; then
    echo "Eval A step ${STEP}..."
    CUDA_VISIBLE_DEVICES=1 nohup python eval_math500.py \
      --model "$MERGED" \
      --output_dir "$EVAL_DIR" \
      --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.50 \
      >> logs/eval_a_step${STEP}.log 2>&1 &
    echo "  PID: $!"
    sleep 10  # stagger starts to avoid OOM
  else
    echo "Skip A step ${STEP} (already done or not merged)"
  fi
done

# Config C evals (GPU 0)
echo "=== Starting Config C evals on GPU 0 ==="
for STEP in 50 100 150 200; do
  MERGED="checkpoints/scale-q1.7b-t4b-math-fullseq/_eval_merged_step_${STEP}"
  EVAL_DIR="checkpoints/scale-q1.7b-t4b-math-fullseq/eval_step_${STEP}"
  
  if [ -d "$MERGED" ] && [ ! -f "$EVAL_DIR/summary.json" ]; then
    echo "Eval C step ${STEP}..."
    CUDA_VISIBLE_DEVICES=0 nohup python eval_math500.py \
      --model "$MERGED" \
      --output_dir "$EVAL_DIR" \
      --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.50 \
      >> logs/eval_c_step${STEP}.log 2>&1 &
    echo "  PID: $!"
    sleep 10
  else
    echo "Skip C step ${STEP} (already done or not merged)"
  fi
done

echo "=== All evals launched ==="
ps aux | grep "eval_math500" | grep -v grep | wc -l && echo "eval processes running"
