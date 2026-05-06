#!/usr/bin/env bash
# Re-run Gemma-3-4B teacher BFCL funcall baseline using the gemma3 venv (vLLM 0.11.0).
set -euo pipefail

PY=/zhi_backup/ziheng/venvs/gemma3/bin/python
MODEL=/zhi_backup/ziheng/hf_cache/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
OUT=/zhi_backup/ziheng/quick-distillation/eval_results/funcall-gemma3-4b-teacher-rerun
mkdir -p "$OUT"

cd /zhi_backup/ziheng/quick-distillation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

echo "=== $(date) BFCL funcall (simple+multiple, 600 problems) ==="
"$PY" eval_funcall.py \
    --model "$MODEL" \
    --output_dir "$OUT" \
    --categories simple,multiple \
    --max_new_tokens 512 --temperature 0.0 \
    --gpu_memory_utilization 0.50

echo "=== $(date) DONE ==="
echo "Summary:"
cat "$OUT/summary.json"
