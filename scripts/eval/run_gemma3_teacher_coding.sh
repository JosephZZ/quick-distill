#!/usr/bin/env bash
# Gemma-3-4B teacher coding baseline (HumanEval+ / MBPP+).
# Uses the gemma3-specific venv (vLLM 0.11.0) since the default env (vLLM 0.8.5)
# fails on Gemma3 multimodal tokenizer init.
set -euo pipefail

PY=/zhi_backup/ziheng/venvs/gemma3/bin/python
MODEL=/zhi_backup/ziheng/hf_cache/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
OUT=/zhi_backup/ziheng/quick-distillation/eval_results/coding-gemma3-4b-teacher
mkdir -p "$OUT"

cd /zhi_backup/ziheng/quick-distillation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

echo "=== $(date) generation: HumanEval ==="
"$PY" scripts/eval_humaneval.py \
    --model "$MODEL" --dataset humaneval \
    --output_dir "$OUT" \
    --n_samples 1 --temperature 0.0 --max_tokens 512 \
    --gpu_memory_utilization 0.50 --trust_remote_code

echo "=== $(date) generation: MBPP ==="
"$PY" scripts/eval_humaneval.py \
    --model "$MODEL" --dataset mbpp \
    --output_dir "$OUT" \
    --n_samples 1 --temperature 0.0 --max_tokens 512 \
    --gpu_memory_utilization 0.50 --trust_remote_code

MODEL_NAME=$(echo "$MODEL" | tr '/' -- | sed 's/--/--/g')
HE_JSONL="$OUT/humaneval_${MODEL//\//--}.jsonl"
MBPP_JSONL="$OUT/mbpp_${MODEL//\//--}.jsonl"

echo "=== $(date) evalplus scoring HumanEval ==="
"$PY" -m evalplus.evaluate --dataset humaneval --samples "$HE_JSONL" 2>&1 | tee "$OUT/he_score.txt"

echo "=== $(date) evalplus scoring MBPP ==="
"$PY" -m evalplus.evaluate --dataset mbpp --samples "$MBPP_JSONL" 2>&1 | tee "$OUT/mbpp_score.txt"

echo "=== $(date) DONE ==="
echo "Summary:"
grep -E "pass@1|HumanEval|MBPP|base|plus" "$OUT/he_score.txt" "$OUT/mbpp_score.txt" || true
