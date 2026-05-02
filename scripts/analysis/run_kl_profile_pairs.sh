#!/bin/bash
# Run cross-tokenizer KL profiles for 2 model-family pairs on GPU 1.
# Each pair: ~50 problems x ~512 tokens, takes ~20-30 min on a 49 GB card.

set -uo pipefail

BASE="/zhi_backup/ziheng/quick-distillation/quick-distillation"
GPU=1

cd "$BASE"
mkdir -p logs docs/kl_profile_xfamily

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache
export CUDA_VISIBLE_DEVICES=$GPU

run_pair () {
    local NAME=$1; local STUDENT=$2; local TEACHER=$3
    local OUT="docs/kl_profile_xfamily/${NAME}.json"
    local LOG="logs/kl_profile_${NAME}.log"
    echo "[$(date)] === $NAME: $STUDENT -> $TEACHER ==="
    python scripts/analysis/pretrain_kl_profile_xtok.py \
        --student_model "$STUDENT" \
        --teacher_model "$TEACHER" \
        --num_problems 50 --max_new_tokens 512 \
        --output "$OUT" \
        > "$LOG" 2>&1
    rc=$?
    echo "[$(date)] $NAME exit=$rc out=$OUT"
    if [ $rc -ne 0 ]; then
        echo "--- last 30 lines of $LOG ---"
        tail -30 "$LOG"
    fi
}

run_pair internlm2.5-1.8B_internlm3-8B \
    "internlm/internlm2_5-1_8b-chat" \
    "internlm/internlm3-8b-instruct"

run_pair minicpm3-4B_minicpm4-8B \
    "openbmb/MiniCPM3-4B" \
    "openbmb/MiniCPM4-8B"

echo "[$(date)] all pairs done"
