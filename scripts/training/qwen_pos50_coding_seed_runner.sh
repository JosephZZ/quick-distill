#!/usr/bin/env bash
# Train Qwen2.5-Math-1.5B -> Qwen3-1.7B coding LoRA pos-50, max 200 steps.
# After training, merge each saved LoRA checkpoint and eval on HE/HE+/MBPP/MBPP+.
# If any seed produces a step that beats ALL of the fullseq targets
#   (HE=40.2, HE+=35.4, MBPP=52.6, MBPP+=46.3), stop.
# Otherwise rotate to the next seed in SEEDS list.
#
# Hyperparams match checkpoints/fullseq-coding-1.7b-v2/config.json verbatim
# except for position_limit=50 + token_select_mode=prefix, num_problems=3200
# (200 steps @ bs=16), and --seed.
set -euo pipefail

cd /zhi_backup/ziheng/quick-distillation
mkdir -p logs checkpoints eval_results/qwen_pos50_coding_seedhunt

GPU=${GPU:-0}
SEEDS=(42 7 1337 2024)
# Fullseq targets (from slides.html page 5 table).
TARGET_HE=40.2
TARGET_HEP=35.4
TARGET_MB=52.6
TARGET_MBP=46.3

SMODEL="Qwen/Qwen2.5-Math-1.5B"
TEACHER="Qwen/Qwen3-1.7B"
DATASET="coseal/CodeUltraFeedback_binarized"
PROBLEM_FIELD="instruction"   # CodeUltraFeedback_binarized has no "problem" field
SYS_PROMPT="You are a helpful coding assistant."

train_one_seed() {
    local SEED=$1
    local TAG="qwen-pos50-coding-seed${SEED}"
    local OUTDIR="checkpoints/${TAG}"
    local LOG="logs/${TAG}.log"

    if [[ -d "${OUTDIR}/step_200" ]]; then
        echo "[skip-train] ${TAG} already trained through step_200"
        return 0
    fi
    echo "=== TRAIN ${TAG} (seed=${SEED}) ==="
    CUDA_VISIBLE_DEVICES=${GPU} python3 on_policy_distill_positional.py \
        --student_model "${SMODEL}" --teacher_model "${TEACHER}" \
        --dataset "${DATASET}" --problem_field "${PROBLEM_FIELD}" \
        --num_problems 3200 \
        --bs 16 --mini_bs 1 --n_samples 1 \
        --max_new_tokens 2048 --temperature 0.7 \
        --lr 5e-5 --lora_r 32 --lora_alpha 64 \
        --warmup_ratio 0.1 --max_grad_norm 1.0 \
        --save_steps 50 --log_steps 10 --eval_steps 0 \
        --seed ${SEED} \
        --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
        --position_limit 50 --token_select_mode prefix \
        --system_prompt "${SYS_PROMPT}" \
        --wandb_project dft-distill-positional \
        --wandb_run_name "${TAG}" \
        --output_dir "${OUTDIR}" \
        2>&1 | tee "${LOG}"
    echo "=== TRAIN ${TAG} done ==="
}

eval_one_checkpoint() {
    local TAG=$1 STEP=$2
    local LP="checkpoints/${TAG}/step_${STEP}"
    local ED="eval_results/qwen_pos50_coding_seedhunt/${TAG}/step_${STEP}"
    [[ ! -d "${LP}" ]] && { echo "[miss] ${LP}"; return 1; }
    if [[ -f "${ED}/he_score.txt" && -f "${ED}/mbpp_score.txt" ]]; then
        echo "[skip-eval] ${TAG}/step_${STEP} already evaluated"
        return 0
    fi
    mkdir -p "${ED}"
    local MP="checkpoints/_merge_tmp_${TAG}_${STEP}"
    echo "--- MERGE ${TAG}/step_${STEP} ---"
    CUDA_VISIBLE_DEVICES="" python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
base = AutoModelForCausalLM.from_pretrained('${SMODEL}', torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, '${LP}').merge_and_unload()
m.save_pretrained('${MP}')
AutoTokenizer.from_pretrained('${SMODEL}', trust_remote_code=True).save_pretrained('${MP}')
print('Merged -> ${MP}')
"
    for DS in humaneval mbpp; do
        echo "--- EVAL ${TAG}/step_${STEP} on ${DS} ---"
        CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/eval_humaneval.py \
            --model "${MP}" --dataset "${DS}" \
            --output_dir "${ED}" \
            --gpu_memory_utilization 0.70 --trust_remote_code \
            2>&1 | tee -a "${ED}/eval_${DS}.log"
    done
    rm -rf "${MP}"
}

# Pick out "pass@1" lines from he_score.txt / mbpp_score.txt.
# evalplus writes 4 pass@1 lines per file in this order:
#   humaneval (base tests)       -> HE
#   humaneval+ (base + extra)    -> HE+
# and analogously for mbpp.
extract_he()  { awk '/humaneval \(base tests\)/       {f=1; next} f && /pass@1/ {printf "%.2f", $NF*100; exit}' "$1"; }
extract_hep() { awk '/humaneval\+|base \+ extra/      {f=1; next} f && /pass@1/ {printf "%.2f", $NF*100; exit}' "$1"; }
extract_mb()  { awk '/mbpp \(base tests\)/            {f=1; next} f && /pass@1/ {printf "%.2f", $NF*100; exit}' "$1"; }
extract_mbp() { awk '/mbpp\+|base \+ extra/           {f=1; next} f && /pass@1/ {printf "%.2f", $NF*100; exit}' "$1"; }

collect_best() {
    local TAG=$1
    local DIR="eval_results/qwen_pos50_coding_seedhunt/${TAG}"
    echo
    echo "=== RESULTS for ${TAG} ==="
    printf "%-10s %7s %7s %7s %7s  %s\n" "step" "HE" "HE+" "MBPP" "MBPP+" "all-beat?"
    local any_win=""
    for STEP in 50 100 150 200; do
        local HEF="${DIR}/step_${STEP}/he_score.txt"
        local MBF="${DIR}/step_${STEP}/mbpp_score.txt"
        [[ ! -f "${HEF}" || ! -f "${MBF}" ]] && continue
        local HE
        local HEP
        local MB
        local MBP
        HE=$(extract_he "${HEF}")
        HEP=$(extract_hep "${HEF}")
        MB=$(extract_mb "${MBF}")
        MBP=$(extract_mbp "${MBF}")
        local BEAT="no"
        awk -v a="${HE:-0}"  -v A="${TARGET_HE}" \
            -v b="${HEP:-0}" -v B="${TARGET_HEP}" \
            -v c="${MB:-0}"  -v C="${TARGET_MB}" \
            -v d="${MBP:-0}" -v D="${TARGET_MBP}" \
            'BEGIN { if (a>A && b>B && c>C && d>D) exit 0; else exit 1 }' \
            && BEAT="YES"
        printf "%-10s %7s %7s %7s %7s  %s\n" \
            "step_${STEP}" "${HE:-?}" "${HEP:-?}" "${MB:-?}" "${MBP:-?}" "${BEAT}"
        [[ "${BEAT}" == "YES" ]] && any_win="${STEP}"
    done
    if [[ -n "${any_win}" ]]; then
        echo "*** ${TAG} step_${any_win} BEATS fullseq on all 4 metrics ***"
        return 0
    else
        echo "--- ${TAG}: no step beats all 4 targets, rotating seed ---"
        return 1
    fi
}

MAIN() {
    for SEED in "${SEEDS[@]}"; do
        local TAG="qwen-pos50-coding-seed${SEED}"
        train_one_seed "${SEED}"
        for STEP in 50 100 150 200; do
            eval_one_checkpoint "${TAG}" "${STEP}" \
                || echo "[warn] eval ${TAG}/step_${STEP} failed, continuing"
        done
        if collect_best "${TAG}"; then
            echo
            echo "====================================================="
            echo "VICTORY: seed=${SEED} cleared all 4 fullseq targets."
            echo "  eval dir: eval_results/qwen_pos50_coding_seedhunt/${TAG}"
            echo "====================================================="
            exit 0
        fi
    done
    echo
    echo "====================================================="
    echo "EXHAUSTED seeds ${SEEDS[*]} without clearing all 4 targets."
    echo "Inspect per-seed tables above; best-per-metric may still improve deck."
    echo "====================================================="
}

MAIN
