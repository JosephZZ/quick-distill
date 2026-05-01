#!/bin/bash
# Run the full top-token selection series on GPU 1 only
# This script runs the remaining experiments: top_entropy_teacher-math

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
mkdir -p logs

[ -f "${HOME}/.bashrc" ] && PS1=nonempty source "${HOME}/.bashrc" 2>/dev/null || true
source "$SCRIPT_DIR/scripts/hf_models_env.sh"

PYTHON_BIN="${PYTHON_BIN:-/sg-pvc/miniconda3/bin/python}"
STUDENT="$MATH_STUDENT_15"
TEACHER="$QWEN3_17"

K=100
LR=5e-5
LORA_R=32
LORA_ALPHA=64

MATH_EXTRA="--use_vllm --max_new_tokens 2048 --teacher_micro_bs 4 --vllm_gpu_util 0.70"
MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'

echo "========== Running remaining Top-Token Selection Experiments (GPU 1 only) =========="

# Run top_entropy_teacher on math using GPU 1 (teacher_gpu=1, student_gpu=1)
echo "=== Training top_entropy_teacher on MATH (K=100) on GPU 1 ==="
CUDA_VISIBLE_DEVICES="1" $PYTHON_BIN on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems 3200 \
    --bs 16 --n_samples 1 \
    --temperature 0.7 --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 1 --teacher_gpu 1 \
    --system_prompt "$MATH_SYS" \
    --wandb_project dft-distill-token-select \
    --output_dir "checkpoints/token-select-k100-topent-teacher-math" \
    --position_limit $K \
    --token_select_mode "top_entropy_teacher" \
    --wandb_run_name "token-select-k100-topent-teacher-math" \
    $MATH_EXTRA \
    2>&1 | tee "logs/token-select-k100-topent-teacher-math.log"

echo "=== top_entropy_teacher-math training completed ==="

# Now run evaluations for all completed token-select experiments on GPU 1
echo "=== Running MATH-500 evaluations for all token-select checkpoints on GPU 1 ==="

for EXP in token-select-k100-topkl-math token-select-k100-topent-student-math token-select-k100-topent-teacher-math; do
    if [ -d "checkpoints/$EXP" ]; then
        echo "=== Evaluating $EXP on GPU 1 ==="
        CUDA_VISIBLE_DEVICES="1" $PYTHON_BIN -m scripts.eval_token_select_checkpoints "$EXP" --gpu 1 || true
    fi
done

echo "=== All top-token selection experiments and evals completed! ==="
echo "Checkpoints in: checkpoints/token-select-k*"
echo "Logs in: logs/token-select-*.log"
