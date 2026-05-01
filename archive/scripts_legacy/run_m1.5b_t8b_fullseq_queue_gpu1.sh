#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR" || exit 1
mkdir -p logs
LOCK_FILE="/tmp/m1.5b_t8b_fullseq_queue_gpu1.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date -Is) another instance is running, exit." >> "$REPO_DIR/logs/m1.5b_t8b_fullseq_gpu1.queue.log"
  exit 0
fi

[ -f "${HOME}/.bashrc" ] && PS1=nonempty source "${HOME}/.bashrc" 2>/dev/null || true
source "$SCRIPT_DIR/hf_models_env.sh"

STUDENT="$MATH_STUDENT_15"
TEACHER="$QWEN3_8"
LR=5e-5
LORA_R=32
LORA_ALPHA=64
BS=16
N_SAMPLES=1
MAX_NEW_TOKENS=3584
MINI_BS=4
TEACHER_MICRO_BS=1
GPU=1

MATH_SYS='Please reason step by step, and put your final answer within \\boxed{}.'
CODING_SYS='You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed.'
FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'
MASTER_LOG="$REPO_DIR/logs/m1.5b_t8b_fullseq_gpu1.queue.log"

_ts() { date -Is; }
_log() { echo "$(_ts) $*" | tee -a "$MASTER_LOG"; }
is_done() { [[ -d "$1/step_200" ]]; }

run_one() {
  local task=$1 dataset=$2 sys_prompt=$3 outdir=$4 run_name=$5 problem_field=$6
  if is_done "$outdir"; then _log "SKIP done: $run_name"; return 0; fi
  local run_log="$REPO_DIR/logs/${run_name}.log"; local attempt=0
  while true; do
    attempt=$((attempt + 1)); _log "START $run_name attempt=$attempt"
    {
      echo "===== $(_ts) START $run_name attempt=$attempt ====="
      echo "CMD: CUDA_VISIBLE_DEVICES=$GPU python on_policy_distill_positional.py --student_model $STUDENT --teacher_model $TEACHER ... --output_dir $outdir --bs $BS --max_new_tokens $MAX_NEW_TOKENS --mini_bs $MINI_BS $problem_field"
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$run_log" 2>&1

    CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL -eL python on_policy_distill_positional.py \
      --student_model "$STUDENT" --teacher_model "$TEACHER" \
      --dataset "$dataset" --num_problems 3200 \
      --bs "$BS" --n_samples "$N_SAMPLES" \
      --temperature 0.7 --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
      --save_steps 50 --log_steps 10 --eval_steps 0 \
      --student_gpu 0 --teacher_gpu 0 \
      --system_prompt "$sys_prompt" \
      --wandb_project dft-distill-scaling \
      --output_dir "$outdir" \
      --position_limit 0 \
      --wandb_run_name "$run_name" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --teacher_micro_bs "$TEACHER_MICRO_BS" \
      --mini_bs "$MINI_BS" \
      $problem_field \
      2>&1 | tee -a "$run_log"
    rc=${PIPESTATUS[0]}
    {
      echo "===== $(_ts) END $run_name attempt=$attempt rc=$rc ====="
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$run_log" 2>&1
    if [[ $rc -eq 0 ]] && is_done "$outdir"; then _log "DONE $run_name"; return 0; fi
    _log "RETRY $run_name rc=$rc (sleep 60s)"; sleep 60
  done
}

_log "QUEUE START t8b GPU1"
run_one "math" "AI-MO/NuminaMath-CoT" "$MATH_SYS" "checkpoints/scale-m1.5b-t8b-math-fullseq" "scale-m1.5b-t8b-math-fullseq" ""
run_one "coding" "coseal/CodeUltraFeedback_binarized" "$CODING_SYS" "checkpoints/scale-m1.5b-t8b-coding-fullseq" "scale-m1.5b-t8b-coding-fullseq" "--problem_field instruction"
run_one "funcall" "data/funcall/train.jsonl" "$FUNCALL_SYS" "checkpoints/scale-m1.5b-t8b-funcall-fullseq" "scale-m1.5b-t8b-funcall-fullseq" "--problem_field problem"
_log "QUEUE DONE t8b GPU1"
