#!/bin/bash
set -uo pipefail
cd /sg-pvc/quick-distillation

LOG=logs/token_select_live_watch.log
PY=/sg-pvc/miniconda3/bin/python

ts(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

is_running_math(){ pgrep -f "on_policy_distill_positional.py .*token-select-k100-topent-student-math" >/dev/null 2>&1; }
is_running_funcall(){ pgrep -f "on_policy_distill_positional.py .*token-select-k100-topkl-funcall" >/dev/null 2>&1; }

start_math(){
  log "start math topent-student on GPU1"
  nohup env CUDA_VISIBLE_DEVICES=1 "$PY" on_policy_distill_positional.py \
    --student_model /sg-pvc/hfmodels/Qwen_Qwen2.5-Math-1.5B \
    --teacher_model /sg-pvc/hfmodels/Qwen_Qwen3-1.7B \
    --dataset AI-MO/NuminaMath-CoT --num_problems 3200 \
    --bs 16 --n_samples 1 --temperature 0.7 --lr 5e-5 --lora_r 32 --lora_alpha 64 \
    --save_steps 50 --log_steps 10 --eval_steps 0 --student_gpu 0 --teacher_gpu 0 \
    --system_prompt "Please reason step by step, and put your final answer within \\boxed{}." \
    --wandb_project dft-distill-token-select --output_dir checkpoints/token-select-k100-topent-student-math \
    --position_limit 100 --token_select_mode top_entropy_student \
    --wandb_run_name token-select-k100-topent-student-math \
    --use_vllm --max_new_tokens 2048 --teacher_micro_bs 4 \
    >> logs/token-select-k100-topent-student-math.log 2>&1 &
}

start_funcall(){
  log "start funcall topkl on GPU0"
  nohup env CUDA_VISIBLE_DEVICES=0 "$PY" on_policy_distill_positional.py \
    --student_model /sg-pvc/hfmodels/Qwen_Qwen2.5-Math-1.5B \
    --teacher_model /sg-pvc/hfmodels/Qwen_Qwen3-1.7B \
    --dataset data/funcall/train.jsonl --num_problems 3200 \
    --bs 16 --n_samples 1 --mini_bs 4 --temperature 0.7 --lr 5e-5 --lora_r 32 --lora_alpha 64 \
    --save_steps 50 --log_steps 10 --eval_steps 0 --student_gpu 0 --teacher_gpu 0 \
    --system_prompt "You are a helpful assistant with access to functions. When the user's request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}]. If no function is needed, respond normally." \
    --wandb_project dft-distill-token-select --output_dir checkpoints/token-select-k100-topkl-funcall \
    --position_limit 100 --token_select_mode top_kl \
    --wandb_run_name token-select-k100-topkl-funcall-retry2 \
    --use_vllm --vllm_gpu_util 0.60 --max_new_tokens 512 --teacher_micro_bs 2 --problem_field problem \
    >> logs/token-select-k100-topkl-funcall-retry.log 2>&1 &
}

log "live watchdog started"
while true; do
  is_running_math || start_math
  is_running_funcall || start_funcall

  gstat=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' '; ')
  mstep=$(ls checkpoints/token-select-k100-topent-student-math 2>/dev/null | rg '^step_' -n -S | wc -l)
  fstep=$(ls checkpoints/token-select-k100-topkl-funcall 2>/dev/null | rg '^step_' -n -S | wc -l)
  log "alive math=$(is_running_math && echo yes || echo no) funcall=$(is_running_funcall && echo yes || echo no) math_steps=$mstep funcall_steps=$fstep gpu=[$gstat]"
  sleep 60
done
