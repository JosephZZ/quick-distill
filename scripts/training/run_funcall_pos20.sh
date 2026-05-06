#!/bin/bash
# Qwen 1.5-Math → 1.7B, funcall, pos-20 (N=20 prefix).
# Mirrors fullseq-funcall-1.7b-v2 hyperparams but with --position_limit 20.
set -eo pipefail

cd /zhi_backup/ziheng/quick-distillation
mkdir -p logs

export HF_HOME=/zhi_backup/ziheng/hf_cache
export HF_HUB_CACHE=/zhi_backup/ziheng/hf_cache/hub
export TRANSFORMERS_CACHE=/zhi_backup/ziheng/hf_cache

STUDENT="Qwen/Qwen2.5-Math-1.5B"
TEACHER="Qwen/Qwen3-1.7B"

FUNCALL_SYS='You are a helpful assistant with access to functions. When the user'"'"'s request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{"name": "function_name", "arguments": {"arg1": "value1"}}]. If no function is needed, respond normally.'

OUTDIR="checkpoints/funcall-pos20-1.7b"
RUN_NAME="funcall-pos20-1.7b"

CUDA_VISIBLE_DEVICES=0 python on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "data/funcall/train.jsonl" --problem_field problem \
    --num_problems 3200 \
    --bs 16 --n_samples 1 --mini_bs 1 \
    --max_new_tokens 2048 \
    --temperature 0.7 --lr 5e-5 --lora_r 32 --lora_alpha 64 \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 0 --teacher_gpu 0 --vllm_gpu 0 --single_gpu \
    --system_prompt "$FUNCALL_SYS" \
    --wandb_project dft-distill-positional \
    --output_dir "$OUTDIR" \
    --token_select_mode prefix \
    --position_limit 20
