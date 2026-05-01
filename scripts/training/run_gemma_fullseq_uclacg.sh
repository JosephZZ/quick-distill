#!/bin/bash
# Run gemma fullseq experiments on UCLACG (2x A6000)
# Teacher on GPU 0, Student+SGLang on GPU 1
# Then run all pending coding evals
set -e

cd /mnt/ziheng/quick-distillation
PYTHON=/mnt/ziheng/conda-sglang/bin/python
STUDENT="google/gemma-2-2b-it"
TEACHER="google/gemma-3-4b-it"
LR=5e-5
LORA_R=32
LORA_ALPHA=64

echo "=== $(date) === Starting gemma fullseq pipeline ==="

# ============================================================
# Wait for current experiment (fullseq-m1.5b-t4b-funcall) to finish
# ============================================================
echo "Waiting for PID 988478 (fullseq-m1.5b-t4b-funcall) to finish..."
while kill -0 988478 2>/dev/null; do
    sleep 60
done
echo "=== $(date) === Previous experiment done, starting gemma fullseq ==="
sleep 10  # let GPU memory settle

# ============================================================
# 1. Gemma fullseq MATH
# ============================================================
echo "=== $(date) === Starting gemma-fullseq-math ==="
CUDA_VISIBLE_DEVICES="0,1" $PYTHON on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "AI-MO/NuminaMath-CoT" --num_problems 3200 \
    --bs 16 --n_samples 1 --mini_bs 1 \
    --temperature 0.7 --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 1 --teacher_gpu 0 --vllm_gpu 0 \
    --position_limit 0 --use_sglang --sglang_gpu 0 \
    --max_new_tokens 3584 \
    --output_dir checkpoints/gemma-fullseq-math \
    --wandb_project dft-distill-gemma --wandb_run_name gemma-fullseq-math \
    --problem_field problem \
    --system_prompt "Please reason step by step, and put your final answer within \\\\boxed{}." \
    2>&1 | tee logs/gemma-fullseq-math.log

echo "=== $(date) === gemma-fullseq-math done ==="

# Eval gemma-fullseq-math (all steps)
for STEP in 50 100 150 200; do
    CKPT="checkpoints/gemma-fullseq-math/step_${STEP}"
    EVAL_DIR="checkpoints/gemma-fullseq-math/eval_step_${STEP}"
    MERGED="checkpoints/gemma-fullseq-math/_eval_merged_step_${STEP}"
    if [ -d "$CKPT" ] && [ ! -f "$EVAL_DIR/summary.json" ]; then
        echo "=== $(date) === Evaluating gemma-fullseq-math step $STEP ==="
        # Merge LoRA on CPU
        CUDA_VISIBLE_DEVICES="" $PYTHON -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('$STUDENT', torch_dtype=torch.bfloat16, trust_remote_code=True)
m = PeftModel.from_pretrained(m, '$CKPT')
m = m.merge_and_unload()
m.save_pretrained('$MERGED')
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('$STUDENT', trust_remote_code=True).save_pretrained('$MERGED')
print('Merged.')
"
        CUDA_VISIBLE_DEVICES="0" $PYTHON eval_math500.py \
            --model "$MERGED" --output_dir "$EVAL_DIR" \
            --n_samples 4 --temperature 0.7 --gpu_memory_utilization 0.85
        rm -rf "$MERGED"
        echo "=== $(date) === gemma-fullseq-math step $STEP eval done ==="
    fi
done

# ============================================================
# 2. Gemma fullseq CODING
# ============================================================
echo "=== $(date) === Starting gemma-fullseq-coding ==="
CUDA_VISIBLE_DEVICES="0,1" $PYTHON on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "coseal/CodeUltraFeedback_binarized" --num_problems 3200 \
    --bs 16 --n_samples 1 --mini_bs 1 \
    --temperature 0.7 --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 1 --teacher_gpu 0 --vllm_gpu 0 \
    --position_limit 0 --use_sglang --sglang_gpu 0 \
    --max_new_tokens 3584 \
    --output_dir checkpoints/gemma-fullseq-coding \
    --wandb_project dft-distill-gemma --wandb_run_name gemma-fullseq-coding \
    --problem_field instruction \
    --system_prompt "You are a helpful coding assistant. Write clean, correct, and well-structured code. Provide clear explanations when needed." \
    2>&1 | tee logs/gemma-fullseq-coding.log

echo "=== $(date) === gemma-fullseq-coding done ==="

# ============================================================
# 3. Gemma fullseq FUNCALL
# ============================================================
echo "=== $(date) === Starting gemma-fullseq-funcall ==="
CUDA_VISIBLE_DEVICES="0,1" $PYTHON on_policy_distill_positional.py \
    --student_model "$STUDENT" --teacher_model "$TEACHER" \
    --dataset "data/funcall/train.jsonl" --num_problems 3200 \
    --bs 16 --n_samples 1 --mini_bs 1 \
    --temperature 0.7 --lr $LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --save_steps 50 --log_steps 10 --eval_steps 0 \
    --student_gpu 1 --teacher_gpu 0 --vllm_gpu 0 \
    --position_limit 0 --use_sglang --sglang_gpu 0 \
    --max_new_tokens 3584 \
    --output_dir checkpoints/gemma-fullseq-funcall \
    --wandb_project dft-distill-gemma --wandb_run_name gemma-fullseq-funcall \
    --problem_field problem \
    --system_prompt "You are a helpful assistant with access to functions. When the user's request can be fulfilled by calling a function, respond with a JSON array of function calls like: [{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}]. If no function is needed, respond normally." \
    2>&1 | tee logs/gemma-fullseq-funcall.log

echo "=== $(date) === gemma-fullseq-funcall done ==="

# ============================================================
# 4. Eval all gemma fullseq funcall steps
# ============================================================
for STEP in 50 100 150 200; do
    CKPT="checkpoints/gemma-fullseq-funcall/step_${STEP}"
    EVAL_DIR="checkpoints/gemma-fullseq-funcall/eval_step_${STEP}"
    MERGED="checkpoints/gemma-fullseq-funcall/_eval_merged_step_${STEP}"
    if [ -d "$CKPT" ] && [ ! -f "$EVAL_DIR/summary.json" ]; then
        echo "=== $(date) === Evaluating gemma-fullseq-funcall step $STEP ==="
        CUDA_VISIBLE_DEVICES="" $PYTHON -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('$STUDENT', torch_dtype=torch.bfloat16, trust_remote_code=True)
m = PeftModel.from_pretrained(m, '$CKPT')
m = m.merge_and_unload()
m.save_pretrained('$MERGED')
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('$STUDENT', trust_remote_code=True).save_pretrained('$MERGED')
print('Merged.')
"
        CUDA_VISIBLE_DEVICES="0" $PYTHON eval_funcall.py \
            --model "$MERGED" --output_dir "$EVAL_DIR" \
            --gpu_id 0 --gpu_memory_utilization 0.85
        rm -rf "$MERGED"
        echo "=== $(date) === gemma-fullseq-funcall step $STEP eval done ==="
    fi
done

# ============================================================
# 5. Eval all gemma fullseq coding steps (HumanEval + MBPP)
# ============================================================
for STEP in 50 100 150 200; do
    CKPT="checkpoints/gemma-fullseq-coding/step_${STEP}"
    EVAL_DIR="checkpoints/gemma-fullseq-coding/eval_step_${STEP}"
    MERGED="checkpoints/gemma-fullseq-coding/_eval_merged_step_${STEP}"
    if [ -d "$CKPT" ] && [ ! -f "$EVAL_DIR/summary.json" ]; then
        echo "=== $(date) === Evaluating gemma-fullseq-coding step $STEP ==="
        CUDA_VISIBLE_DEVICES="" $PYTHON -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('$STUDENT', torch_dtype=torch.bfloat16, trust_remote_code=True)
m = PeftModel.from_pretrained(m, '$CKPT')
m = m.merge_and_unload()
m.save_pretrained('$MERGED')
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('$STUDENT', trust_remote_code=True).save_pretrained('$MERGED')
print('Merged.')
"
        CUDA_VISIBLE_DEVICES="0" $PYTHON scripts/eval_humaneval.py \
            --model "$MERGED" --dataset humaneval --output_dir "$EVAL_DIR" \
            --gpu_memory_utilization 0.85
        CUDA_VISIBLE_DEVICES="0" $PYTHON scripts/eval_humaneval.py \
            --model "$MERGED" --dataset mbpp --output_dir "$EVAL_DIR" \
            --gpu_memory_utilization 0.85
        rm -rf "$MERGED"
        echo "=== $(date) === gemma-fullseq-coding step $STEP eval done ==="
    fi
done

# ============================================================
# 6. Also eval existing gemma-pos100-coding (has stubs only)
# ============================================================
for STEP in 50 100 150 200; do
    CKPT="checkpoints/gemma-pos100-coding/step_${STEP}"
    EVAL_DIR="checkpoints/gemma-pos100-coding/eval_step_${STEP}"
    MERGED="checkpoints/gemma-pos100-coding/_eval_merged_step_${STEP}"
    SUMMARY="$EVAL_DIR/summary.json"
    # Check if summary only has {"status":"done"} (stub)
    if [ -d "$CKPT" ] && [ -f "$SUMMARY" ] && grep -q '"status"' "$SUMMARY" && ! grep -q '"avg_k"' "$SUMMARY" && ! grep -q '"HE"' "$SUMMARY"; then
        echo "=== $(date) === Evaluating gemma-pos100-coding step $STEP (was stub) ==="
        CUDA_VISIBLE_DEVICES="" $PYTHON -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('$STUDENT', torch_dtype=torch.bfloat16, trust_remote_code=True)
m = PeftModel.from_pretrained(m, '$CKPT')
m = m.merge_and_unload()
m.save_pretrained('$MERGED')
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('$STUDENT', trust_remote_code=True).save_pretrained('$MERGED')
print('Merged.')
"
        CUDA_VISIBLE_DEVICES="0" $PYTHON scripts/eval_humaneval.py \
            --model "$MERGED" --dataset humaneval --output_dir "$EVAL_DIR" \
            --gpu_memory_utilization 0.85
        CUDA_VISIBLE_DEVICES="0" $PYTHON scripts/eval_humaneval.py \
            --model "$MERGED" --dataset mbpp --output_dir "$EVAL_DIR" \
            --gpu_memory_utilization 0.85
        rm -rf "$MERGED"
        echo "=== $(date) === gemma-pos100-coding step $STEP eval done ==="
    fi
done

echo "=== $(date) === ALL DONE ==="
