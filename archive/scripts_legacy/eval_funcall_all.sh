#!/bin/bash
# Evaluate all funcall checkpoints on BFCL
# Merges LoRA → base, runs eval, saves results
set -e

BASE_DIR="/CGLab/ziheng/projects/dft-distill"
cd "$BASE_DIR"

STUDENT="Qwen/Qwen2.5-Math-1.5B"
EVAL_DATA="data/funcall/eval_bfcl.jsonl"
GPU_ID=1
CATEGORIES="simple,multiple"

merge_and_eval() {
    local CKPT_DIR=$1
    local STEP=$2
    local EXPERIMENT=$3

    local LORA_PATH="$CKPT_DIR/step_${STEP}"
    local MERGED_PATH="$CKPT_DIR/_eval_merged_step_${STEP}"
    local EVAL_DIR="$CKPT_DIR/eval_step_${STEP}"

    if [ -f "$EVAL_DIR/summary.json" ]; then
        echo "=== $EXPERIMENT step $STEP already evaluated, skipping ==="
        return
    fi

    if [ ! -d "$LORA_PATH" ]; then
        echo "=== $EXPERIMENT step $STEP not found, skipping ==="
        return
    fi

    echo "=== Merging $EXPERIMENT step $STEP ==="
    # Merge on CPU to leave GPU free for vLLM
    CUDA_VISIBLE_DEVICES="" python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print('Loading base model...')
base = AutoModelForCausalLM.from_pretrained('$STUDENT', torch_dtype=torch.bfloat16)
print('Loading LoRA from $LORA_PATH...')
model = PeftModel.from_pretrained(base, '$LORA_PATH')
print('Merging...')
merged = model.merge_and_unload()
print('Saving to $MERGED_PATH...')
merged.save_pretrained('$MERGED_PATH')
tokenizer = AutoTokenizer.from_pretrained('$STUDENT', trust_remote_code=True)
tokenizer.save_pretrained('$MERGED_PATH')
print('Done merging.')
"

    echo "=== Evaluating $EXPERIMENT step $STEP ==="
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_funcall.py \
        --model "$MERGED_PATH" \
        --eval_data "$EVAL_DATA" \
        --output_dir "$EVAL_DIR" \
        --gpu_id 0 \
        --gpu_memory_utilization 0.85 \
        --categories "$CATEGORIES"

    echo "=== Cleaning up merged model ==="
    rm -rf "$MERGED_PATH"

    echo "=== $EXPERIMENT step $STEP eval done ==="
    cat "$EVAL_DIR/summary.json" | python -m json.tool
}

# First, eval baseline (no distillation)
BASELINE_EVAL="checkpoints/funcall-baseline/eval"
if [ ! -f "$BASELINE_EVAL/summary.json" ]; then
    echo "=== Evaluating baseline (no distillation) ==="
    mkdir -p "$BASELINE_EVAL"
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_funcall.py \
        --model "$STUDENT" \
        --eval_data "$EVAL_DATA" \
        --output_dir "$BASELINE_EVAL" \
        --gpu_id 0 \
        --gpu_memory_utilization 0.85 \
        --categories "$CATEGORIES"
    echo "=== Baseline eval done ==="
    cat "$BASELINE_EVAL/summary.json" | python -m json.tool
fi

# Eval all LoRA funcall experiments at all steps
for EXPERIMENT in funcall-pos50-n1 funcall-pos100-n1 funcall-pos150-n1 funcall-pos200-n1 funcall-fullseq-n1; do
    CKPT_DIR="checkpoints/$EXPERIMENT"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "=== $EXPERIMENT not found, skipping ==="
        continue
    fi
    for STEP in 50 100 150 200; do
        merge_and_eval "$CKPT_DIR" "$STEP" "$EXPERIMENT"
    done
done

# Eval all FullFT funcall experiments at all steps (no merge needed)
eval_fullft() {
    local CKPT_DIR=$1
    local STEP=$2
    local EXPERIMENT=$3

    local MODEL_PATH="$CKPT_DIR/step_${STEP}"
    local EVAL_DIR="$CKPT_DIR/eval_step_${STEP}"

    if [ -f "$EVAL_DIR/summary.json" ]; then
        echo "=== $EXPERIMENT step $STEP already evaluated, skipping ==="
        return
    fi

    if [ ! -d "$MODEL_PATH" ]; then
        echo "=== $EXPERIMENT step $STEP not found, skipping ==="
        return
    fi

    echo "=== Evaluating $EXPERIMENT step $STEP (FullFT, no merge needed) ==="
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_funcall.py \
        --model "$MODEL_PATH" \
        --eval_data "$EVAL_DATA" \
        --output_dir "$EVAL_DIR" \
        --gpu_id 0 \
        --gpu_memory_utilization 0.85 \
        --categories "$CATEGORIES"

    echo "=== $EXPERIMENT step $STEP eval done ==="
    cat "$EVAL_DIR/summary.json" | python -m json.tool
}

for EXPERIMENT in funcall-pos50-n1-fullft funcall-pos100-n1-fullft funcall-pos150-n1-fullft funcall-pos200-n1-fullft funcall-fullseq-n1-fullft; do
    CKPT_DIR="checkpoints/$EXPERIMENT"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "=== $EXPERIMENT not found, skipping ==="
        continue
    fi
    for STEP in 50 100 150 200; do
        eval_fullft "$CKPT_DIR" "$STEP" "$EXPERIMENT"
    done
done

# Also eval teacher model
TEACHER_EVAL="checkpoints/funcall-teacher/eval"
if [ ! -f "$TEACHER_EVAL/summary.json" ]; then
    echo "=== Evaluating teacher (Qwen3-1.7B) ==="
    mkdir -p "$TEACHER_EVAL"
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_funcall.py \
        --model "Qwen/Qwen3-1.7B" \
        --eval_data "$EVAL_DATA" \
        --output_dir "$TEACHER_EVAL" \
        --gpu_id 0 \
        --gpu_memory_utilization 0.85 \
        --categories "$CATEGORIES"
    echo "=== Teacher eval done ==="
    cat "$TEACHER_EVAL/summary.json" | python -m json.tool
fi

echo ""
echo "=== ALL EVALUATIONS COMPLETE ==="
echo ""
echo "=== SUMMARY ==="
echo ""

# Print summary table
python -c "
import json, os, glob

results = []
# Baseline
bf = 'checkpoints/funcall-baseline/eval/summary.json'
if os.path.exists(bf):
    with open(bf) as f:
        d = json.load(f)
    results.append(('Baseline (Qwen2.5-Math-1.5B)', '-', d['name_acc'], d['full_acc'], d['parse_rate']))

# Teacher
tf = 'checkpoints/funcall-teacher/eval/summary.json'
if os.path.exists(tf):
    with open(tf) as f:
        d = json.load(f)
    results.append(('Teacher (Qwen3-1.7B)', '-', d['name_acc'], d['full_acc'], d['parse_rate']))

# Experiments
for exp in ['funcall-pos50-n1', 'funcall-pos100-n1', 'funcall-pos150-n1', 'funcall-pos200-n1', 'funcall-fullseq-n1',
            'funcall-pos50-n1-fullft', 'funcall-pos100-n1-fullft', 'funcall-pos150-n1-fullft', 'funcall-pos200-n1-fullft', 'funcall-fullseq-n1-fullft']:
    for step in [50, 100, 150, 200]:
        sf = f'checkpoints/{exp}/eval_step_{step}/summary.json'
        if os.path.exists(sf):
            with open(sf) as f:
                d = json.load(f)
            results.append((exp, str(step), d['name_acc'], d['full_acc'], d['parse_rate']))

print(f\"{'Model':<35} {'Step':<6} {'Name Acc':<10} {'Full Acc':<10} {'Parse Rate':<10}\")
print('-' * 75)
for name, step, nacc, facc, pr in results:
    print(f'{name:<35} {step:<6} {nacc:<10.1%} {facc:<10.1%} {pr:<10.1%}')
"
