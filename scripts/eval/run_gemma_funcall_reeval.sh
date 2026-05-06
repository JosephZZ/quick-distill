#!/usr/bin/env bash
# Re-eval all Gemma BFCL funcall: student baseline + every LoRA checkpoint,
# using the gemma3 venv (vLLM 0.11.0). The original eval ran against vLLM 0.8.5
# which had the boi_token tokenizer bug and scored Gemma teacher 25.0 instead
# of the correct 72.83 — so all student/LoRA numbers from that pipeline are
# also suspect.
set -euo pipefail

PY=/zhi_backup/ziheng/venvs/gemma3/bin/python
STUDENT=/mnt/ziheng/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
echo "STUDENT=$STUDENT"
ls "$STUDENT" >/dev/null  # sanity check

cd /zhi_backup/ziheng/quick-distillation
export TOKENIZERS_PARALLELISM=false
export GPU=${CUDA_VISIBLE_DEVICES:-0}

OUT_BASE=/zhi_backup/ziheng/quick-distillation/eval_results/gemma_funcall_rerun
mkdir -p "$OUT_BASE"

run_eval () {
    local model_path="$1"
    local out_dir="$2"
    mkdir -p "$out_dir"
    if [[ -f "$out_dir/summary.json" ]]; then
        echo "[skip] $out_dir already has summary.json"
        return 0
    fi
    echo "=== $(date) eval: $model_path -> $out_dir ==="
    CUDA_VISIBLE_DEVICES=$GPU "$PY" eval_funcall.py \
        --model "$model_path" \
        --output_dir "$out_dir" \
        --categories simple,multiple \
        --max_new_tokens 512 --temperature 0.0 \
        --gpu_memory_utilization 0.50
}

merge_lora () {
    local lora_path="$1"
    local merged_path="$2"
    if [[ -d "$merged_path" ]]; then
        echo "[skip merge] $merged_path exists"
        return 0
    fi
    echo "[merge] $lora_path -> $merged_path"
    CUDA_VISIBLE_DEVICES="" "$PY" - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("$STUDENT", torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
m = PeftModel.from_pretrained(base, "$lora_path")
merged = m.merge_and_unload()
merged.save_pretrained("$merged_path")
AutoTokenizer.from_pretrained("$STUDENT", trust_remote_code=True).save_pretrained("$merged_path")
print("merged ok")
PY
}

# 1. Student baseline
echo "######## STUDENT BASELINE ########"
run_eval "$STUDENT" "$OUT_BASE/baseline"

# 2. LoRA checkpoints — for each experiment & each step
for EXP in v1-gemma-funcall-pos50 v1-gemma-funcall-pos100 v1-gemma-funcall-fullseq; do
    for STEP in 50 100 150 200; do
        LORA=checkpoints/$EXP/step_$STEP
        if [[ ! -f "$LORA/adapter_model.safetensors" ]]; then
            echo "[no checkpoint] $LORA"
            continue
        fi
        MERGED=checkpoints/$EXP/_eval_merged_step_$STEP
        OUT=$OUT_BASE/$EXP/step_$STEP
        merge_lora "$LORA" "$MERGED"
        run_eval "$MERGED" "$OUT"
        # don't keep merged — they're huge
        rm -rf "$MERGED"
    done
done

echo "########  DONE  ########"
echo "Summaries:"
for f in $(find $OUT_BASE -name summary.json | sort); do
    echo "--- $f ---"
    cat "$f" | python -c "import sys,json; d=json.load(sys.stdin); print(f\"  full_acc={d.get('full_acc',0)*100 if d.get('full_acc',0)<=1 else d.get('full_acc'):.2f}  name_acc={d.get('name_acc',0)*100 if d.get('name_acc',0)<=1 else d.get('name_acc'):.2f}  parse={d.get('parse_rate',0)*100 if d.get('parse_rate',0)<=1 else d.get('parse_rate'):.2f}\")" 2>/dev/null || cat "$f"
done
