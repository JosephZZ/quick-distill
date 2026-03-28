#!/bin/bash
# Batch merge and eval for Config A and C math checkpoints
set -e

STUDENT_A="/sg-pvc/hfmodels/Qwen_Qwen2.5-Math-1.5B"
STUDENT_C="/sg-pvc/hfmodels/Qwen_Qwen3-1.7B"

echo "=== Merging Config A checkpoints (CPU) ==="
for STEP in 50 100 150 200; do
  LORA_PATH="checkpoints/scale-m1.5b-t4b-math-fullseq/step_${STEP}"
  MERGED_PATH="checkpoints/scale-m1.5b-t4b-math-fullseq/_eval_merged_step_${STEP}"
  
  if [ -f "${LORA_PATH}/adapter_model.safetensors" ] && [ ! -d "$MERGED_PATH" ]; then
    echo "Merging A step ${STEP}..."
    CUDA_VISIBLE_DEVICES="" python3 << PY &
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("$STUDENT_A", torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, "$LORA_PATH")
merged = model.merge_and_unload()
merged.save_pretrained("$MERGED_PATH")
AutoTokenizer.from_pretrained("$STUDENT_A", trust_remote_code=True).save_pretrained("$MERGED_PATH")
print(f"Merged A step ${STEP}")
PY
  fi
done

echo "=== Merging Config C checkpoints (CPU) ==="
for STEP in 50 100 150 200; do
  LORA_PATH="checkpoints/scale-q1.7b-t4b-math-fullseq/step_${STEP}"
  MERGED_PATH="checkpoints/scale-q1.7b-t4b-math-fullseq/_eval_merged_step_${STEP}"
  
  if [ -f "${LORA_PATH}/adapter_model.safetensors" ] && [ ! -d "$MERGED_PATH" ]; then
    echo "Merging C step ${STEP}..."
    CUDA_VISIBLE_DEVICES="" python3 << PY &
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("$STUDENT_C", torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, "$LORA_PATH")
merged = model.merge_and_unload()
merged.save_pretrained("$MERGED_PATH")
AutoTokenizer.from_pretrained("$STUDENT_C", trust_remote_code=True).save_pretrained("$MERGED_PATH")
print(f"Merged C step ${STEP}")
PY
  fi
done

wait
echo "=== All merges complete ==="
echo "To eval, run manually (sequentially to avoid GPU OOM):"
echo "for step in 50 100 150 200; do"
echo "  CUDA_VISIBLE_DEVICES=0 python eval_math500.py --model checkpoints/scale-m1.5b-t4b-math-fullseq/_eval_merged_step_\$step --output_dir checkpoints/scale-m1.5b-t4b-math-fullseq/eval_step_\$step"
echo "  CUDA_VISIBLE_DEVICES=1 python eval_math500.py --model checkpoints/scale-q1.7b-t4b-math-fullseq/_eval_merged_step_\$step --output_dir checkpoints/scale-q1.7b-t4b-math-fullseq/eval_step_\$step"
echo "done"
