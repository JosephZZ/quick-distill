# Phase 5: Think Mode Distillation Notes

## Key Template Differences

### Non-think mode (all previous experiments):
- Student prompt ends with: `<|im_start|>assistant\n<think>\n\n</think>\n\n`
- Teacher scoring: insert `nothink_ids = encode("<think>\n\n</think>\n\n")` between prompt and response
- Student generates plain text (no `<think>` block)

### Think mode (Phase 5):
- Student prompt ends with: `<|im_start|>assistant\n` (NO nothink prefix)
- Student generates: `<think>...reasoning...</think>\nfinal answer`
- Teacher scoring: do NOT insert nothink_ids (or insert empty list)
- Verify student response starts with `<think>` tokens

## Code Changes Needed for Phase 5
1. In `build_prompt`: use `enable_thinking=True` (default) instead of `enable_thinking=False`
2. In `query_teacher_hf*`: pass `nothink_ids=[]` instead of the think/nothink tokens
3. `max_new_tokens` must be ≥ 3000 (CoT is long)
4. Use 2 GPUs (48GB each, A6000 on UCLACG or scai4)
5. `mini_bs=1` required (CoT sequences are long)

## Verification Checklist
- [ ] Student response starts with `<think>`
- [ ] KL values are positive
- [ ] Loss is decreasing normally
- [ ] max_new_tokens ≥ 3000
- [ ] Memory fits on target GPU (48GB)

## Models
- Student: Qwen3-1.7B (with LoRA)
- Teacher: Qwen3.5-4B (or Qwen3-4B if 3.5 unavailable)
