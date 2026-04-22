# CoT Distillation Results (Qwen3-1.7B → Qwen3-4B, thinking mode)

Both student and teacher use thinking mode (enable_thinking=True).
LoRA r=32, n1bs16, 200 steps, 3200 problems.

## pos-100

| Step | avg@4 |
|------|-------|
| 50 | 65.00% |
| 100 | 64.20% |
| **150** | **66.80%** |
| 200 | 63.00% |

## fullseq

| Step | avg@4 |
|------|-------|
| 200 | 63.55% |
| final | **64.65%** |

## Summary

| Method | Best avg@4 |
|--------|-----------|
| **CoT pos-100** | **66.80%** (step 150) |
| CoT fullseq | 64.65% (final) |
| Delta | **+2.15pp** |

Positional distillation also works in thinking/CoT mode.
