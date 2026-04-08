# Phase 4: Qwen3-4B → Qwen3-8B (Same-Family Distillation)

**Code**: `on_policy_distill_positional_v1.py` (patched with enable_thinking=False)
**Student**: Qwen3-4B | **Teacher**: Qwen3-8B
**Config**: LoRA r=32, alpha=64, lr=5e-5, bs=16, mini_bs=1, n_samples=1, 3200 problems, 200 steps
**Server**: scai3, GPU 0 (student) + GPU 1 (teacher), A100-SXM4-40GB × 2
**Checkpoint base**: `/home/antarachugh/idountang/quick-distillation/checkpoints/v1-pos{50,100}-q3-4b-t8b-{math,coding}/`

## Baselines (Qwen3-4B)
| MATH-500 avg@4 | HE pass@1 |
|----------------|-----------|
| 77.95% | 73.17% |

## Math (MATH-500, avg@4)

| Method | Step 50 | Step 100 | Step 150 | Step 200 | Best |
|--------|---------|----------|----------|----------|------|
| pos-50 | 73.25 | 73.40 | 72.75 | **73.80** | **73.80** (-4.2pp) |
| pos-100 | 73.65 | 73.30 | 73.10 | **74.15** | **74.15** (-3.8pp) |

## Coding (HumanEval pass@1)

| Method | Step 50 | Step 100 | Step 150 | Step 200 | Best |
|--------|---------|----------|----------|----------|------|
| pos-50 | 67.68 | **69.51** | 69.51 | 69.51 | **69.51** (-3.7pp) |
| pos-100 | 70.73 | **72.00** | 72.00 | pending | **72.00** (-1.2pp) |

## Key Finding

**Same-family distillation (Qwen3-4B → Qwen3-8B) is uniformly negative.**
- Math: -3.8 to -4.2pp below baseline
- Coding: -1.2 to -3.7pp below baseline
- The teacher (8B) and student (4B) are too similar in architecture and training data
- Insufficient capability gap to drive improvement
- Consistent with scaling_results.md Config C/D/E results

## Note on Qwen3.5 Teacher

Qwen3.5-4B is a **multimodal VLM** (vision+language), not a pure text model:
- Architecture: `Qwen3_5ForConditionalGeneration` with vision encoder
- Vocab: 248,320 (vs Qwen3's 151,643) — incompatible tokenizer
- Requires transformers 5.5+ dev
- **Not suitable for pure text distillation**
