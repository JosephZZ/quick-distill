---
# Qwen3.5 Cross-Family Teacher Experiments

**Code**: `on_policy_distill_positional.py` (with cross-tokenizer fix for nothink_ids)
**Tokenizer**: Cross-family, Qwen3 (151K) → Qwen3.5 (248K), 86.8% overlap
**Config**: LoRA r=32, alpha=64, lr=5e-5, bs=16, n_samples=1, 3200 problems, 200 steps, pos-100

## Qwen3-1.7B (student) → Qwen3.5-2B (teacher)

**Server**: scai3 GPU 0+1
**Baseline**: Qwen3-1.7B = 69.20% avg@4

| Step | avg@4 |
|------|-------|
| 50 | 60.65 |
| **100** | **61.25** |
| 150 | 58.20 ⚠️ |
| 200 | 58.70 ⚠️ |

**Result: NEGATIVE.** Best = 61.25%, which is -7.95pp below baseline.
Degrades after step 100. Cross-family distillation with Qwen3.5 teacher HURTS Qwen3 student.

## Qwen3-4B (student) → Qwen3.5-4B (teacher)

**Server**: scai5 GPU 1+3
**Baseline**: Qwen3-4B = 77.95% avg@4

| Step | avg@4 |
|------|-------|
| **50** | **71.35** |
| 100 | 67.65 ⚠️ |
| 150 | 69.30 |
| 200 | 70.20 |

**Result: NEGATIVE.** Best = 71.35%, which is -6.6pp below baseline.
Degrades at step 100 then partially recovers. Cross-family distillation HURTS.

## Fullseq Results

### Qwen3-4B → Qwen3.5-4B fullseq
| Step | avg@4 |
|------|-------|
| 50 | 27.10 ⚠️ |
| 100 | 0.00 ⚠️⚠️ |

**Complete collapse.** Cross-tokenizer fullseq is catastrophically broken for this pair.

### Qwen3-1.7B → Qwen3.5-2B fullseq
Loss exploded at step 160 (negative KL). Also broken.

## Analysis

ALL Qwen3→Qwen3.5 experiments are negative:
- Qwen3-1.7B: -7.95pp (69.20 → 61.25)
- Qwen3-4B: -6.6pp (77.95 → 71.35)

Possible causes:
1. **Tokenizer mismatch**: 86.8% overlap means 13.2% of student tokens have no teacher equivalent
2. **Architecture mismatch**: Qwen3.5 uses different arch (qwen3_5_text) vs Qwen3
3. **Capability gap wrong direction**: Qwen3.5-2B (1.88B params) may be WEAKER than Qwen3-1.7B on math
4. **Loss on shared vocab only**: KL computed only over 53% of teacher vocab — information loss

This contrasts with successful cross-family distillation:
- Qwen2.5-Math-1.5B → Qwen3-1.7B: +12.2pp (different families but compatible tokenizer)
- gemma-2-2b-it → gemma-3-4b-it: +13.75pp math (same family, 69% overlap)
