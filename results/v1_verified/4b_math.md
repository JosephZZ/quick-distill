# 4B Teacher Math Results

**Student**: Qwen2.5-Math-1.5B | **Teacher**: Qwen3-4B
**Dataset**: AI-MO/NuminaMath-CoT
**Eval**: MATH-500, n_samples=4, temperature=0.7

## V1 Verified (Pos experiments)

**Code**: `on_policy_distill_positional_v1.py` (commit `bd81eb6`)
**Server**: scai4, A6000 48GB × 2
**Config**: LoRA r=32, alpha=64, lr=5e-5, bs=16, mini_bs=1, n_samples=1, 3200 problems, 200 steps

### Pos-50
| Step | avg@4 |
|------|-------|
| 150 | **61.70%** |
**Checkpoint**: `checkpoints/v1-pos50-m1.5b-t4b-math/` (scai4)

### Pos-100
| Step | avg@4 |
|------|-------|
| 200 | **64.20%** |
**Checkpoint**: `checkpoints/v1-pos100-m1.5b-t4b-math/` (scai4)

## Scaling Experiments (Potentially non-V1)

**Code**: `on_policy_distill_positional.py` (newer code with batch path — may have KL bugs)
**Server**: scai3/scai4
**Config**: LoRA r=32, pos-100, n_samples=1, chunk_size=16, 3200 problems, 200 steps

### Pos-100 (Config A from scaling_results.md)

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 65.50% | — | 79.80% |
| 100 | 66.20% | — | 80.40% |
| **150** | **68.95%** | — | **81.00%** |
| 200 | 67.20% | — | 80.80% |

**⚠️ WARNING**: 68.95% is from potentially buggy code. V1 verified value is 64.20%.

### Full-seq (from scaling_results.md section 6.1)

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| **50** | **67.45%** | 73.60% | **80.60%** |
| 100 | 54.45% | 69.80% | 79.60% |
| 150 | 58.85% | 73.00% | 78.80% |
| 200 | 55.05% | 72.60% | 78.40% |

**⚠️ WARNING**: From potentially buggy code. Severe degradation after step 50.

## V1 Fullseq Re-run
**Server**: scai4, GPU 5
**Status**: TRAINING IN PROGRESS (~step 10/200, ~177s/step)
**Checkpoint**: `checkpoints/v1-fullseq-m1.5b-t4b-math/` (scai4)

## Summary

| Config | Best avg@4 | Source | Reliable? |
|--------|-----------|--------|-----------|
| Pos-50 | 61.70% | V1 verified | ✅ |
| Pos-100 | 64.20% | V1 verified | ✅ |
| Full-seq | 67.45%* | scaling_results.md | ⚠️ Non-V1 |
| Full-seq (V1) | In progress | scai4 | Pending |

*Full-seq 67.45% is step 50 only before severe degradation to 54.45%.
