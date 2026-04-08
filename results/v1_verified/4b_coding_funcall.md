# 4B Teacher Coding & Funcall Results (Non-V1)

**⚠️ ALL RESULTS IN THIS FILE ARE FROM THE NEWER CODE (scaling experiments), NOT V1.**
These may be affected by the batch path KL bugs. Use with caution.

**Code**: `on_policy_distill_positional.py` (newer code, batch path)
**Student**: Qwen2.5-Math-1.5B | **Teacher**: Qwen3-4B
**Config**: LoRA r=32, pos-100, n_samples=1, chunk_size=16, 3200 problems, 200 steps
**Source**: `docs/scaling_results.md` Config A (M-1.5B → Q3-4B)

## Coding (HumanEval/MBPP, pass@1, temp=0.0)

### Pos-100

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 37.20 | 34.76 | 50.26 | 44.18 |
| **100** | **43.29** | **38.41** | 49.74 | 44.44 |
| 150 | 42.68 | 37.20 | **52.38** | **47.62** |
| 200 | 42.07 | 37.20 | 52.38 | 47.09 |

### Full-seq (step 50 only)

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 35.37 | 31.71 | 50.79 | 45.00 |

**Note**: Paper currently has 4B fullseq HE=39.63, MBPP=50.79. The 39.63 doesn't match scaling_results.md step 50 HE=35.37. Source of 39.63 unclear — possibly from a different evaluation run.

## Function Calling (BFCL, 600 problems)

### Pos-100

| Step | Name Acc | Full Acc | Parse Rate |
|------|----------|----------|------------|
| 50 | 13.00% | 7.83% | — |
| 100 | 50.00% | 29.00% | — |
| 150 | 62.83% | 43.00% | — |
| **200** | **67.83%** | **45.83%** | — |

### Full-seq
No V1 or non-V1 fullseq funcall data available for 4B teacher.
Paper currently has 4B fullseq BFCL=34.83 — source unclear.

## Summary (Used in paper main table)

| Config | HE | MBPP | BFCL | Source |
|--------|------|------|------|--------|
| Pos-100 | 43.29 | 49.74 | 45.83 | scaling_results.md ⚠️ |
| Full-seq | 39.63 | 50.79 | 34.83 | Unknown source ⚠️ |

## Missing / Needed
- 4B pos-50 coding: No experiments exist
- 4B pos-50 funcall: Training was started on scai4 but killed for AIME eval
- V1 re-runs of all 4B coding/funcall experiments
