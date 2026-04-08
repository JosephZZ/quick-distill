# Gemma Cross-Family Results (New Code Only)

**⚠️ V1 code does NOT support Gemma** (hardcoded system role in `build_prompt`).
All Gemma experiments use the newer code with `_supports_system_role` detection.

**Code**: `on_policy_distill_positional.py` (newer code, cross-tokenizer path)
**Student**: gemma-2-2b-it | **Teacher**: gemma-3-4b-it
**Tokenizer**: Cross-family, 69% vocab overlap, requires re-tokenization
**Server**: UCLACG, A6000 48GB
**Config**: LoRA r=32, alpha=64, lr=5e-5, bs=16, n_samples=1

## Baselines

### gemma-2-2b-it (Student)
| MATH-500 avg@4 | HE pass@1 | MBPP pass@1 | BFCL full_acc |
|----------------|-----------|-------------|---------------|
| 13.45% | 23.78% | 40.48% | 73.50% |

### gemma-3-4b-it (Teacher)
| MATH-500 avg@4 | HE pass@1 | MBPP pass@1 | BFCL full_acc |
|----------------|-----------|-------------|---------------|
| — | — | — | — |

## Pos-100 Results

| Task | Metric | Value | Step |
|------|--------|-------|------|
| MATH-500 | avg@4 | **27.20%** | — |
| HumanEval | pass@1 | **31.10%** | — |
| MBPP | pass@1 | **50.53%** | — |
| BFCL | full_acc | **82.50%** | — |

**Checkpoint**: on UCLACG

## Pos-50 Results
**Status**: Training was in progress on UCLACG
- Gemma pos-50 coding: UCLACG GPU 0, ~3 steps at last check
- Gemma pos-50 funcall: Queued after coding
- Gemma pos-50 math: Not started

## Full-seq Results

### Math
| Step | avg@4 |
|------|-------|
| — | 11.70% (degrades below baseline 13.45%) |

**Marked with §** in paper — fullseq degrades below baseline.

### Other tasks (coding, funcall)
Not evaluated for fullseq — degradation on math suggests fullseq is harmful for Gemma cross-family.

## Summary (Used in paper)

| Config | MATH-500 | HE | MBPP | BFCL |
|--------|----------|------|------|------|
| Baseline | 13.45 | 23.78 | 40.48 | 73.50 |
| Full-seq | 11.70§ | — | — | — |
| Pos-50 | — | — | — | — |
| **Pos-100** | **27.20** | **31.10** | **50.53** | **82.50** |

## Missing / Needed
- Gemma pos-50 all tasks (training in progress)
- Gemma fullseq coding/funcall (not planned due to math degradation)
- Gemma teacher baseline evals (for reference)
