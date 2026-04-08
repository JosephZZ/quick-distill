# Token Selection Experiments (V1 Code Rerun)

**Code**: `on_policy_distill_positional.py` (new code with fixed batch KL)
**Student**: Qwen2.5-Math-1.5B | **Teacher**: Qwen3-1.7B
**Config**: LoRA r=32, alpha=64, lr=5e-5, bs=16, n_samples=1, 3200 problems, 200 steps
**Mode**: fullseq generation + select k=100 tokens for loss
**Server**: scai5, GPU 2, L40S 46GB
**Checkpoint base**: `/home/antarachugh/idountang/quick-distillation/checkpoints/v1-toksel-*`

## Results (MATH-500 avg@4)

### Random-100 ✅ Complete

| Step | avg@4 |
|------|-------|
| 50 | 59.35 |
| **100** | **61.25** |
| 150 | 40.70 ⚠️ |
| 200 | 40.70 ⚠️ |

**Degrades at step 150!** Random tokens include harmful late positions.

### Top-Entropy-Teacher-100 ✅ Complete

| Step | avg@4 |
|------|-------|
| 50 | 59.90 |
| 100 | 62.45 |
| **150** | **63.30** |
| 200 | 61.20 |

Peaks at 63.30% (step 150) — slightly above pos-100! But degrades mildly at step 200.

### Top-Entropy-Student-100 ✅ Complete

| Step | avg@4 |
|------|-------|
| 50 | 60.70 |
| **100** | **62.70** |
| 150 | 62.35 |
| 200 | 62.55 |

Stable, best = 62.70%. No significant degradation.

### Top-KL-100 ✅ Complete

| Step | avg@4 |
|------|-------|
| 50 | 43.25 ⚠️ |
| 100 | 50.05 |
| **150** | 52.60 |
| **200** | **53.35** |

**Worst method!** Below baseline at step 50. High-KL tokens are misleading, not informative.

### Middle-100 ✅ Complete

**Checkpoint**: `checkpoints/v1-toksel-middle-k100-m1.5b-t1.7b-math/`
**Server**: scai5 GPU 2

Selects k tokens centered at the middle of the response.

| Step | avg@4 |
|------|-------|
| 50 | 58.90 |
| 100 | 61.80 |
| 150 | 61.75 |
| **200** | **62.30** |

Stable, best = 61.80%. No degradation.

### Last-100 ✅ Complete

**Checkpoint**: `checkpoints/v1-toksel-last-k100-m1.5b-t1.7b-math/`
**Server**: scai5 GPU 1

Selects the last k tokens of the response.

| Step | avg@4 |
|------|-------|
| 50 | 58.85 |
| **100** | **62.30** |
| 150 | 39.75 ⚠️ |
| 200 | 40.70 ⚠️ |

**Degrades catastrophically at step 150!** Same pattern as random-100 and full-seq.
Last-position tokens include the most harmful signal — late tokens cause training collapse.

## Complete Comparison (best step, MATH-500 avg@4)

| Method | Best | Step | Degrades? |
|--------|------|------|-----------|
| Baseline | 50.95 | — | — |
| **Pos-100 (ours)** | **63.15** | 150 | **No** ✅ |
| Ent-teacher-100 | **63.30** | 150 | Mild (→61.20) |
| Ent-student-100 | 62.70 | 100 | No |
| Last-100 | 62.30 | 100 | **Yes (→39.75)** ⚠️ |
| Fullseq | 62.35 | 100 | Yes (→37.75) ⚠️ |
| Middle-100 | 62.30 | 200 | No |
| Random-100 | 61.25 | 100 | Yes (→40.70) ⚠️ |
| Top-KL-100 | 53.35 | 200 | Starts below BL ⚠️ |

## Key Findings

1. **Positional prefix (pos-100) is the best overall** — highest stable performance, no degradation
2. **Entropy-teacher selection peaks slightly higher** (63.30 vs 63.15) but degrades mildly
3. **Random degrades catastrophically** — late-position noise accumulates
4. **Top-KL is worst** — high-KL tokens are misleading format/style disagreements, not useful signal
5. **Position > signal concentration** — selecting by position (contiguous prefix) works better than selecting by signal strength (entropy/KL), confirming the cascade effect hypothesis
