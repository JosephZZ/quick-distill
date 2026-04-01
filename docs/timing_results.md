# Timing and Memory Results (V1 Original Code, Verified)

## Hardware Tested
- **A100-SXM4-40GB** (scai3): All training experiments
- **A6000 48GB** (UCLACG, scai4): Gemma + 4B teacher experiments
- **V100-SXM2-16GB** (scai1): Only math pos experiments (coding/funcall OOM)

## Per-Step Timing (A100 40GB, 2-GPU setup, bs=16)

| Config | Gen (s) | Score (s) | Train (s) | **Total (s/step)** | Tokens/step |
|--------|---------|-----------|-----------|-------------------|-------------|
| pos-5 math | 0 | 1 | 3.5 | **~5** | 80 |
| pos-50 math | 3 | 1 | 3.7 | **~8** | 800 |
| **pos-100 math** | **6** | **1** | **3.7** | **~11** | 1600 |
| pos-200 math | 12 | 1 | 4.0 | **~17** | (variable) |
| **fullseq math** | **122** | **5** | **6.1** | **~133** | 11700 |

**Speedup: pos-100 is 12x faster than fullseq per step on A100.**

## Per-Step Timing (A100 40GB, 1-GPU single_gpu mode, bs=16)

| Config | Gen (s) | Score (s) | Train (s) | **Total (s/step)** |
|--------|---------|-----------|-----------|-------------------|
| pos-100 coding | 6 | 1 | 4.0 | **~11** |
| fullseq coding | (running, ~100-130s gen expected) | | | **~140** |

## GPU Memory Usage (Peak, A100 40GB)

| Config | GPU Mode | Student GPU | Teacher GPU | Peak Total |
|--------|----------|-------------|-------------|------------|
| pos-100 math | 2-GPU | ~5 GB | ~4 GB | **~9 GB** |
| fullseq math | 2-GPU | ~13 GB | ~4 GB | **~17 GB** |
| fullseq coding | 1-GPU | **~34 GB** (student+teacher+gen) | same | **~34 GB** |
| pos-100 coding | 1-GPU | ~8 GB | same | **~8 GB** |

## OOM Observations

- **V100 16GB**: OOM on coding experiments (needs 7.47 GB allocation with 1.69 GB free). Math pos experiments work fine (~5 GB per GPU).
- **A100 40GB**: No OOM on any configuration. Fullseq coding uses ~34 GB on single GPU.
- **Fullseq math** on A100 2-GPU: ~13 GB student side, fits comfortably.

## Total Training Time Estimate (200 steps)

| Config | Per-step | **200 steps** |
|--------|----------|--------------|
| pos-100 | ~11s | **~37 min** |
| fullseq math | ~133s | **~7.4 hours** |
| fullseq coding | ~140s (est) | **~7.8 hours** |

**Speedup: pos-100 completes in 37 minutes vs fullseq in 7+ hours = 12x faster.**
