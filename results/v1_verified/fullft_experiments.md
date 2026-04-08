# Full-Finetune Experiments (V1 Code)

**Code**: `on_policy_distill_positional_v1.py` (commit `bd81eb6`)
**Student**: Qwen2.5-Math-1.5B | **Teacher**: Qwen3-1.7B
**Config**: Full finetune (no LoRA), lr=5e-6, bs=16, mini_bs=1, n_samples=1, 3200 problems, 200 steps

## Training Status

| Experiment | Server | GPU | Status | Checkpoints |
|-----------|--------|-----|--------|-------------|
| v1-fullft-pos50-math | scai3 | GPU 0 | 🔄 Training (serial queue) | — |
| v1-fullft-pos100-math | scai3 | GPU 0 | 🔄 Queued after pos50 | — |
| v1-fullft-pos50-coding | scai3 | GPU 0 | 🔄 Queued after pos100 | — |
| v1-fullft-pos100-coding | scai3 | GPU 0 | 🔄 Queued after pos50-coding | — |
| v1-fullft-fullseq-math | UCLACG | GPU 0 | 🔄 Training | — |
| v1-fullft-fullseq-coding | UCLACG | GPU 1 | 🔄 Training | — |

## Checkpoint Paths
- scai (shared FS): `/home/antarachugh/idountang/quick-distillation/checkpoints/v1-fullft-*`
- UCLACG: `/zhi_backup/ziheng/quick-distillation/checkpoints/v1-fullft-*`

## Results (to be filled)

### Math (MATH-500, avg@4)

| Method | Step 50 | Step 100 | Step 150 | Step 200 | Best |
|--------|---------|----------|----------|----------|------|
| Baseline | — | — | — | — | 50.95 |
| FullFT pos-50 | 51.80 | 52.20 | 51.60 | **53.70** | **53.70** |
| FullFT pos-100 | 50.85 | 52.35 | 53.00 | **53.15** | **53.15** |
| FullFT fullseq | 52.70 | 54.10 | **54.35** | 53.50 | **54.35** |

### Coding (HumanEval / MBPP pass@1)

| Method | Step | HE | MBPP |
|--------|------|------|------|
| Baseline | — | 32.93 | 52.12 |
| FullFT pos-50 | 50 | 32.32 | 52.91 |
| | 100 | 34.76 | 52.65 |
| | 150 | 35.37 | 52.65 |
| | **200** | **37.20** | 52.12 |
| FullFT pos-100 | 50 | 31.71 | **53.17** |
| | 100 | 34.76 | 51.32 |
| | **150** | **35.37** | 51.85 |
| | 200 | 34.76 | 52.38 |
| FullFT fullseq | pending | — | — |

### LoRA vs FullFT Comparison (best step)

| Metric | LoRA pos-100 | FullFT pos-100 | Delta |
|--------|-------------|----------------|-------|
| MATH avg@4 | **63.15%** | 53.15% | **-10.0pp** |
| HE pass@1 | **39.63%** | 35.37% | **-4.3pp** |
| MBPP pass@1 | 51.32% | **53.17%** | +1.9pp |

LoRA significantly outperforms FullFT on math and HumanEval. MBPP slightly better with FullFT.

## Notes
- FullFT uses lr=5e-6 (10x lower than LoRA lr=5e-5)
- Fullseq experiments run in parallel on UCLACG (larger disk)
- Pos experiments run serially on scai3 GPU 0
- Git version: commit `bd81eb6` (V1 original code)
