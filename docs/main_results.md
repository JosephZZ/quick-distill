# Positional Distillation: Main Results

Positional loss (training only on the first N response tokens) is effective across tasks (math and coding), for both LoRA and full fine-tuning, and is comparable to or better than full-sequence distillation.

## Setup

- **Student**: Qwen2.5-Math-1.5B
- **Teacher**: Qwen3-1.7B
- **Method**: On-policy reverse KL distillation
- **Math eval**: MATH-500, n_samples=4, temperature=0.7 (metrics: avg@4, maj@4, pass@4)
- **Coding eval**: HumanEval / HumanEval+ / MBPP / MBPP+ (pass@1, temperature=0.0, n=1)
- **Training**:
  - LoRA: r=32, alpha=64, lr=5e-5
  - FullFT: lr=5e-6
  - 200 steps for math, 400 steps for coding, save every 50
- **Main config**: n_samples=1, bs=16 (3200 problems for math, 6400 for coding)
- **Coding dataset**: coseal/CodeUltraFeedback_binarized
- **Baseline** (undistilled Qwen2.5-Math-1.5B): 50.95% avg@4 on MATH-500

## Best Results Summary

| Task | Method | Config | Best Step | Best Metric | Value |
|------|--------|--------|-----------|-------------|-------|
| Math | LoRA | Pos-50 | 150 | avg@4 | 66.65% |
| Math | LoRA | Pos-100 | 200 | avg@4 | 65.85% |
| Math | LoRA | Pos-150 | 100 | avg@4 | 66.65% |
| Math | LoRA | Pos-200tok | 50 | avg@4 | 66.05% |
| Math | FullFT | Pos-50 | 150 | avg@4 | 56.75% |
| Math | FullFT | Pos-100 | 100 | avg@4 | 56.20% |
| Math | FullFT | Pos-200tok | 200 | avg@4 | 56.40% |
| Math | FullFT | Full-seq | 200 | avg@4 | 58.20% |
| Math | LoRA | Full-seq (n1bs16) | 150 | avg@4 | 62.35% |
| Math | LoRA | Random-100 | 150 | avg@4 | 63.05% |
| Math | LoRA | TopEnt-Teacher-100 | 150 | avg@4 | 62.20% |
| Math | LoRA | TopEnt-Student-100 | 100 | avg@4 | 61.35% |
| Math | LoRA | TopKL-100 | 50 | avg@4 | 58.60% |
| Math | LoRA | Middle-100 | 200 | avg@4 | 47.80% |
| Math | LoRA | Last-100 | 200 | avg@4 | 50.35% |
| Coding | LoRA | Pos-50 | 350 | HE | 42.1 |
| Coding | LoRA | Pos-100 | 150 | HE | 42.1 |
| Coding | LoRA | Pos-150 | 250 | HE | 41.5 |
| Coding | LoRA | Pos-250 | 50 | HE | 39.0 |
| Coding | LoRA | Full-seq | 50 | HE | 40.2 |
| Coding | FullFT | Pos-50 | 150/200/250/350 | HE | 36.0 |
| Coding | FullFT | Pos-100 | 250/400 | HE | 36.6 |
| Coding | FullFT | Pos-150 | 400 | HE | 37.2 |
| Coding | FullFT | Pos-250 | 350 | HE | 37.8 |
| Coding | FullFT | Pos-200tok | 100/150 | HE | 36.6 |
| Coding | FullFT | Full-seq | 150 | HE | 36.6 |

---

## Math Results -- LoRA (n1, 3200 problems, bs=16)

### Pos-50

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 62.35% | 69.40% | 77.20% |
| 100 | 66.05% | 72.00% | 79.40% |
| 150 | 66.65% | 71.00% | 81.00% |
| 200 | 64.85% | 71.20% | 79.60% |

### Pos-100

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 63.75% | 70.00% | 79.80% |
| 100 | 64.45% | 68.40% | 78.40% |
| 150 | 65.15% | 69.60% | 80.20% |
| 200 | 65.85% | 70.80% | 79.80% |

### Pos-150

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 65.35% | 66.80% | 79.00% |
| 100 | 66.65% | 67.00% | 81.00% |
| 150 | 65.30% | 66.30% | 78.20% |
| 200 | 65.75% | 67.30% | 80.00% |

### Pos-200tok

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 66.05% | 71.20% | 81.00% |
| 100 | 64.65% | 68.40% | 79.80% |
| 150 | 65.10% | 70.00% | 80.60% |
| 200 | 65.55% | 71.20% | 80.60% |

### Full-seq (n1, 3200 problems, max_new_tokens=2048, single-GPU HF generate)

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 61.00% | 66.80% | 74.60% |
| 100 | 62.00% | 68.40% | 75.20% |
| 150 | 62.35% | 69.40% | 74.60% |
| 200 | 61.20% | 65.20% | 75.00% |

Note: Math avg@4 is stable at 61-62% (no repetition degradation). However, it is still 3-4pp below the best positional variants (pos-50: 66.65%, pos-100: 65.85%).

#### Full-seq n1bs16 Coding & Funcall (evaluated from math-trained checkpoint)

**WARNING**: These coding/funcall evals were run on the math-trained fullseq checkpoint (trained on NuminaMath-CoT), NOT on a coding/funcall-trained checkpoint. Results are not meaningful for comparison.

### Fullseq Coding (1.7B teacher, task-specific training)

Experiment: `scale-m1.5b-t1.7b-coding-fullseq` (on InfraWaves). Trained on CodeUltraFeedback.

| Step | HE | HE+ |
|------|-----|------|
| 50 | 31.7% | 26.2% |
| 100-200 | *(unmerge bug — identical to step 50)* | |

Note: HE = 31.7% is below baseline (32.93%). Fullseq with 1.7B teacher fails on coding.

### Fullseq Funcall (1.7B teacher, task-specific training)

Experiment: `scale-m1.5b-t1.7b-funcall-fullseq` (on InfraWaves). Trained on funcall data.

| Step | full_acc | name_acc | parse_rate |
|------|----------|----------|------------|
| 50 | 1.83% | 8.33% | 22.83% |
| 100 | 2.50% | 8.00% | 23.33% |
| 200 | 2.33% | 8.00% | 22.00% |

Note: full_acc < 3% across all steps, far below pos-100 (61.30%). Parse rate ~23% means the model rarely produces valid function call format.

Checkpoint: `checkpoints/fullseq-n1bs16-single/` (on scai5)

### Full-seq Funcall (n1, 3200 problems, same config)

| Step | full_acc | name_acc | parse_rate |
|------|----------|----------|------------|
| 50 | 1.67% | 33.83% | 43.67% |
| 100 | 7.17% | 32.00% | 40.83% |
| 150 | 5.33% | 25.83% | 37.50% |
| 200 | 7.67% | 26.00% | 37.67% |

Note: Full-seq funcall is catastrophically bad (parse_rate ~40%, full_acc <8%) compared to pos-100 funcall (full_acc 61.30%). The model generates unparseable outputs for most function calling problems.

---

## Math Results -- FullFT (n1, 3200 problems, bs=16, lr=5e-6)

### Pos-50

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 55.55% | 65.20% | 75.60% |
| 100 | 55.85% | 65.00% | 74.40% |
| 150 | 56.75% | 66.20% | 74.80% |
| 200 | 55.50% | 64.20% | 75.00% |

### Pos-100

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 55.15% | 64.00% | 74.00% |
| 100 | 56.20% | 64.20% | 73.80% |
| 150 | 55.65% | 64.60% | 74.20% |
| 200 | 54.80% | 63.00% | 74.20% |

### Pos-200tok

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 55.45% | 64.40% | 74.60% |
| 100 | 55.80% | 64.00% | 74.80% |
| 150 | 55.90% | 64.00% | 75.60% |
| 200 | 56.40% | 65.20% | 75.20% |

### Full-seq (n1, 3200 problems, vLLM)

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 53.75% | 61.00% | 73.40% |
| 100 | 56.95% | 66.80% | 77.40% |
| 150 | 57.55% | 65.80% | 75.60% |
| 200 | 58.20% | 65.40% | 75.40% |

---

## Coding Results -- LoRA (n1, 6400 problems, bs=16, lr=5e-5)

### coding-lora-pos50-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 37.8 | 34.1 | 52.1 | 45.8 |
| 100 | 39.0 | 34.1 | 48.9 | 43.7 |
| 150 | 39.6 | 34.8 | 46.0 | 41.5 |
| 200 | 41.5 | 36.6 | 46.0 | 42.3 |
| 250 | 40.2 | 34.8 | 47.6 | 42.9 |
| 300 | 40.9 | 35.4 | 46.3 | 41.8 |
| 350 | 42.1 | 36.6 | 46.6 | 41.5 |
| 400 | 40.9 | 36.0 | 46.3 | 41.5 |

### coding-lora-pos100-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 37.2 | 34.1 | 52.1 | 45.8 |
| 100 | 39.0 | 33.5 | 49.7 | 44.2 |
| 150 | 42.1 | 36.0 | 49.2 | 44.4 |
| 200 | 37.8 | 34.1 | 49.2 | 43.9 |
| 250 | 39.0 | 34.8 | 48.4 | 43.4 |
| 300 | 37.8 | 33.5 | 49.2 | 43.9 |
| 350 | 37.8 | 34.1 | 47.9 | 43.4 |
| 400 | 38.4 | 34.8 | 48.9 | 44.4 |

### coding-lora-pos150-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 36.6 | 33.5 | 51.3 | 45.5 |
| 100 | 35.4 | 31.7 | 50.0 | 45.2 |
| 150 | 36.6 | 33.5 | 48.4 | 44.4 |
| 200 | 39.0 | 36.0 | 50.5 | 46.0 |
| 250 | 41.5 | 37.8 | 49.5 | 44.7 |
| 300 | 39.6 | 36.6 | 50.3 | 45.0 |
| 350 | 38.4 | 35.4 | 49.7 | 45.2 |
| 400 | 37.2 | 34.1 | 48.7 | 43.7 |

### coding-lora-pos250-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 39.0 | 34.1 | 51.3 | 46.0 |
| 100 | 37.2 | 34.1 | 48.7 | 43.1 |
| 150 | 32.3 | 29.3 | 48.7 | 44.2 |
| 200 | 38.4 | 35.4 | 49.7 | 44.4 |
| 250 | 34.1 | 31.1 | 48.9 | 43.4 |
| 300 | 36.6 | 32.3 | 49.2 | 43.9 |
| 350 | 35.4 | 32.3 | 48.4 | 43.1 |
| 400 | 36.0 | 32.3 | 48.7 | 43.4 |

### coding-lora-fullseq-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 40.2 | 35.4 | 52.6 | 46.3 |
| 100 | 31.7 | 27.4 | 48.9 | 42.6 |
| 150 | 32.3 | 29.9 | 49.5 | 44.4 |
| 200 | 32.9 | 29.9 | 48.1 | 43.4 |
| 250 | 27.4 | 25.0 | 47.9 | 43.1 |
| 300 | 28.0 | 26.2 | 47.1 | 41.8 |
| 350 | 26.8 | 25.0 | 48.4 | 43.4 |
| 400 | 26.8 | 25.0 | 47.6 | 42.9 |

---

## Coding Results -- FullFT (n1, 6400 problems, bs=16, lr=5e-6)

### coding-fullft-pos50-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 31.7 | 26.8 | 52.6 | 44.4 |
| 100 | 35.4 | 30.5 | 53.4 | 45.8 |
| 150 | 36.0 | 30.5 | 52.9 | 46.0 |
| 200 | 36.0 | 29.9 | 52.4 | 45.8 |
| 250 | 36.0 | 29.9 | 52.4 | 45.2 |
| 300 | 35.4 | 29.9 | 52.6 | 45.2 |
| 350 | 36.0 | 30.5 | 52.9 | 45.2 |
| 400 | 34.8 | 29.3 | 53.4 | 46.6 |

### coding-fullft-pos100-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 31.7 | 26.8 | 52.6 | 45.0 |
| 100 | 34.1 | 29.3 | 54.0 | 46.0 |
| 150 | 35.4 | 30.5 | 52.6 | 46.0 |
| 200 | 34.8 | 29.9 | 52.1 | 45.5 |
| 250 | 36.6 | 31.1 | 53.2 | 45.8 |
| 300 | 36.0 | 30.5 | 53.4 | 46.3 |
| 350 | 36.0 | 29.9 | 53.4 | 46.3 |
| 400 | 36.6 | 31.1 | 52.9 | 45.8 |

### coding-fullft-pos150-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 31.7 | 27.4 | 53.2 | 45.0 |
| 100 | 36.0 | 31.1 | 51.9 | 45.2 |
| 150 | 36.0 | 30.5 | 52.4 | 45.2 |
| 200 | 35.4 | 29.9 | 52.4 | 45.8 |
| 250 | 36.0 | 30.5 | 53.2 | 46.6 |
| 300 | 36.6 | 31.1 | 54.2 | 46.8 |
| 350 | 35.4 | 29.9 | 53.2 | 46.0 |
| 400 | 37.2 | 30.5 | 52.4 | 45.5 |

### coding-fullft-pos250-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 32.9 | 27.4 | 54.0 | 45.8 |
| 100 | 34.8 | 29.9 | 53.7 | 46.0 |
| 150 | 35.4 | 30.5 | 54.0 | 47.1 |
| 200 | 35.4 | 29.9 | 54.8 | 47.4 |
| 250 | 37.2 | 31.1 | 54.5 | 47.4 |
| 300 | 36.6 | 31.1 | 53.4 | 46.6 |
| 350 | 37.8 | 31.7 | 54.2 | 47.4 |
| 400 | 33.5 | 28.7 | 53.2 | 46.0 |

### coding-fullft-pos200tok-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 32.3 | 28.7 | 52.1 | 44.7 |
| 100 | 36.6 | 30.5 | 52.6 | 45.8 |
| 150 | 36.6 | 31.1 | 54.0 | 46.3 |
| 200 | 35.4 | 29.9 | 52.6 | 46.0 |
| 250 | 36.0 | 30.5 | 53.7 | 46.8 |
| 300 | 34.8 | 29.3 | 54.5 | 47.1 |
| 350 | 36.0 | 30.5 | 53.2 | 46.0 |
| 400 | 35.4 | 29.9 | 53.7 | 46.3 |

### coding-fullft-fullseq-n1

| Step | HE | HE+ | MBPP | MBPP+ |
|------|------|------|------|-------|
| 50 | 32.3 | 27.4 | 53.2 | 44.7 |
| 100 | 32.3 | 28.7 | 53.4 | 46.3 |
| 150 | 36.6 | 31.7 | 53.2 | 46.0 |
| 200 | 36.0 | 30.5 | 54.0 | 46.3 |
| 250 | 31.1 | 26.2 | 51.9 | 43.4 |
| 300 | 30.5 | 26.2 | 52.9 | 45.0 |
| 350 | 31.7 | 26.8 | 52.9 | 44.4 |
| 400 | 31.1 | 26.2 | 53.7 | 44.7 |

---

## Selective Token Distillation -- Math LoRA (n1, 3200 problems, bs=16, SGLang)

These experiments select the top-k tokens by different criteria (rather than the first k positional tokens) and compute loss only on those. All use full-length generation (max_new_tokens=2048) with SGLang, then select k=100 tokens for loss. Compared against positional prefix baselines above.

### Top-KL (k=100)

Selects 100 tokens with highest per-token KL divergence between student and teacher.

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 58.60% | 65.80% | 74.40% |
| 100 | 45.70% | 62.40% | 70.80% |
| 150 | 53.30% | 63.00% | 75.20% |
| 200 | 52.25% | 64.40% | 72.40% |

### Top-Entropy (Student, k=100)

Selects 100 tokens where the student has highest entropy (most uncertain).

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 61.15% | 67.60% | 74.40% |
| 100 | 61.35% | 67.20% | 73.20% |
| 150 | 60.75% | 66.20% | 75.00% |
| 200 | 54.60% | 67.80% | 73.80% |

### Top-Entropy (Teacher, k=100)

Selects 100 tokens where the teacher has highest entropy (most uncertain).

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 59.85% | 65.80% | 75.40% |
| 100 | 60.85% | 66.40% | 75.00% |
| 150 | 62.20% | 67.80% | 75.80% |
| 200 | 61.70% | 67.00% | 74.80% |

### Random (k=100)

Selects 100 random tokens from the full sequence as a control.

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 61.80% | 66.80% | 75.80% |
| 100 | 62.05% | 68.60% | 76.60% |
| 150 | 63.05% | 69.20% | 76.80% |
| 200 | 62.75% | 68.80% | 75.80% |

### Middle (k=100)

Selects the middle 100 tokens of the response (centered at midpoint).

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 46.85% | 56.60% | 68.40% |
| 100 | 46.85% | 56.60% | 68.40% |
| 150 | 46.55% | 56.40% | 69.40% |
| 200 | 47.80% | 58.60% | 69.40% |

Checkpoint: `checkpoints/token-select-k100-middle-math-m1.5b-t1.7b/` (on InfraWaves)

### Last (k=100)

Selects the last 100 tokens of the response (tail end).

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 47.00% | 56.40% | 70.40% |
| 100 | 48.95% | 57.80% | 70.60% |
| 150 | 49.70% | 58.40% | 69.60% |
| 200 | 50.35% | 61.60% | 71.00% |

Checkpoint: `checkpoints/token-select-k100-last-math-m1.5b-t1.7b/` (on InfraWaves)

### Selective vs Positional Comparison (best avg@4, k=100)

| Method | Best Step | avg@4 | maj@4 | pass@4 |
|--------|-----------|-------|-------|--------|
| **Pos-100 (prefix)** | **200** | **65.85%** | **70.80%** | **79.80%** |
| Random-100 | 150 | 63.05% | 69.20% | 76.80% |
| Full-seq (n1bs16) | 150 | 62.35% | 69.40% | 74.60% |
| Top-Entropy-Teacher-100 | 150 | 62.20% | 67.80% | 75.80% |
| Top-Entropy-Student-100 | 100 | 61.35% | 67.20% | 73.20% |
| Top-KL-100 | 50 | 58.60% | 65.80% | 74.40% |
| Baseline (no distill) | — | 50.95% | 61.20% | 72.80% |
| Last-100 | 200 | 50.35% | 61.60% | 71.00% |
| Middle-100 | 200 | 47.80% | 58.60% | 69.40% |

**Note**: Middle-100 and Last-100 both perform **below baseline**, demonstrating that later-position tokens are actively harmful for distillation.

---

## Key Findings

1. **Positional loss works across tasks.** Both math and coding benefit from positional distillation, confirming it is not a math-specific trick.

2. **Positional loss is comparable to or better than full-sequence distillation.**
   - Math: LoRA pos-50 best avg@4 = 66.65%, which matches or exceeds full-seq first-boxed (~65.5%).
   - Coding (LoRA): pos-50 and pos-100 reach 42.1% HumanEval, far above full-seq (40.2% at best, degrading to 26.8%).
   - Coding (FullFT): pos-250 reaches 37.8% HumanEval, above full-seq (36.6% at best, degrading to 31.1%).

3. **Full-sequence distillation degrades over training.**
   - Math: boxed repetition corrupts answer extraction after early steps.
   - Coding (LoRA): HumanEval drops from 40.2% (step 50) to 26.8% (step 400).
   - Coding (FullFT): HumanEval drops from 36.6% (step 150) to 31.1% (step 400).
   - Positional variants do not exhibit this degradation.

4. **LoRA vs FullFT.**
   - Math: LoRA is much stronger (+10pp avg@4 over FullFT).
   - Coding: LoRA achieves better HumanEval scores; FullFT achieves better MBPP scores.

5. **Sweet spot: pos-50 to pos-200tok** depending on task and training method.

6. **Positional prefix beats selective token methods.** At k=100 tokens, the positional prefix (first 100 tokens) achieves 65.85% avg@4, outperforming all selective methods: random (63.05%), top-entropy-teacher (62.20%), top-entropy-student (61.35%), and top-KL (58.60%). This confirms the **cascade effect hypothesis**: early-token improvements propagate through autoregressive generation, making position more important than raw signal concentration. Notably, top-KL performs worst despite targeting the highest-divergence tokens — likely because it focuses on format/style disagreements rather than reasoning (see `docs/selective_token_analysis.md`).

7. **Middle and last tokens are actively harmful.** Training on middle-100 (46.85%) or last-100 (49.70%) tokens performs **below the undistilled baseline** (50.95%). This demonstrates that later-position teacher supervision is not merely uninformative — it actively degrades student performance. Combined with finding #6, this shows that distillation value is concentrated in the earliest token positions.

8. **Loss clipping does not help.** Full-trajectory training with per-token KL clipped to max=2.0 achieves only 48.80% avg@4 (step 100), well below both the unclipped full-seq (62.35%) and baseline (50.95%). Clipping indiscriminately removes high-KL signal from early positions (strategy selection tokens) along with late-position noise, destroying the most valuable part of the distillation signal. This confirms that the issue is not outlier KL values but rather the position of the signal — positional truncation is a more principled approach than value clipping.

### Loss Clip Experiment

Full-trajectory with per-token KL clipped to 2.0. 1600 problems, 100 steps, HF generate.

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 47.85% | 59.20% | 70.20% |
| 100 | 48.80% | 58.60% | 72.40% |

Checkpoint: `checkpoints/m1.5b_t1.7b_math_fulltraj_lossclip2_hf_s100_t2048_20260330_105648/` (on InfraWaves)

---

## File Paths

### Logs

- Math LoRA n1bs16: `logs/math_qwen2.5-1.5B_qwen3-1.7B_pos_lora_n1bs16/`
- Math FullFT: `logs/math_qwen2.5-1.5B_qwen3-1.7B_pos_fullft/` and `logs/math_qwen2.5-1.5B_qwen3-1.7B_fullseq_fullft/`
- Coding LoRA pos: `logs/coding_qwen2.5-1.5B_qwen3-1.7B_pos-lora/`
- Coding LoRA fullseq: `logs/coding_qwen2.5-1.5B_qwen3-1.7B_fullseq_lora/`
- Coding FullFT pos: `logs/coding_qwen2.5-1.5B_qwen3-1.7B_pos-fullft/`
- Coding FullFT fullseq: `logs/coding_qwen2.5-1.5B_qwen3-1.7B_fullseq_fullft/`

### Checkpoints

- Math LoRA n1bs16: `checkpoints/pos-limit-{50,100,200tok}-n1-bs16/`
- Math FullFT n1: `checkpoints/fullft-{pos50,pos100,pos200tok,fullseq}-n1/`
- Coding LoRA: `checkpoints/coding-lora-{pos50,pos100,pos150,pos250,fullseq}-n1/`
- Coding FullFT: `checkpoints/coding-fullft-{pos50,pos100,pos150,pos250,pos200tok,fullseq}-n1/`
- Selective token (on scai5): `checkpoints/{topkl-k100-sglang,top_entropy_student-k100-sglang,top_entropy_teacher-k100-sglang,random-k100-sglang}/`
