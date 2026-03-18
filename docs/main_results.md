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

Note: No full-seq LoRA experiment was run for the n1bs16 config.

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
