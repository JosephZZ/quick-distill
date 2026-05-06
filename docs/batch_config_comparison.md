# Batch Configuration Comparison for On-Policy Distillation

This report compares how different batch configurations affect distillation performance on MATH-500. All experiments use the same student (Qwen2.5-Math-1.5B) and teacher (Qwen3-1.7B) with reverse KL loss.

**Baseline (no distillation):** 50.95% avg@4, 61.2% maj@4, 72.8% pass@4

## Configurations

| Config | n_samples | Problems | Batch Size | Total Steps | Per-Step Composition |
|--------|-----------|----------|------------|-------------|----------------------|
| **n16bs16** | 16 | 200 | 16 | 200 | 1 problem x 16 trajectories |
| **n1bs16** | 1 | 3200 | 16 | 200 | 16 problems x 1 trajectory |
| **n1bs1** | 1 | 3200 | 1 | 3200 | 1 problem x 1 trajectory |

n1bs1 was unintended (code bug set batch size to 1), but the results are informative.

---

## LoRA Experiments (lr=5e-5)

### n16bs16 (Series 1)

**Pos-50:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 62.45% | 68.8% | 77.0% |
| 100 | 62.15% | 69.2% | 78.4% |
| 150 | 60.50% | 66.0% | 77.6% |
| 200 | 61.90% | 66.8% | 78.6% |

**Pos-100:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 61.15% | 68.6% | 77.6% |
| 100 | 63.55% | 68.4% | 79.2% |
| 150 | 64.25% | 69.6% | 80.2% |
| 200 | 63.85% | 69.8% | 80.2% |

**Pos-150:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 63.55% | 69.6% | 79.6% |
| 100 | 65.70% | 71.4% | 77.4% |
| 150 | 64.80% | 70.2% | 80.0% |
| 200 | 64.40% | 68.2% | 79.4% |

**Pos-200tok:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 63.25% | 70.0% | 79.6% |
| 100 | 66.75% | 72.0% | 80.2% |
| 150 | 64.55% | 70.0% | 79.4% |
| 200 | 65.25% | 72.0% | 79.2% |

**Full-seq:**

| Step | avg@4 (last-boxed) | avg@4 (first-boxed) | maj@4 | pass@4 | \boxed avg count |
|------|-------------------|-------------------|-------|--------|------------------|
| 50 | 65.60% | 65.55% | 62.4% | 80.0% | 1.0 |
| 100 | 46.30% | 64.95% | 39.4% | 72.2% | 88.1 |
| 150 | 49.70% | 65.55% | 41.0% | 77.0% | 54.1 |
| 200 | 43.80% | 64.95% | 33.4% | 74.2% | 58.0 |

Note: Full-seq models after step 50 repeat `\boxed{}` 58-88 times, corrupting last-boxed extraction. First-boxed shows stable ~65% accuracy. See `fullseq_degradation_analysis.md` for details.

**Pos-5:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 55.15% | 62.2% | 74.0% |
| 100 | 55.65% | 63.0% | 74.8% |
| 150 | 56.50% | 64.8% | 76.0% |
| 200 | 56.30% | 61.8% | 73.4% |

**Pos-10:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 59.50% | 66.4% | 75.8% |
| 100 | 56.40% | 63.6% | 73.2% |
| 150 | 56.30% | 64.2% | 75.4% |
| 200 | 56.00% | 63.8% | 74.2% |

**Pos-20:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 60.20% | 67.6% | 77.0% |
| 100 | 57.35% | 65.0% | 76.0% |
| 150 | 56.40% | 65.6% | 73.4% |
| 200 | 56.55% | 61.8% | 75.8% |

**Progressive 1→200:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 61.10% | 67.8% | 77.2% |
| 100 | 62.05% | 68.0% | 77.8% |
| 150 | 61.95% | 69.0% | 78.2% |
| 200 | 62.30% | 69.0% | 78.8% |

### n1bs16 (Series 2)

**Pos-50:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 62.35% | 69.40% | 77.20% |
| 100 | 66.05% | 72.00% | 79.40% |
| 150 | 66.65% | 71.00% | 81.00% |
| 200 | 64.85% | 71.20% | 79.60% |

**Pos-100:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 63.75% | 70.00% | 79.80% |
| 100 | 64.45% | 68.40% | 78.40% |
| 150 | 65.15% | 69.60% | 80.20% |
| 200 | 65.85% | 70.80% | 79.80% |

**Pos-200tok:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 66.05% | 71.20% | 81.00% |
| 100 | 64.65% | 68.40% | 79.80% |
| 150 | 65.10% | 70.00% | 80.60% |
| 200 | 65.55% | 71.20% | 80.60% |

### n1bs1 (Series — only pass@4 available)

**Pos-50:**

| Step | pass@4 |
|------|--------|
| 50 | 73.0% |
| 100 | 75.4% |
| 150 | 77.0% |
| 200 | 77.0% |
| 1600 | 80.0% |
| 3200 | 80.4% |

**Pos-100:**

| Step | pass@4 |
|------|--------|
| 1600 | 79.2% |
| 3200 | 78.4% |

#### BS=1 vs BS=16 Detailed Comparison

**Pos-50:**

| BS | Step 200 pass@4 | Best pass@4 |
|----|-----------------|-------------|
| 16 | 78.6% (200 problems) | 78.6% (step 200) |
| 1 | 77.0% (200 problems) | 80.4% (step 3200) |

At the same number of problems seen (200), BS=1 is 1.6% lower. But with 16x more data (3200 problems), BS=1 reaches 80.4% — surpassing BS=16's best.

**Pos-100:**

| BS | Best pass@4 |
|----|-------------|
| 16 | 80.2% (step 150) |
| 1 | 79.2% (step 1600) |

Pos-100 BS=1 slightly underperforms BS=16 despite 10x more data.

**BS=1 stability explanation:**
- LoRA's low-rank constraint acts as implicit regularization
- Cosine lr schedule (5e-5 → 0 over 3200 steps) decays quickly enough
- KL loss on short sequences has naturally low variance
- Note: Only pass@4 was recorded (no avg@4/maj@4), which may mask quality differences

---

## FullFT Experiments (lr=5e-6)

### n16bs16 (Series 3)

**Pos-50:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 52.30% | 60.60% | 72.40% |
| 100 | 53.70% | 63.00% | 74.00% |
| 150 | 54.30% | 62.80% | 73.00% |
| 200 | 53.75% | 61.20% | 73.40% |

**Pos-100:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 52.90% | 62.60% | 74.00% |
| 100 | 53.25% | 62.20% | 74.40% |
| 150 | 53.00% | 62.40% | 73.20% |
| 200 | 54.20% | 61.40% | 75.00% |

**Pos-200tok:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 53.70% | 63.40% | 74.80% |
| 100 | 54.95% | 61.20% | 75.60% |
| 150 | 55.25% | 65.00% | 75.60% |
| 200 | 54.20% | 63.00% | 73.80% |

### n1bs16 (Series 4)

**Pos-50:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 55.55% | 65.20% | 75.60% |
| 100 | 55.85% | 65.00% | 74.40% |
| 150 | 56.75% | 66.20% | 74.80% |
| 200 | 55.50% | 64.20% | 75.00% |

**Pos-100:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 55.15% | 64.00% | 74.00% |
| 100 | 56.20% | 64.20% | 73.80% |
| 150 | 55.65% | 64.60% | 74.20% |
| 200 | 54.80% | 63.00% | 74.20% |

**Pos-200tok:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 55.45% | 64.40% | 74.60% |
| 100 | 55.80% | 64.00% | 74.80% |
| 150 | 55.90% | 64.00% | 75.60% |
| 200 | 56.40% | 65.20% | 75.20% |

**Full-seq:**

| Step | avg@4 | maj@4 | pass@4 |
|------|-------|-------|--------|
| 50 | 53.75% | 61.00% | 73.40% |
| 100 | 56.95% | 66.80% | 77.40% |
| 150 | 57.55% | 65.80% | 75.60% |
| 200 | 58.20% | 65.40% | 75.40% |

---

## Best Results Comparison

| Method | n16bs16 best avg@4 | n1bs16 best avg@4 | n1bs1 best pass@4 |
|--------|--------------------|--------------------|---------------------|
| LoRA Pos-50 | 62.45% | 66.65% | 80.4% |
| LoRA Pos-100 | 64.25% | 65.85% | 78.4%* |
| LoRA Pos-200tok | 66.75% | 66.05% | — |
| FullFT Pos-50 | 53.70% | 56.75% | — |
| FullFT Pos-200tok | 55.25% | 56.40% | — |

*n1bs1 only has pass@4, not avg@4

---

## Key Findings

1. **n1bs16 ~ n16bs16 for LoRA:** Both achieve ~66% best avg@4. n1 sees 16x more unique problems but fewer trajectories per problem. Equal performance suggests problem diversity and trajectory diversity contribute equally.

2. **n1bs16 > n16bs16 for FullFT:** n1 (~56% avg@4) consistently outperforms n16 (~54%). Full fine-tuning benefits more from data diversity.

3. **n1bs1 is surprisingly stable:** Despite single-trajectory gradients (extremely noisy), training does not diverge. LoRA's low-rank constraint provides implicit regularization.

4. **n1bs1 eventually surpasses n16bs16 with more data:** Pos-50 bs=1 reaches 80.4% pass@4 at step 3200, beating bs=16's 78.6% at step 200. But requires 16x more compute.

5. **Longer position limits may need larger batches:** Pos-100 bs=1 (79.2%) does not surpass bs=16 (80.2%), unlike pos-50.

---

## File Paths

### Log Paths
- n16bs16: `logs/math_qwen2.5-1.5B_qwen3-1.7B_pos_lora_n16bs16/`
- n1bs16: `logs/math_qwen2.5-1.5B_qwen3-1.7B_pos_lora_n1bs16/`
- n1bs1: `logs/math_qwen2.5-1.5B_qwen3-1.7B_pos_lora_n1bs1/`
- FullFT: `logs/math_qwen2.5-1.5B_qwen3-1.7B_pos_fullft/`

### Checkpoint Paths
- n16bs16: `checkpoints/pos-limit-{5,10,20}tok/`, `checkpoints/positional-distill-50tok-v2/`, `checkpoints/pos-limit-{100,150,200tok}/`, `checkpoints/progressive-pos-1to200/`, `checkpoints/full-seq-3584tok/`
- n1bs16: `checkpoints/pos-limit-{50,100,200tok}-n1-bs16/`
- n1bs1: `checkpoints/pos-limit-{50,100}-n1-bs1/`
- FullFT n16: `checkpoints/fullft-{pos50,pos100,pos200tok}-n16/`
- FullFT n1: `checkpoints/fullft-{pos50,pos100,pos200tok,fullseq}-n1/`
