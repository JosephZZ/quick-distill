# Ablation: Top-K Loss & Soft Weighting Experiments

**Code**: `on_policy_distill_positional_v1.py` (patched with top-K and soft weighting support)
**Student**: Qwen2.5-Math-1.5B | **Teacher**: Qwen3-1.7B
**Config**: LoRA r=32, alpha=64, lr=5e-5, bs=16, n_samples=1, 3200 problems, 200 steps
**Server**: scai3, GPU 0+1, A100-SXM4-40GB × 2

## 1. Top-K Truncated Reverse KL (Revisiting OPD Baseline)

Top-K selects teacher's top-32 tokens at each position, renormalizes both distributions over this support set, then computes reverse KL. This is the method from "Revisiting On-Policy Distillation" (Fu et al., 2025).

### Top-K pos-100 (K=32)

**Checkpoint**: `checkpoints/v1-topk32-pos100-m1.5b-t1.7b-math/`

| Step | avg@4 |
|------|-------|
| 50 | 57.15 |
| 100 | 59.65 |
| 150 | 60.95 |
| **200** | **61.70** |

### Top-K Renorm pos-100 (K=32, correct implementation)

**Checkpoint**: `checkpoints/v1-topk32r-pos100-m1.5b-t1.7b-math/`

| Step | avg@4 |
|------|-------|
| 50 | 56.90 |
| **100** | **61.90** |
| 150 | 61.70 |
| 200 | 61.75 |

### Comparison: Full-Vocab vs Top-K

| Method | pos-100 Best | Δ vs Baseline |
|--------|-------------|---------------|
| Baseline | 50.95 | — |
| **Full-vocab KL (ours)** | **63.15** | **+12.20** |
| Top-K renorm (K=32) | 61.90 | +10.95 |

**Full-vocab KL outperforms Top-K renorm by +1.25pp.** Even with correct renormalization (gathering raw logits, single log_softmax), full-vocab reverse KL is superior. The long tail of the vocabulary distribution carries useful gradient signal that top-K truncation discards.

### Top-K Renorm fullseq (K=32) ✅ Complete

**Checkpoint**: `checkpoints/v1-topk32r-fullseq-m1.5b-t1.7b-math/`

| Step | avg@4 |
|------|-------|
| 50 | 61.10 |
| **100** | **62.20** |
| 150 | 61.75 |
| 200 | 61.90 |

Top-K renorm fullseq best = 62.20%. Stable but slightly below full-vocab fullseq (62.35%).

### Full Comparison: Full-Vocab vs Top-K

| Method | Best avg@4 | Best Step | Degrades? |
|--------|-----------|-----------|-----------|
| **Full-vocab pos-100 (ours)** | **63.15** | 150 | **No** ✅ |
| Full-vocab fullseq | 62.35 | 100 | Yes (→37.75) ⚠️ |
| Top-K renorm fullseq | 62.20 | 100 | No |
| Top-K renorm pos-100 | 61.90 | 100 | No |

## 2. Soft Weighting Experiments

All soft weighting methods use fullseq generation (position_limit=0) with position-dependent or entropy-dependent per-token weights.

### Exponential Decay (K=100, λ=0.015, w_min=0.05) ✅ Complete

**Checkpoint**: `checkpoints/v1-soft-exp100-m1.5b-t1.7b-math/`

| Step | avg@4 |
|------|-------|
| 50 | 55.40 |
| **100** | **56.70** |
| 150 | 54.45 |
| 200 | 55.80 |

Weak overall. Exponential decay doesn't concentrate enough weight on early positions.

### Teacher Entropy (α=2.0, H₀=2.0, w_min=0.1) 🔄 Training

**Checkpoint**: `checkpoints/v1-ent-teacher-m1.5b-t1.7b-math/`
**Server**: scai3 GPU 0+1 (step ~70/200)

| Step | avg@4 |
|------|-------|
| 50 | 54.30 |
| **100** | **57.90** |
| 150 | 54.10 ⚠️ |
| 200 | ... |

### Student Entropy (α=2.0, H₀=2.5, w_min=0.1) 🔄 Training

**Checkpoint**: `checkpoints/v1-ent-student-m1.5b-t1.7b-math/`
**Server**: scai4 GPU 3 (step 50/200)

| Step | avg@4 |
|------|-------|
| 50 | 55.15 |
| **100** | **57.45** |
| 150 | 53.60 ⚠️ |
| 200 | 53.85 ⚠️ |

### Joint Entropy (α=1.5, H₀=2.0, β=1.0, w_min=0.1) 🔄 Training

**Checkpoint**: `checkpoints/v1-ent-joint-m1.5b-t1.7b-math/`
**Server**: scai4 GPU 5 (step 50/200)

| Step | avg@4 |
|------|-------|
| 50 | 55.60 |
| **100** | **57.95** |
| 150 | 50.80 ⚠️ |
| 200 | 52.40 ⚠️ |
