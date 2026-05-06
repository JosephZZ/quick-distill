# Soft Weighting Methods for Positional Distillation

## Motivation

Our positional distillation experiments show that hard truncation at position N works well (pos-100 and pos-200 are best), but introduces a sharp discontinuity: tokens at position N-1 get full weight, tokens at position N get zero weight. This is suboptimal because:

1. The KL signal decays gradually (3.06 at pos-0, 0.69 at pos-50, 0.45 at pos-100, 0.36 at pos-192), not as a step function.
2. Late tokens still carry *some* useful signal---throwing it all away loses information.
3. The optimal truncation point varies per trajectory (some responses are short and dense, others long and diluted).

Soft weighting replaces the binary mask `sel_mask` with a continuous weight vector `w(t) in [w_min, 1.0]`, applied to `per_pos_kl` before averaging.

---

## Method 1: Position-Based Exponential Decay

### Concept

Full weight for the first K tokens (the "plateau"), then exponential decay toward a floor value.

### Formula

```
w(t) = max(w_min, exp(-lambda * max(0, t - K)))
```

where:
- `t` = token position in the response (0-indexed)
- `K` = plateau length (positions 0..K-1 get weight 1.0)
- `lambda` = decay rate (controls how fast weight falls)
- `w_min` = minimum weight floor (prevents complete zeroing)

Equivalently, if you want to parameterize by "reach" (the position R where weight drops to w_min):

```
lambda = -ln(w_min) / (R - K)
```

So specifying K=100, R=300, w_min=0.1 gives lambda = -ln(0.1)/200 = 0.01151.

### Hyperparameters and Their Effects

| Parameter | Range | Effect |
|-----------|-------|--------|
| K (plateau) | 50-200 | Positions with full weight. Our best results are pos-100 to pos-200, so K in this range is natural. |
| lambda (decay rate) | 0.005-0.05 | Controls tail contribution. lambda=0.01 halves weight every ~70 tokens. lambda=0.05 halves every ~14 tokens (nearly hard truncation). |
| w_min (floor) | 0.0-0.2 | Minimum contribution of any token. 0.0 = asymptotic hard truncation. 0.1 = late tokens contribute 10%. |

### Properties

- `w(0) = ... = w(K-1) = 1.0` (plateau is exactly 1)
- `w(K) = 1.0` (continuous at transition)
- `w(t) -> w_min` as `t -> infinity`
- Monotonically non-increasing
- Sum of weights for a response of length L: `K + (1/lambda) * (1 - exp(-lambda*(L-K)))` (approximately, ignoring floor)

### Edge Cases

- K=0: Pure exponential decay from position 0, no plateau
- K >= L (response length): All tokens get weight 1.0 (equivalent to full-seq)
- lambda -> infinity: Hard truncation at position K
- lambda -> 0: Uniform weight 1.0 everywhere
- w_min = 0: Equivalent to very soft truncation

### PyTorch Implementation

```python
def positional_decay_weights(resp_lens, K, decay_lambda, w_min, max_len, device):
    """
    Args:
        resp_lens: [batch_size] actual response lengths
        K: plateau length
        decay_lambda: exponential decay rate
        w_min: minimum weight
        max_len: max response length in batch
        device: torch device
    Returns:
        weights: [batch_size, max_len] soft weights
    """
    positions = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
    decay_dist = (positions - K).clamp(min=0).float()  # [1, max_len]
    weights = torch.exp(-decay_lambda * decay_dist)  # [1, max_len]
    weights = weights.clamp(min=w_min)  # floor

    # Zero out padding positions
    valid_mask = positions < resp_lens.unsqueeze(1)  # [batch_size, max_len]
    weights = weights.expand_as(valid_mask) * valid_mask.float()

    return weights
```

### Integration into Loss Computation

Replace the current binary `sel_mask.float()` with soft weights:

```python
# Current code (hard truncation):
# per_pos_kl = per_pos_kl * sel_mask.float()
# n_sel_per_traj = sel_mask.float().sum(dim=-1).clamp(min=1)
# loss_per_traj = per_pos_kl.sum(dim=-1) / n_sel_per_traj

# Soft weighting:
weights = positional_decay_weights(resp_lens, K, decay_lambda, w_min, max_len, device)
weighted_kl = per_pos_kl * weights
loss_per_traj = weighted_kl.sum(dim=-1) / weights.sum(dim=-1).clamp(min=1)
```

### Recommended Starting Configurations

| Config Name | K | lambda | w_min | Rationale |
|-------------|---|--------|-------|-----------|
| `soft-100` | 100 | 0.015 | 0.05 | Matches pos-100 plateau, gentle tail |
| `soft-200` | 200 | 0.02 | 0.1 | Matches pos-200 plateau, moderate tail |
| `soft-50-long` | 50 | 0.005 | 0.1 | Short plateau, very long tail (almost uniform) |
| `sharp-100` | 100 | 0.05 | 0.0 | Nearly hard truncation for comparison |

### Pros

- Simple, no extra forward passes or computation
- Intuitive: matches the empirically observed KL decay curve
- One obvious ablation axis (lambda) between hard truncation and full-seq
- Fully deterministic---same weight every time for position t

### Cons

- Position-based, not content-aware: assumes all trajectories have the same signal structure
- Cannot adapt to variable-quality regions (e.g., a trajectory where late tokens happen to be highly informative)
- Three hyperparameters to tune (though K can be fixed from prior experiments)

---

## Method 2: Entropy-Regulated Weighting

### The Entropy Interpretation Problem

Before defining formulas, we must address the key question: **does high entropy mean "informative" or "noisy"?**

The answer depends on *whose* entropy and *what it signals*:

| Source | High Entropy Means | Implication for Weighting |
|--------|--------------------|---------------------------|
| **Teacher entropy** | Teacher is uncertain about the right token | Could be informative (genuine ambiguity) or noisy (teacher doesn't know). Likely **informative**: these are positions where the student can learn the most about the teacher's soft distribution. |
| **Student entropy** | Student is uncertain | Student hasn't learned this pattern yet. High student entropy + low teacher entropy = student needs to learn here. High student entropy + high teacher entropy = genuinely hard. |
| **KL divergence** (not entropy, but related) | Large teacher-student gap | Most direct signal of where distillation has the most to teach. Already used in `top_kl` selection mode. |

**Key insight**: Entropy alone is ambiguous. The *relationship* between teacher and student entropy is what matters:

- **High teacher entropy, low student entropy**: Student is confidently wrong. Very informative---high weight.
- **High teacher entropy, high student entropy**: Genuinely ambiguous. Moderately informative.
- **Low teacher entropy, high student entropy**: Student doesn't know, but teacher is confident. Very informative---high weight.
- **Low teacher entropy, low student entropy**: Both agree. Low information content---low weight (but still train on it for stability).

This suggests the best entropy-based weight is a function of *disagreement*, not raw entropy. However, we already have KL divergence which captures exactly this. So entropy-based weighting should focus on something KL doesn't capture.

### Variant 2A: Teacher Entropy Weighting

**Rationale**: At positions where the teacher has a rich, spread-out distribution (high entropy), the student can learn more nuanced token probabilities. At positions where the teacher puts 99% mass on one token, there's less distributional information to transfer.

```
H_T(t) = -sum_v p_T(v|t) * log p_T(v|t)     (teacher entropy at position t)

w(t) = w_min + (1 - w_min) * sigmoid(alpha * (H_T(t) - H_0))
```

where:
- `H_T(t)` = teacher entropy at position t
- `H_0` = entropy threshold (center of sigmoid)
- `alpha` = sharpness of transition
- `w_min` = minimum weight

**Properties**:
- `w(t) in [w_min, 1.0]`
- When H_T(t) >> H_0: w(t) -> 1.0 (high teacher entropy = high weight)
- When H_T(t) << H_0: w(t) -> w_min (low teacher entropy = low weight)
- Sigmoid provides smooth, bounded transition

**Hyperparameters**:

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| H_0 | 1.0-3.0 nats | Entropy threshold. For a 150k vocab, max entropy is ~12 nats. Typical token entropy is 1-4 nats. |
| alpha | 1.0-5.0 | Sharpness. alpha=1 is gradual, alpha=5 is nearly binary. |
| w_min | 0.1-0.3 | Floor weight for low-entropy (confident) positions. |

### Variant 2B: Student Entropy Weighting

**Rationale**: Where the student is most uncertain, it has the most to learn. Caveat: must use **stop-gradient** to prevent the student from gaming the weights by increasing its own entropy.

```
H_S(t) = -sum_v p_S(v|t) * log p_S(v|t)     (student entropy, detached)

w(t) = w_min + (1 - w_min) * sigmoid(alpha * (H_S(t) - H_0))
```

**Critical**: H_S must be computed with `torch.no_grad()` / `.detach()`. Otherwise, the student is incentivized to increase its entropy to increase the weight on positions where it has high loss, creating a degenerate feedback loop.

### Variant 2C: Joint Entropy / Disagreement Weighting

**Rationale**: Use the *difference* between student and teacher entropy as a signal. Large disagreement in uncertainty = position where knowledge transfer is most valuable.

```
D(t) = |H_S(t) - H_T(t)|     (entropy disagreement)

w(t) = w_min + (1 - w_min) * sigmoid(alpha * (D(t) - D_0))
```

Or, use a combined score:

```
w(t) = w_min + (1 - w_min) * sigmoid(alpha * (H_T(t) + beta * H_S(t) - H_0))
```

where beta controls the relative importance of student vs teacher entropy. Setting beta=0 recovers Variant 2A; beta=1 weights by joint uncertainty.

### PyTorch Implementation (All Variants)

```python
def compute_entropy(log_probs):
    """Compute entropy from log probabilities.
    Args:
        log_probs: [batch_size, seq_len, vocab_size] log probabilities
    Returns:
        entropy: [batch_size, seq_len]
    """
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    # Handle numerical issues: clamp negative values from floating point
    return entropy.clamp(min=0.0)


def entropy_weights(teacher_log_probs, student_log_probs, variant,
                    alpha=2.0, H_0=2.0, w_min=0.1, beta=1.0):
    """
    Args:
        teacher_log_probs: [batch, seq_len, vocab] teacher log probs
        student_log_probs: [batch, seq_len, vocab] student log probs (detached)
        variant: "teacher", "student", or "joint"
        alpha: sigmoid sharpness
        H_0: entropy threshold
        w_min: minimum weight
        beta: student entropy coefficient (for joint variant)
    Returns:
        weights: [batch, seq_len]
    """
    with torch.no_grad():
        H_T = compute_entropy(teacher_log_probs)
        H_S = compute_entropy(student_log_probs.detach())

        if variant == "teacher":
            signal = H_T
        elif variant == "student":
            signal = H_S
        elif variant == "joint":
            signal = H_T + beta * H_S
        elif variant == "disagreement":
            signal = (H_S - H_T).abs()
        else:
            raise ValueError(f"Unknown variant: {variant}")

        weights = w_min + (1.0 - w_min) * torch.sigmoid(alpha * (signal - H_0))

    return weights
```

### Integration into Loss

```python
# Compute entropy-based weights (no grad, since weights are not trainable)
ent_weights = entropy_weights(
    t_log_probs_padded, s_log_probs_resp,
    variant="teacher", alpha=2.0, H_0=2.0, w_min=0.1
)
# Optionally combine with positional decay
# combined_weights = positional_weights * ent_weights

# Apply to per-position KL
weighted_kl = per_pos_kl * ent_weights * resp_valid_mask.float()
loss_per_traj = weighted_kl.sum(dim=-1) / ent_weights.sum(dim=-1).clamp(min=1)
```

### Recommended Configurations

| Config | Variant | alpha | H_0 | w_min | Rationale |
|--------|---------|-------|-----|-------|-----------|
| `ent-teacher` | teacher | 2.0 | 2.0 | 0.1 | Upweight rich distributions |
| `ent-student` | student | 2.0 | 2.5 | 0.1 | Upweight uncertain positions |
| `ent-joint` | joint | 1.5 | 3.5 | 0.1 | Upweight positions where either is uncertain |
| `ent-disagree` | disagreement | 3.0 | 1.0 | 0.1 | Upweight positions where models disagree on confidence |

### Pros

- Content-aware: adapts per-trajectory and per-position
- Can capture signal quality beyond just position
- Naturally handles variable-length responses
- Teacher entropy is essentially "free" (already computed in the forward pass)

### Cons

- Student entropy requires stop-gradient care to avoid degenerate training dynamics
- Adds conceptual complexity and hyperparameters (alpha, H_0)
- Entropy is noisy---a single token's entropy can be high for uninteresting reasons (e.g., the teacher model is just bad at punctuation)
- The sigmoid parameterization means H_0 must be tuned to the actual entropy distribution, which varies across tasks and training steps

---

## Method 3: Combined Position + Entropy (Recommended)

### Rationale

Position and entropy capture orthogonal information:
- Position captures the *structural* signal decay (early reasoning tokens > late tokens)
- Entropy captures *per-instance* signal quality (informative vs trivial positions)

Combining them multiplicatively gives a weight that is high only when both the position is early *and* the content is informative.

### Formula

```
w(t) = w_pos(t) * w_ent(t)

where:
  w_pos(t) = max(w_min_pos, exp(-lambda * max(0, t - K)))
  w_ent(t) = w_min_ent + (1 - w_min_ent) * sigmoid(alpha * (H_T(t) - H_0))
```

The product preserves the overall shape (decaying with position) but modulates it with local signal quality.

### Properties

- `w(t) in [w_min_pos * w_min_ent, 1.0]`
- Late, low-entropy tokens get the minimum weight (product of two small values)
- Early, high-entropy tokens get full weight
- Late but high-entropy tokens get moderate weight (position penalizes, entropy boosts)

---

## Method 4: KL-Aware Reweighting (Alternative)

Since we already compute per-position KL during the forward pass, we can use it directly as a self-weighting signal.

### Formula

```
w(t) = sigmoid(alpha * (KL(t) - KL_0))
```

where KL(t) is the per-position forward KL (teacher-student), detached.

**Important**: This must use `detach()` to prevent gradient flow through the weights.

### Why This Might Not Work

This is essentially self-reinforcing: positions with high KL get high weight, which pushes the loss to focus on them, which may cause training instability. The positions with highest KL are often the ones where the student is worst, and focusing exclusively on those can destabilize learning (akin to hard example mining without safety mechanisms).

A damped version is safer:

```
w(t) = 1.0 + gamma * (sigmoid(alpha * (KL(t) - KL_0)) - 0.5)
```

This centers weights around 1.0, with KL providing a small modulation (+/- gamma/2). Setting gamma=0.5 gives weights in [0.75, 1.25].

---

## Summary Comparison

| Method | Content-Aware | Extra Compute | Hyperparams | Stability Risk | Expected Benefit |
|--------|--------------|---------------|-------------|----------------|-----------------|
| Hard truncation (current) | No | None | 1 (K) | Low | Baseline |
| Exponential decay | No | Negligible | 3 (K, lambda, w_min) | Low | Moderate: recovers some late-token signal |
| Teacher entropy | Yes | ~Free (already have teacher logprobs) | 3 (alpha, H_0, w_min) | Low | Moderate: focuses on rich distributions |
| Student entropy | Yes | ~Free (already have student logits) | 3 (alpha, H_0, w_min) | Medium (stop-grad critical) | Moderate: focuses on student weakness |
| Combined pos+ent | Yes | ~Free | 5-6 | Low | High: best of both worlds |
| KL reweighting | Yes | ~Free | 3 (alpha, KL_0, gamma) | Medium-High | Uncertain: risk of instability |

## Recommended Experiment Plan

**Phase 1**: Position-based decay (simplest, cheapest to run)
1. `soft-100`: K=100, lambda=0.015, w_min=0.05
2. `soft-200`: K=200, lambda=0.02, w_min=0.1
3. Compare against pos-100 and pos-200 baselines

**Phase 2**: Teacher entropy weighting
1. First, measure the empirical entropy distribution from existing trajectories to set H_0 properly
2. `ent-teacher` with alpha=2.0, H_0 from empirical median
3. Combined `pos-100 + ent-teacher`

**Phase 3**: If Phase 2 shows promise, try joint/disagreement variants

---

## Relevant Literature

1. **Hinton et al. (2015)** - "Distilling the Knowledge in a Neural Network". Original KD paper. Uses temperature to soften distributions, which is conceptually related to entropy-based weighting (higher temperature = higher entropy = more information in soft targets).

2. **Furlanello et al. (2018)** - "Born Again Neural Networks". On-policy distillation where student learns from its own generation. Relevant context for why positional signal quality matters.

3. **Tang et al. (2020)** - "Understanding and Improving Knowledge Distillation". Analyzes when KD helps and when it hurts. Finds that "dark knowledge" (non-argmax probabilities) is more useful at high-entropy positions.

4. **Wen et al. (2023)** - "f-Divergence Minimization for Sequence-Level Training". Discusses token-level weighting in sequence-level training, including importance weighting and its stability properties.

5. **Agarwal et al. (2024)** - "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes". GKD paper. Relevant for understanding on-policy dynamics.

6. **Lin et al. (2020)** - "Autoregressive Knowledge Distillation through Imitation Learning". Discusses the compounding error problem in autoregressive distillation, which is related to why early-token signal matters more.

7. **Curriculum Learning (Bengio et al., 2009)** - The positional decay is conceptually a curriculum: train more on "easy" (early, high-signal) positions and less on "hard" (late, noisy) positions.

---

## Implementation Notes for This Codebase

The integration point is in `on_policy_distill_positional.py` around lines 1434-1460 (same-tokenizer path). Currently:

```python
# Line 1456: Binary mask application
per_pos_kl = per_pos_kl * sel_mask.float()
# Line 1458-1459: Average over selected positions
n_sel_per_traj = sel_mask.float().sum(dim=-1).clamp(min=1)
loss_per_traj = per_pos_kl.sum(dim=-1) / n_sel_per_traj
```

To add soft weighting, add a new `--weight_mode` flag with options `{hard, exp_decay, entropy_teacher, entropy_student, combined}` and corresponding hyperparameter flags. The soft weights replace `sel_mask.float()` with a continuous weight tensor. The normalization denominator changes from count of selected tokens to sum of weights.

For backward compatibility (per CLAUDE.md design principles), the default should remain `hard` (current behavior), and soft weighting modes should be additive options that don't change existing experiment reproducibility.
