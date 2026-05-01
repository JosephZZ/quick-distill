# Conceptual Framework — External Review (Gemini)

Date: 2026-04-26 (overnight session)

## Context

Empirical finding (already in paper): prefix-100 LoRA distillation hits 65.85%
on MATH-500, beating full-seq (~65%) and a no-distill baseline (50.95%).
Reviewer concern: this is "a trick, not a mechanism."

Proposed framework: prefix-100 wins because the prefix region is naturally
enriched in `hiKL_hiE` tokens (student uncertain AND disagrees with teacher
= "reasoning pivots"), and naturally depleted in `hiKL_loE` tokens (student
overconfident, format/habit). Direct test: train only on `hiKL_hiE` tokens
across the full sequence and compare vs prefix-100.

Distributional evidence (`docs/position_x_bucket.md`):

| Position | n     | hiKL_hiE | hiKL_loE | mean KL |
|----------|-------|----------|----------|---------|
| 0-100    | 9991  | 46.5%    | 3.1%     | 1.41    |
| 100-300  | 18967 | 24.6%    | 4.2%     | 0.41    |
| 300-500  | 15595 | 17.2%    | 4.8%     | 0.32    |
| 500+     | 29996 | 11.9%    | 4.1%     | 0.21    |

Prefix-100 is 2.22× enriched in hiKL_hiE relative to the full-sequence
baseline (~21%). The framework: prefix-100 ≈ a positional proxy for the
principled rule "select tokens where the student is uncertain AND the
teacher disagrees."

## Skeptical critique (Gemini)

### Strongest counter-explanation: the cascade hypothesis

The benefit of prefix-100 may be primarily **error avoidance**, not **signal
selection**. An early student error (e.g. token 50) places the trajectory
in a state divergent from the teacher's reasoning path; from there, nearly
every subsequent token has high KL — not because each is an independent
"reasoning pivot," but because the entire generation is off-course. In
this view, prefix-100 succeeds by avoiding the "garbage time" that follows
an unrecoverable error cascade. The aggregate bucket distribution is then
an *epiphenomenon* — the bucket identity does not carry the causal weight;
position does.

Subordinate concern: KL is a noisy proxy for "usefulness." High KL can
also reflect teacher verbosity, stylistic preferences, or alternative
(equally valid) reasoning paths. Prefix tokens may be more constrained
(problem setup, equation copy) and thus the KL signal is more reliable
*there*, independent of the bucket-share story.

### Predicted patterns

**Supports framework:**
- `P(hi_kl_hi_surp) > P(prefix-100)` — smoking gun: explicit pivot
  selection from the whole sequence beats the positional heuristic, so
  bucket identity is causal.
- `P(full_seq) < P(format_mask) < P(prefix-100)` — format mask removes
  harmful gradient but doesn't up-sample useful gradient; prefix-100 does
  both naturally.

**Refutes framework:**
- `P(hi_kl_hi_surp) ≤ P(full_seq)` — major blow. The "best" tokens across
  the full sequence carry no marginal value, suggesting their high KL/
  surprise is a *symptom* of a problem (cascade), not a signal of value.
- prefix-100 stays the best — implies position itself (or cascade
  dynamics tied to position) is the primary causal factor.

### Decisive disentangling experiment (Gemini's proposal)

**Inverted prefix / suffix-100+ experiments.** Position is intrinsically
tied to cascade; we need to test selection-rule vs position in opposition:

1. **suffix-100+** — distill only on tokens at position ≥ 100. Both
   hypotheses predict poor performance (depleted of hiKL_hiE, AND
   contaminated by cascade). Sets a new low baseline.
2. **suffix-100+_hiKL_hiE** — from the suffix-100+ region, train only on
   the hiKL_hiE tokens that *do* exist there.

**Interpretation:**
- If `P(suffix-100+_hiKL_hiE) ≫ P(suffix-100+)` and approaches
  `P(full_seq)` or higher → bucket identity is causal regardless of
  position. The framework is rescued from the cascade objection.
- If `P(suffix-100+_hiKL_hiE) ≈ P(suffix-100+)` (low) → tokens flagged
  as hiKL_hiE in late positions are cascade artifacts, not true pivots.
  The position-causal / cascade hypothesis dominates.

This is the cleanest available test: it pits *token identity* against
*position*, with each hypothesis making a sharply different prediction.

## Implications for the paper

The current paper would survive cascade-only criticism if both:
- `hi_kl_hi_surp` matches/beats prefix-100 (ongoing)
- `suffix-100+_hiKL_hiE` substantially outperforms `suffix-100+` (NOT YET
  RUN)

If only the first holds, the framework is *consistent* but cascade is
still a viable confound and the reviewer can press on it.

If both hold, the bucket-identity-is-causal story is well-grounded and
the paper has a real mechanism, not a trick.

## Experiment queue (overnight, single GPU)

Ordered by information-per-GPU-hour:

1. **format_mask** (running) — tests "remove harmful gradient" leg.
2. **hi_kl_hi_surp** — tests "select useful gradient" leg.
3. **suffix-100+_hiKL_hiE** — disentangles selection vs position. Needs
   a `--position_lower` flag in `on_policy_distill_positional.py`
   (only existing flag is `--position_limit` for upper bound).
4. (optional) **suffix-100+** plain — if (3) outperforms expectations,
   we want a clean lower bound for context.

Implementation note for (3): add `--position_lower N` that zeroes the
loss mask for positions < N, then combine with `token_select_mode
hi_kl_hi_surp`. Mini-bs and `gen_batch_size 4` settings carry over.
