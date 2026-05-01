# Suffix-100+ Disentangling Experiment — Plan

## Motivation

Gemini's review (`docs/conceptual_framework_review.md`) identifies the
**cascade hypothesis** as the strongest unrefuted alternative to the
"prefix-100 = positional proxy for hiKL_hiE" framework:

> The benefit of prefix-100 is primarily *error avoidance*, not *signal
> selection*. An early student error places the trajectory in a state
> divergent from the teacher; from there, almost every later token has
> high KL — not because each is a reasoning pivot, but because the entire
> generation is off-course. The bucket distribution then becomes an
> epiphenomenon and position itself is the causal variable.

To test "selection rule" against "position causality" head-on, we
construct two runs that are identical *except* for the selection rule
applied to identical positions ≥ 100:

1. **suffix-100+** — distill on every valid token at position ≥ 100.
2. **suffix-100+_hiKL_hiE** — distill on tokens at position ≥ 100 that
   *also* satisfy `KL > p75` and `surp > p75` (per-batch).

## Predictions

| Outcome | Reading |
|---|---|
| `P(suffix-100+_hiKL_hiE) ≫ P(suffix-100+)`, approaching `P(prefix-100)` | Bucket identity is causal regardless of position. Framework rescued from cascade objection. |
| `P(suffix-100+_hiKL_hiE) ≈ P(suffix-100+)` (both low) | Late-position hiKL_hiE tokens are cascade artifacts, not real pivots. Cascade dominates. |
| `P(suffix-100+_hiKL_hiE) > P(full_seq)` would also be a strong positive signal — even depleted regions contain ~12% hiKL_hiE; selecting only those should outperform unfiltered full-seq if the bucket story holds. |

## Implementation requirements

`on_policy_distill_positional.py` currently has `--position_limit N`
(upper bound: positions ≥ N are masked). It does **not** have a lower
bound. Need to add:

```python
parser.add_argument("--position_lower", type=int, default=0,
    help="Mask out positions < N (lower bound). 0 = no lower mask.")
```

And in the selection block, intersect with `(positions >= args.position_lower)`:

```python
# Apply lower-bound position mask BEFORE token selection logic
if args.position_lower > 0:
    pos_idx = torch.arange(max_resp_len, device=student_device).unsqueeze(0)
    pos_lower_mask = pos_idx >= args.position_lower  # [1, max_resp_len]
    resp_valid_mask = resp_valid_mask & pos_lower_mask
```

This is a minimal change: it gates the existing `resp_valid_mask`, so
all downstream selection logic (prefix, top_kl, format_mask,
hi_kl_hi_surp) honors the lower bound automatically.

For per-batch quantile thresholds in `hi_kl_hi_surp`, the thresholds
will be computed *after* the lower-bound mask is applied, so p75 of
KL/surp is the p75 *within the suffix region* — which is what we want
(otherwise the global p75 would simply not select any late-position
tokens because the late distribution is shifted left).

Caution: at `position_lower=100`, max_new_tokens must remain large
(2048) — we need the model to actually *generate* into the suffix region
or the mask will select nothing. Generation cost is unchanged from the
running format-mask config.

## Run order (single GPU, sequential)

After format-mask (running) and hi_kl_hi_surp (queued) finish:

| # | Run name | Mode | position_lower | n_problems | steps |
|---|---|---|---:|---:|---:|
| 1 | suffix-100p-fullseq | fullseq (no token select) | 100 | 3200 | 200 |
| 2 | suffix-100p-hiklhie | hi_kl_hi_surp | 100 | 3200 | 200 |

Same n1bs16 LoRA recipe as everything else; same eval (avg@4 at steps
50/100/150/200).

## Decision matrix after eval

```
                       suffix-100+   suffix-100+_hiKL_hiE
                       (fullseq)      (filtered)
prefix-100 ≈ 65.85%
full-seq  ≈ 65.0%

A. ≤ 55 / ≤ 55         → cascade dominates; framework refuted in suffix region
B. ≤ 55 / 60-65        → bucket identity is causal even in cascade-contaminated region
C. ≤ 55 / ≥ 65         → strong positive: filtering rescues bad region completely
D. ≥ 60 / ≥ 60         → suffix isn't actually as harmful as we thought; cascade overstated
```

Outcome B or C cleanly supports the paper's framework against the
cascade objection. Outcome A would force a major paper revision —
probably reframing the contribution as "where to distill" rather than
"why the prefix wins."
