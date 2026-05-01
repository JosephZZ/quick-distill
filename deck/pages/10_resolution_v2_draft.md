# Resolving the Paradox — Mechanism Story (DRAFT)

> **Status: draft, gated on experimental results.**
> Activate this slide ONLY IF `P(hi_kl_hi_surp) ≥ 0.95 × P(prefix-100)` AND
> the suffix-100+ disentangling experiment shows `P(suffix-100+_hiKL_hiE) >
> P(suffix-100+) + 5pp`. Otherwise the original slide 10 (honest "open
> problem") remains the safer claim.

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### The principle: select on (KL, surprise), not on KL alone

Per-token bucketing at $p_{75}$:

| Bucket | Student | Teacher | Interpretation |
|---|---|---|---|
| **hiKL_hiE** | uncertain | disagrees | reasoning pivot |
| hiKL_loE | confident | disagrees | format / habit |
| loKL_hiE | uncertain | agrees | coverage |
| loKL_loE | confident | agrees | easy |

Top-KL alone fails because it's dominated by **hiKL_loE** (5.9% of
tokens, 23% of total KL mass) — student is locally overconfident in
notation, teacher prefers a slightly different token.

### Prefix-100 = positional proxy for hiKL_hiE selection

| Position | n | hiKL_hiE share | Enrichment |
|---|---:|---:|---:|
| **0–100** | 9,991 | **46.5%** | **2.22×** |
| 100–300 | 18,967 | 24.6% | 1.17× |
| 300–500 | 15,595 | 17.2% | 0.82× |
| 500+ | 29,996 | 11.9% | 0.57× |

Prefix concentrates "useful gradient" 2× and *avoids* "format/habit"
gradient — same direction the principle predicts.

</div>
<div style="width: 50%;">

### Two direct tests

**(a) hi_kl_hi_surp** — train on tokens with $|\Delta\mathrm{lp}| > p_{75}$
AND $-\mathrm{lp}_s > p_{75}$, regardless of position.
**Result: avg@4 = TBD%** (vs prefix-100 65.85%).

**(b) suffix-100+_hiKL_hiE** — apply the same selection rule but only
to positions ≥ 100. This pits *token identity* against *position* in
opposition.
**Result: avg@4 = TBD%** (vs suffix-100+ TBD%).

If (a) ≥ prefix-100 → bucket identity is causal.
If (b) ≫ suffix-100+ → bucket identity is causal *even in the cascade-
contaminated suffix region* — closing the strongest counter-explanation.

### What's no longer "open problem"

Old slide concession: "format-ness is a property of the teacher
distribution, not the emitted character — there is no parameter-free
disentangling experiment."

Updated: the (KL, surprise) bucket *is* a property of the joint
student-teacher distribution. The selection rule has two threshold
parameters ($p_{75}$ each), but they are not designer-dependent — any
quantile produces the same qualitative ranking.

</div>
</div>

<br>

### What we honestly have (revised)

1. **Distributional bridge**: prefix-100 is 2.22× enriched in hiKL_hiE,
   0.77× in hiKL_loE — naturally instantiates the principled rule.
2. **Direct selection** (hi_kl_hi_surp): TBD-pp gap to prefix-100.
3. **Cascade test** (suffix-100+_hiKL_hiE vs suffix-100+): TBD-pp gap.
4. **Cascade effect** (already in deck): pos-200 reduces KL by 35–38%
   at *untrained* positions 200–400 — independent positional dynamic.

> **Bottom line (conditional on results)**: the prefix advantage
> reduces to "select reasoning pivots; avoid format/habit gradients."
> Position is a cheap, on-policy-stable proxy; (KL, surprise) is the
> underlying mechanism.

<!--
Author note: Replace this slide with the v1 honest-limitation version
if the conditional results don't materialize. Don't claim mechanism
without the disentangling experiment.
-->
