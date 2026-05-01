# §7 Cascading-error theory

### Aligning the planner auto-aligns the executor — but not vice versa

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### Evidence 1 — Tail KL drops *without* tail training

Trained on first 200 tokens only, then measured per-position KL out to position 700:

| Range | Raw | Pos-200tok s200 | Full-seq s200 |
|-------|----:|----------------:|--------------:|
| 0–50 | 2.064 | 1.015 | 1.036 |
| 50–100 | 0.759 | 0.322 | 0.312 |
| 100–150 | 0.495 | 0.228 | 0.236 |
| 150–200 | 0.387 | 0.248 | 0.236 |
| **200–300** | 0.382 | **0.242** | 0.273 |
| **300–400** | 0.331 | **0.229** | 0.308 |
| **400–500** | 0.317 | 0.289 | 0.202 |

Pos-200tok was **never trained on positions 200+**, yet KL there drops to roughly the same level as full-seq. **The fix propagates downstream.**

</div>
<div style="width: 50%;">

### Evidence 2 — Test-time prefix swap

Take a math problem. Generate prefix-100 with model A, then continue with model B from token 100 onward.

| Prefix | Tail | avg@4 |
|--------|------|------:|
| Pre-distill | Pre-distill | 50.95 |
| **Pos-100 student** | Pre-distill | **64.10** |
| Pre-distill | Pos-100 student | 51.85 |
| Pos-100 student | Pos-100 student | 65.85 |

**Replacing only the prefix recovers ~88% of the gain** ((64.10 − 50.95) / (65.85 − 50.95) = 88.3%). Replacing only the tail recovers ~6%.

→ The supervision lives in the prefix. The tail follows.

### Intuition: conditional drift

If the student matches the teacher on the first $k$ tokens, all conditional distributions
$p(y_t \mid y_{<t})$ for $t>k$ are evaluated on **the same context**, so the natural drift between teacher and student is bounded by the per-step on-policy disagreement (small).

If the prefix is wrong, every later conditional is on a *different* context — the teacher and student diverge geometrically.

</div>
</div>

<!--
[~2 min] The "why" — prefix supervision implicitly aligns the tail because the tail
is conditioned on the prefix. The two empirical pieces (auto tail-KL drop + prefix swap)
are mutually corroborating: same conclusion from different directions.
-->
