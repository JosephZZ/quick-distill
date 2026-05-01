# Part 1 — Which tokens carry the signal?

## Three candidates: KL, entropy, position

<br>

We hold the loss budget fixed at **K=100 tokens per response** and ask which selector best concentrates the supervision.

<br>

### Setup (locked)

- Student / Teacher: Qwen2.5-Math-1.5B → Qwen3-1.7B
- Eval: MATH-500, avg@4 (n=4 samples, t=0.7)
- Training: LoRA r=32 / α=64, 200 steps, n=1 sample × bs=16 over 3200 problems
- Baselines: **no-distill = 50.95%** , **full-seq = 62.35%**

<br>

### A natural prior — "follow the mass"

Among the **L ≈ 974** tokens of an average response (full-seq run, step 200), where does the reverse-KL / entropy mass concentrate?

| Selector (K=100) | % of total reverse-KL mass | % of total teacher-entropy mass |
|---|---:|---:|
| Top-100 by KL | **97.1%** | — |
| Top-100 by teacher-entropy | — | **95.1%** |
| Prefix-100 (positional) | 34.7% | 32.6% |

If supervision tracks where the teacher disagrees, top-K-by-KL covers ~all of it; top-K-by-entropy covers ~all of the uncertainty. **Prefix-100 catches barely a third of either.**

<br>

### Reading the result against this prior

We compare each indicator to full-seq, not as a strict pass/fail but as three regimes:

- **Below full-seq** → indicator selects *against* signal (the budget hurts).
- **≈ full-seq** → indicator captures roughly random-budget value.
- **Above full-seq** → indicator captures structure that full-seq dilutes.

The next three slides walk through KL → entropy → position in this order. The mass-coverage prior predicts KL > entropy > position. **The result inverts that ordering.**

<!--
[~1.5 min] Reframe Part 1: the question is which tokens carry signal.
Set up the mass-coverage prior so the audience expects KL or entropy to win.
The inversion lands harder when the prior is concrete.
-->
