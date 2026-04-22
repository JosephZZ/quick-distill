# Resolving the Paradox

### Why does high-KL selection fail?

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

**High-KL tokens are where the teacher is *confused*, not *superior*:**

| Token set | Teacher Entropy | KL |
|-----------|----------------|-----|
| Top-KL (K=100) | **0.449** (confused) | 4.09 |
| Prefix (0–99) | 0.235 (confident) | 2.02 |

<br>

High KL = teacher is uncertain AND disagrees with student

→ **Noisy gradient signal**, not useful correction

</div>
<div style="width: 50%;">

**Why prefix works despite lower KL:**

1. **Contiguity** — coherent gradient across adjacent tokens
2. **Strategy tokens** — early positions encode *what approach to take*
3. **Teacher independence** — teacher hasn't been conditioned by student yet
4. **Signal quality > signal quantity** — 46% of KL but the *right* 46%

<br>

**The lesson:** In on-policy KD, position is a better proxy for signal quality than any information-theoretic measure.

</div>
</div>

<!--
[~1.5 min]
The resolution: high KL ≠ high quality.
Position is a proxy for teacher independence, which is the actual quality metric.
-->
