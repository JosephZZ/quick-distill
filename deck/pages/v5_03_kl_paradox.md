# §1 KL is the wrong indicator

### Top-K by per-token KL divergence — the most "principled" choice — performs *worst*

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### MATH-500, K=100, n1bs16 LoRA

| Selector | avg@4 | vs full-seq |
|----------|------:|------------:|
| No-distill baseline | 50.95 | −11.40 |
| **Top-KL-100** | **58.60** | **−3.75** |
| Full-seq | 62.35 | 0.00 |

Top-KL **drops 3.75pp below full-seq** despite covering >90% of total per-token KL mass.

</div>
<div style="width: 50%;">

### Why? High-KL ≠ teaching signal

Token classification at high-KL positions:

| Category | Mean KL | What it is |
|----------|--------:|-----------|
| LaTeX format | **3.99** | `\(`, `\[`, `\\` |
| Planning | 1.75 | "To", "First" |
| Structural | 0.89 | `**`, `:` |
| Math operators | 0.38 | `=`, `+`, `−` |
| Numbers | 0.28 | 0–9 |

**Highest KL = format/style disagreements,** not reasoning quality.
Training on top-KL tokens teaches the student how the teacher *types*, not how it *thinks*.

</div>
</div>

<br>

> KL is a divergence metric, not a teaching signal.

<!--
[~2 min] First negative result. Top-KL is the obvious choice and it loses badly.
Then the diagnostic table explains why — high-KL tokens are dominated by LaTeX format.
This sets up the bar: any *good* indicator must beat full-seq, not just baseline.
-->
