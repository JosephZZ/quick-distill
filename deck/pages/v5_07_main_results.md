# §4 Pos-K beats full-seq across the board

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### MATH-500 (Qwen 1.5B → 1.7B, n1bs16 LoRA)

| Method | Best step | avg@4 | maj@4 | pass@4 |
|--------|----------:|------:|------:|------:|
| No-distill | — | 50.95 | 61.20 | 72.80 |
| Full-seq | 150 | 62.35 | 69.40 | 74.60 |
| **Pos-50** | 150 | **66.65** | 71.00 | 81.00 |
| Pos-100 | 200 | 65.85 | 70.80 | 79.80 |
| **Pos-150** | 100 | **66.65** | 67.00 | 81.00 |
| Pos-200tok | 50 | 66.05 | 71.20 | 81.00 |

**+3.5–4.3pp avg@4 over full-seq, +4.3–7.4pp pass@4.**

### Coding (Qwen 1.5B → 1.7B, LoRA, HumanEval)

| Method | Best HE | HE+ |
|--------|--------:|----:|
| Full-seq | 40.2 (s50) → 26.8 (s400) | 25.0 |
| **Pos-50** | **42.1** (s350) | **36.6** |
| Pos-100 | 42.1 (s150) | 36.0 |

Full-seq **degrades 13pp** during training. Pos-K is stable.

</div>
<div style="width: 50%;">

### Selector comparison summary, K=100

(MATH-500 avg@4, same training budget, n1bs16 LoRA)

| Selector | avg@4 | KL covered |
|----------|------:|-----------:|
| No-distill | 50.95 | — |
| Top-KL-100 | 58.60 | 93.2% |
| Top-ent-student | 61.35 | 67.0% |
| Format-mask | 62.05 | — |
| Top-ent-teacher | 62.20 | 50.5% |
| Full-seq | 62.35 | 100% |
| Random-100 | 63.05 | 21.1% |
| **Prefix-100** | **65.85** | **45.6%** |

**Position covers only 46% of cumulative KL — and wins by 3.5pp.**

### Negative controls

| Selector | avg@4 | vs baseline |
|----------|------:|------------:|
| Middle-100 | 47.80 | **−3.15** |
| Last-100 | 50.35 | −0.60 |

**Late-token training is actively harmful** — it drags the model below the no-distill baseline. The signal isn't just absent late; it's noise.

</div>
</div>

<!--
[~2 min] The headline. Pos-K wins on math, on coding, in side-by-side selector comparison.
Negative controls (middle/last) are the strongest argument that position is not a coverage proxy:
if the budget were the only thing that mattered, mid/last wouldn't go BELOW baseline.
-->
