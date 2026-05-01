# Math Results: 3 Families × 2 Methods

### MATH-500 avg@4 (n=4, T=0.7, last-boxed extraction) — Positional ≥ fullseq in **4 of 6** settings

<br>

| | Qwen LoRA | Qwen FullFT | Gemma LoRA | Gemma FullFT | Llama LoRA | Llama FullFT |
|--|-----------|-------------|------------|-------------|------------|-------------|
| Baseline | 50.95 | 50.95 | 13.45 | 13.45 | 15.20 | 15.20 |
| **Best pos** | **65.85** (N=100) | 54.55 (N=150) | **25.80** (N=50) | **26.70** (N=100) | **22.45** (N=200) | 17.80 (N=300) |
| Fullseq | 63.55 | 57.10 | **11.70** ⚠️ | **13.90** ⚠️ | 20.65 | 20.10 |

<br>

### Three patterns:

1. **Positional ≥ fullseq on 4/6 combos.** Exceptions: Qwen FullFT (57.10 vs 54.55) and Llama FullFT (20.10 vs 17.80) — both FullFT settings where slow updates partially absorb late-position noise.
2. **Fullseq is dangerous.** Gemma collapses **below baseline** in both LoRA and FullFT; Qwen fullseq is unstable across seeds (next slide).
3. **LoRA + positional** = double regularization (parameter + signal). Together: best of both.

<br>

*All numbers are n1bs16 LoRA / n1bs16 FullFT, single seed (42 for Qwen). Qwen multi-seed in next slide.*

<!--
[~2 min]
The main results table. Cover all 6 cells.
Highlight: Gemma fullseq below baseline is striking.
LoRA + positional = double regularization.
-->
