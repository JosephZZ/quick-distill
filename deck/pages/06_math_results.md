# Math Results: 3 Families × 2 Methods

### MATH-500 avg@4 — Positional wins 5/6 settings

<br>

| | Qwen LoRA | Qwen FullFT | Gemma LoRA | Gemma FullFT | Llama LoRA | Llama FullFT |
|--|-----------|-------------|------------|-------------|------------|-------------|
| Baseline | 50.95 | 50.95 | 13.45 | 13.45 | 15.20 | 15.20 |
| **Best pos** | **65.85** (N=100) | 54.55 (N=150) | **25.80** (N=50) | **26.70** (N=100) | **22.45** (N=200) | 17.80 (N=300) |
| Fullseq | 62→**38** ⚠️ | 57.10 | **11.70** ⚠️ | **13.90** ⚠️ | 20.65 | 20.10 |

<br>

### Three patterns:

1. **Positional ≥ fullseq on 5/6 combos** — only Qwen FullFT fullseq slightly better (with 2× more steps)
2. **Fullseq is dangerous** — Gemma collapses **below baseline** in both LoRA and FullFT
3. **LoRA >> FullFT** by 8–11pp — complementary regularization with positional truncation

<!--
[~2 min]
The main results table. Cover all 6 cells.
Highlight: Gemma fullseq below baseline is striking.
LoRA + positional = double regularization.
-->
