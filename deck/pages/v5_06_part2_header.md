# Part 2 — Position is a great strategy for token selection

<br>

### What we'll show

1. **Pos-K beats full-seq across configurations.** Math, code, function-calling. LoRA and full FT.
2. **Generality (好 / 快 / 稳).** Cross-model (Qwen / Gemma / Llama), cross-scale, cross-method.
3. **Stability.** Where full-seq collapses on 2/3 seeds, pos-K is stable on 3/3.
4. **Efficiency.** ~10× wall-clock, ~4× memory, no extra hyperparameters.
5. **N-selection.** A single elbow at N ≈ 50–150 across tasks.

<br>

> One-line code change. Strict superset of the gains. No new failure modes.

<!--
[~30s] Section header for the "好快稳" pitch.
Sets up the four payoff slides.
-->
