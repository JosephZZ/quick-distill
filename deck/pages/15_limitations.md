# Limitations: What We Did Not Prove

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### Where positional underperforms

In **2 of 6** main math settings — both **FullFT**:

| Setting | Pos best | Fullseq |
|---|---|---|
| Qwen FullFT | 54.55 | **57.10** |
| Llama FullFT | 17.80 | **20.10** |

Slow FullFT updates partially absorb late-position noise. Practitioners running long FullFT schedules on flat-profile teachers should not assume positional dominates.

### 45% heuristic

Calibrated on **3 families** (≤8B). Presented as a starting point for tuning, not a validated law. One out-of-distribution family could break it.

</div>
<div style="width: 50%;">

### Position vs. content confound ⚠️

The **strongest critical reading**: what the prefix really filters is *format/style noise* (LaTeX, code blocks), **not late-position tokens per se**.

The cleanest disentangling experiment:

> Run **full-sequence loss with format tokens masked**.
> If it matches positional, the operative variable is *content*, not *position*.

We did **not** run this ablation — flagged as the **highest-value follow-up**. Cross-family KL decay + cascade evidence suggests position carries independent signal, but our current evidence does not isolate the two factors.

### Mechanism scope

Token classification is **Qwen-only**. Gemma/Llama mechanism inferred from consistent KL profiles, not directly verified.

### Task / scale scope

Structured reasoning only (math, function calling, code). Students ≤4B, teachers ≤8B. Open-ended generation and larger scale are open. Reverse KL only.

</div>
</div>

<!--
Honest limitations slide. Explicitly surface the position-vs-content confound from R4.
-->
