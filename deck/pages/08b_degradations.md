# Per-Step Trajectories: Fullseq Degrades, Positional Doesn't

### ⚠️ = below or at no-distillation baseline. **bold** = best step in that column.

<div style="display: flex; gap: 1.2rem; align-items: flex-start; margin-top: 0.5rem;">
<div style="width: 50%;">

### Coding (Qwen LoRA, HumanEval pass@1)

Baseline: 32.93%. **Fullseq monotonically collapses from step 50 onward.**

| Step | pos-50 | pos-100 | pos-150 | full-seq |
|---:|---:|---:|---:|---:|
| 50  | 37.8 | 37.2 | 36.6 | **40.2** |
| 100 | 39.0 | 39.0 | 35.4 | 31.7 ⚠️ |
| 150 | 39.6 | **42.1** | 36.6 | 32.3 ⚠️ |
| 200 | 41.5 | 37.8 | 39.0 | 32.9 ⚠️ |
| 250 | 40.2 | 39.0 | 41.5 | 27.4 ⚠️ |
| 300 | 40.9 | 37.8 | 39.6 | 28.0 ⚠️ |
| 350 | **42.1** | 37.8 | 38.4 | 26.8 ⚠️ |
| 400 | 40.9 | 38.4 | 37.2 | 26.8 ⚠️ |

Δ vs baseline: **pos +9pp, fullseq −6pp** (step 400).

</div>
<div style="width: 50%;">

### Math (Qwen LoRA n1bs16, MATH-500 avg@4, last-boxed)

3 seeds at step 200. Baseline: 50.95%.

| Seed | pos-100 | full-seq |
|---:|---:|---:|
| 42  | **65.85** | 62.35 |
| 123 | 61.00 | **50.45 ⚠️** |
| 456 | 60.40 | 57.55 |
| **Mean ± Std** | **62.42 ± 3.0** | 56.78 ± 6.0 |

Per-step for seed 42 (n1bs16):

| Step | pos-100 | full-seq (n1bs16 trace) |
|---:|---:|---:|
| 50  | 63.75 | 61.00 |
| 100 | 64.45 | 62.00 |
| 150 | 65.15 | **62.35** |
| 200 | **65.85** | 61.20 |

Pos still climbing at step 200; fullseq plateaus then dips on this seed — and *crashes* below baseline on seed 123.

</div>
</div>

<br>

### Cross-family (best step, last-boxed for math; full_acc for funcall):

| Setting | Baseline | Pos best | Step | Full-seq | Step | Δ pos | Δ full-seq |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 2B → 4B, math LoRA | 13.45 | **25.80** | 50 | **11.70** | — | **+12.35** | **−1.75 ⚠️** |
| Gemma 2B → 4B, math FullFT | 13.45 | **26.70** | 100 | 13.90 | — | **+13.25** | +0.45 (≈ baseline) |
| Llama 1B → 8B, BFCL funcall | 55.30 | 59.00 | 150 | **32.00** | — | +3.70 | **−23.30 ⚠️** |
| Gemma 2B → 4B, BFCL funcall | 0.00 | 30.90 | 50 | 3.90 | — | +30.90 | +3.90 (≈ floor) |

<br>

**3 fullseq settings degrade *below baseline*** (Qwen seed 123 math, Gemma math LoRA, Llama funcall). **Zero positional settings do.** Worst-case asymmetry: positional's worst is +0.45pp; fullseq's worst is **−23.30pp**.

<!--
[~2 min]
Per-step coding trajectory (left): the cleanest visual — fullseq monotonically falls from
40.2 (step 50) to 26.8 (step 400), well below the 32.93% baseline. Positional (pos-50, pos-100, pos-150)
all stay in 37-42% range.

Per-step math (right): seed 42's fullseq trajectory peaks at step 150 (62.35) then dips at 200,
and crucially, on seed 123 the same training schedule lands at 50.45 — below baseline 50.95.
That's the seed-level reward-hack failure.

Cross-family table (bottom): Gemma math LoRA fullseq -1.75pp; Llama funcall fullseq -23.3pp.
Three of nine settings degrade below baseline under fullseq; zero under positional.
-->
