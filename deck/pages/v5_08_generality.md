# §5 Generality: cross-model, cross-task, cross-method

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 33%;">

### Cross-family math

| Pair | Pos best | Full-seq |
|------|---------:|---------:|
| Qwen 1.5→1.7B | **66.65** (N=50) | 62.35 |
| Gemma 2→4B | **30.9** (N=50) | 3.9* |
| Llama 1→8B | **59.0** (N=150) | 32.0* |

\* full-seq collapses or degrades

### Cross-task (Qwen 1.5→1.7B)

| Task | Pos best | Full-seq |
|------|---------:|---------:|
| Math (MATH-500) | **66.65** | 62.35 |
| Code (HE) | **42.1** | 40.2→26.8 |
| Funcall (BFCL) | **61.3** | 58.2 |

</div>
<div style="width: 33%;">

### Cross-method

| Method | Pos best | Full-seq |
|--------|---------:|---------:|
| LoRA Math | **66.65** | 62.35 |
| FullFT Math | 56.75 | 58.20 |
| LoRA Code (HE) | **42.1** | 40.2→26.8 |
| FullFT Code (HE) | 37.8 | 36.6 |
| FullFT Code (MBPP+) | 47.4 | 46.3 |

LoRA + position = strictly better than LoRA fullseq.
FullFT softens the gap, but pos-K never *loses* on coding.

### CoT (thinking) mode

Qwen3-1.7B → Qwen3-4B with `enable_thinking=True`:

| Method | avg@4 |
|--------|------:|
| **Pos-100** | **66.80** |
| Full-seq | 64.65 |

Position wins **even in thinking mode** (+2.15pp).

</div>
<div style="width: 33%;">

### Stability (3 seeds, Qwen math)

| Method | Mean ± std | Worst seed |
|--------|-----------:|-----------:|
| Full-seq | 56.78 ± 6.0 | **50.45** (below baseline) |
| **Pos-100** | **62.42 ± 3.0** | 60.40 |

Pos-100 std is **2.0× tighter** than full-seq, and its worst seed beats full-seq's mean by 3.6 pp.

### Efficiency

| Quantity | Full-seq | Pos-100 | Ratio |
|----------|---------:|--------:|------:|
| Wall-clock / 200 steps | ~9.5 h | ~1.0 h | **9.5×** |
| Peak GPU memory | ~38 GB | ~9 GB | **4×** |
| Code change | — | 1 line | — |

Same hyper-parameters. No tuning per task.

</div>
</div>

<!--
[~2.5 min] The 好快稳 sweep. Three columns:
  Cross-family / Cross-task / Cross-method (好)
  CoT thinking mode generalization
  Stability + efficiency (稳 + 快)
Key callout: full-seq's stability/speed problems are not minor — they're load-bearing.
-->
