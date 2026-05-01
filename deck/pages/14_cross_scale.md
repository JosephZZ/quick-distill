# Cross-Scale Teacher: Larger Teachers Help More

### Same student (Qwen2.5-Math-1.5B). Sweep teacher size.

<br>

| Teacher | Baseline | Pos-100 best | Δ over baseline |
|---|---|---|---|
| Qwen3-**1.7B** | 50.95% | **65.85%** | +14.9pp |
| Qwen3-**4B** | 50.95% | **68.95%** | **+18.0pp** |
| Qwen3-**8B** | 50.95% | **67.85%** | +16.9pp |

<br>

### Findings:

- **Larger teacher → bigger student gain** at the same $N$ (saturates around 4–8B).
- **No need to re-tune $N$** — pos-100 stays near optimal across teacher scales.
- **First-100-token signal is what matters**, not teacher size per se. A bigger teacher gives a *cleaner* prefix signal, not a longer-range one.

<br>

This decouples scaling laws from positional structure: signal decay is a property of the **conditioning operation**, not the teacher's capacity.

<!--
Show the method holds as we scale the teacher — important for "is this useful at scale?" reviewer concern.
-->
