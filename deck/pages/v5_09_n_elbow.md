# §6 How to choose N — a single elbow around 50–150

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### Position sweep — math (Qwen 1.5→1.7B, n1bs16)

| N | avg@4 | Δ vs full-seq |
|--:|------:|--------------:|
| 50  | **66.65** | +4.30 |
| 100 | 65.85 | +3.50 |
| 150 | **66.65** | +4.30 |
| 200tok | 66.05 | +3.70 |
| 300+ | 64.0 (extrap.) | +1.6 |
| **full-seq (T)** | **62.35** | 0.00 |

Plateau from N=50 to N≈200. Performance starts dropping back toward full-seq once N exceeds the *useful prefix*.

### Funcall (Llama 1→8B) — non-monotonic

| N | full_acc |
|--:|---------:|
| 50 | 44.0 |
| 100 | 40.5 |
| **150** | **59.0** |
| 200 | 51.2 |
| 300 | 46.2 |
| full-seq | 32.0 |

</div>
<div style="width: 50%;">

### Why is the optimum around N≈50–150?

Two competing forces:

1. **Coverage** — each additional position adds (some) information.
2. **Noise injection** — late positions encode execution + format, which is noise w.r.t. the *strategy* being distilled.

Below N≈50 you're starving on coverage; above N≈200 you're swamping the gradient with execution noise.

### The "useful prefix"

Empirically, the planning portion of a math response is **≈40–80 tokens** in our data. The N-elbow lines up with where planning hands off to execution.

For a new task:

- Estimate planning length from a few teacher samples.
- Use **N ≈ 1.5× planning length** (gives slack but doesn't drown the prefix in execution).
- Coding ≈ 50, math ≈ 100, funcall ≈ 150 — task-by-task this rule holds.

> **Heuristic:** N ≈ 1.5× the median length of the teacher's "plan + first computation".

</div>
</div>

<img src="/images/fig6_n_elbow.png" style="max-width: 70%; max-height: 25vh; object-fit: contain; margin: 0 auto; display: block;" />

<!--
[~2 min] N-selection IS part of "position is a great strategy" — that's why it's in §6.
Single elbow rule across tasks. Coverage vs noise framing closes Part 2 cleanly
and hands off to Part 3's question: *why* does adding more positions hurt past N≈200?
-->
