# §2 Entropy: necessary, but not sufficient

### Three entropy-based selectors all *match* full-seq — none exceed it

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### MATH-500 K=100 (avg@4)

| Selector | avg@4 | vs full-seq |
|----------|------:|------------:|
| Top-KL-100 | 58.60 | −3.75 |
| Top-ent-student-100 | 61.35 | −1.00 |
| Format-mask | 62.05 | −0.30 |
| Top-ent-teacher-100 | 62.20 | −0.15 |
| **Full-seq** | **62.35** | **0.00** |
| Random-100 | 63.05 | +0.70 |

All three entropy variants **land inside ±1pp of full-seq**.

`format-mask` (mask out low-entropy format tokens above a learned threshold)
also lands at full-seq — same indicator family.

### Pending: entropy-threshold sweep

`top-20% / top-56% / top-80%` of high-entropy tokens by absolute threshold
(H > 0.664 / 0.01 / 0.0004). *Running on UCLACG GPU 0.*

</div>
<div style="width: 50%;">

### Why entropy matches but doesn't exceed

Entropy ranks tokens by **local uncertainty**, which correlates with where the teacher has something to teach — but not perfectly:

- A high-entropy late token is on a trajectory that's **already broken**. Aligning it doesn't fix the answer.
- A high-entropy early token sets the trajectory. Aligning it fixes everything downstream.

Top-entropy selection weights both equally → ≈ full-seq.

<br>

### Random-100 ≈ full-seq

A uniform random K=100 also lands at full-seq. This is the diagnostic: **any non-pathological selector that isn't dominated by format/style noise lands here.**

The full-seq number is a **floor for serious indicators**, not a ceiling.

</div>
</div>

<!--
[~2 min] Second result. Entropy is *not wrong* the way KL is, but it's also not better.
Full-seq is what you get when you put a roughly-correct mass on a roughly-correct subset.
Need a different *kind* of indicator — not a per-token signal, but a structural one.
-->
