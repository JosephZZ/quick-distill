# Stability: Multi-Seed Results

### 3 seeds, Qwen LoRA n1bs16, MATH-500 avg@4 at step 200 (last-boxed)

<br>

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

| Seed | pos-100 | fullseq |
|------|---------|---------|
| 42 | **65.85%** | 62.35% |
| 123 | 61.00% | **50.45%** ⚠️ **below baseline** |
| 456 | 60.40% | 57.55% |
| **Mean** | **62.42%** | **56.78%** |
| **Std** | **3.0** | **6.0** |
| **Δ baseline (50.95)** | +11.5 ± 3.0 | +5.8 ± 6.0 |

</div>
<div style="width: 50%;">

### Key takeaway:

- pos-100 std is **2.0× tighter** (3.0 vs 6.0)
- pos-100 mean is **+5.6pp higher** than fullseq
- **pos-100 wins on every seed**
- Fullseq seed 123 = **50.45% < 50.95% baseline** — **200 steps of training made it worse than no training** (`\boxed{}` repetition mode)

<br>

**The advantage is not just average performance — it's reliability.**

With fullseq, **1 of 3 seeds actively degrades below baseline** — 200 steps made it worse than no training. With positional, every seed clears baseline by ≥9pp.

</div>
</div>

<!--
[~1.5 min]
3-seed n1bs16 stability test, last-boxed extraction (canonical metric).
The variance difference (3.0 vs 6.0; 2.0× tighter) is the headline.
Seed 123 fullseq = 50.45% essentially equals baseline (50.95%) —
that IS the reward-hack failure mode (boxed-rep), and we report it
under the canonical metric, not a "fixed" extraction.
-->
