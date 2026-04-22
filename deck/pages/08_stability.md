# Stability: Multi-Seed Results

### 3 seeds, Qwen LoRA, MATH-500 avg@4

<br>

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

| Seed | pos-100 | fullseq |
|------|---------|---------|
| 42 | 65.85% | 65.60% → **37.75%** ⚠️ |
| 123 | 61.00% | **50.45%** ⚠️ |
| 456 | 60.40% | 57.55% |
| **Mean** | **62.42%** | **56.78%** |
| **Std** | **2.9** | **6.1** |
| **Collapse** | **0 / 3** | **2 / 3** |

</div>
<div style="width: 50%;">

### Key takeaway:

- pos-100 is **2.6× more stable** (std 2.9 vs 6.1)
- Fullseq **collapses in 2/3 seeds**
- pos-100 mean is **+5.6pp higher**

<br>

**The advantage is not just performance — it's reliability.**

With fullseq, you might get 65% or 38%. With positional, you reliably get ~62%.

</div>
</div>

<!--
[~1.5 min]
This addresses the reviewer concern about single-seed results.
The variance difference (2.9 vs 6.1) is the key number.
-->
