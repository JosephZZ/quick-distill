# §8 Student surpasses teacher (mode-seeking)

### Reverse-KL on a contiguous prefix → student commits to a high-conviction mode

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### BFCL function calling

| Pair | Teacher | Student raw | **Pos-K student** | Δ vs teacher |
|------|--------:|------------:|------------------:|-------------:|
| Qwen 1.5→1.7B | 54.0 | 2.7 | **61.3** (N=100) | **+7.3** |
| Gemma 2→4B | 25.0 | 0.0 | **30.9** (N=50) | **+5.9** |
| Llama 1→8B | 63.7 | 55.3 | **59.0** (N=150) | −4.7 |

**Two of three pairs: student exceeds the teacher.**

### Mechanism — token level

Reverse-KL $D_{\mathrm{KL}}(p_\theta \,\|\, p_T)$ punishes the student for putting mass on tokens the teacher doesn't support, **but does not punish concentrating** mass on a single supported token.

So the student commits to **the** correct mode of the teacher's distribution — even if that mode is *not the teacher's argmax*.

</div>
<div style="width: 50%;">

### Quantitative evidence — token level (n=100 trajectories, Qwen math)

From `signal_analysis/range_summary.json`:

| Position range | Teacher entropy | Teacher top-1 prob | T–S argmax agreement |
|----------------|----------------:|-------------------:|---------------------:|
| **0–50** (planning) | **0.259** | **0.909** | **0.785** |
| 50–100 | 0.235 | 0.917 | 0.837 |
| 100–150 | 0.181 | 0.936 | 0.876 |
| 150–200 | 0.171 | 0.940 | 0.895 |
| 200–300 | 0.155 | 0.944 | — |

**The teacher hedges most where it matters — at the planning prefix.** Teacher entropy at positions 0–50 is **52% higher** than at 150–200. The student-teacher argmax agreement gap is **11 percentage points wider** at planning vs. execution.

Reverse-KL $D_{\mathrm{KL}}(p_\theta \,\|\, p_T)$ on the prefix punishes the student for putting mass on tokens the teacher *doesn't* support, but does **not** punish concentrating on a single supported token. So the student commits to **the** correct mode of the teacher's distribution — even when that mode isn't the teacher's argmax.

→ This is why the student can *exceed* the teacher: it commits where the teacher hedges.

*(Pending follow-up: rank-of-correct-branch CDF — needs full-vocab logprob extraction beyond the chosen token. Queued.)*

</div>
</div>

<!--
[~2 min] The capstone. Mode-seeking reverse-KL on a contiguous prefix has a precise
quantitative meaning: the student commits to the teacher's correct (but non-modal) branch
at planning tokens. This is *why* student>teacher on funcall is possible.
-->
