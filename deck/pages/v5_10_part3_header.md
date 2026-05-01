# Part 3 — Why is position so good?

<br>

### Two complementary mechanisms

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### A. Cascading-error theory

> Aligning the **planner** auto-aligns the **executor**.
> Aligning the executor doesn't auto-align the planner.

Three pieces of evidence:
- Test-time prefix swap
- Auto tail-KL drop without tail loss
- Conditional-drift bound (intuition)

</div>
<div style="width: 50%;">

### B. Mode-seeking + student surpasses teacher

> Reverse-KL is mode-seeking; on a contiguous *prefix* it commits the student to a single high-conviction trajectory mode.

NEW token-level evidence:
- Rank-of-correct-branch in teacher distribution
- Top-3 mass at planning vs execution tokens
- Quantitative form of "the teacher knows the right answer is in there, but doesn't peak on it"

</div>
</div>

<br>

> Part 3 is **discussion**: empirical regularities + a narrative that lines up with them.
> The headline result (Part 2) does not depend on this section being right.

<!--
[~30s] Section header for Part 3.
Important framing: the cascading + mode-seeking story is *interpretation*, not proof.
The empirical results in Part 2 stand independently.
-->
