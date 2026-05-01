# Whoever Writes the First 100 Tokens, Owns the Trajectory

### Test-time prefix swap on MATH-500 (n=4, T=0.7)

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### Student writes prefix → teacher continues

| Prefix length | avg@4 | vs. teacher solo |
|:---:|:---:|:---:|
| Teacher solo (no prefix) | **65.30%** | — |
| 100 tok | 62.70% | −2.6 |
| 200 tok | 56.30% | −9.0 |
| 300 tok | **51.75%** | **−13.6** |
| Student solo | 50.95% | −14.4 |

By 300 student tokens, the teacher is dragged all the way down to **student baseline**.

</div>
<div style="width: 50%;">

### Teacher writes prefix → student continues

| Prefix length | avg@4 | vs. student solo |
|:---:|:---:|:---:|
| Student solo | 50.95% | — |
| 100 tok | 47.00% | **−3.95** |
| 200 tok | 47.45% | −3.5 |
| 300 tok | 52.65% | +1.7 |

A short teacher prefix **hurts** the student (style mismatch). Only at 300 tok does the teacher's setup carry it past baseline.

</div>
</div>

<br>

### Two findings, one conclusion

1. **The prefix sets the trajectory's ceiling.** A 300-token student preamble drops a strong teacher to baseline; a 100-token teacher preamble cannot lift a student.
2. **Performance is dominated by the *continuator*, not the prefix giver** — except when the prefix is *long enough* to commit to a wrong path.

→ The early 100–200 tokens are where reasoning is *decided*, not described.
→ This is precisely the window positional distillation targets.

<!--
[~2 min]
This is the strongest causal evidence for "head tokens carry causal weight".
Same teacher, same student, same problems — only the boundary moves.
The performance curve traces out the cascade.
-->
