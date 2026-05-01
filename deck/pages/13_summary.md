---
layout: center
---

# Summary

<br>

### One insight:
Teacher supervision quality has **positional structure** — it decays from independent judgment to passive confirmation.

### One method:
**Positional distillation** — loss on first N tokens only. One line of code.

### The results:

| | Math (3 families) | Funcall (3 families) | Stability (3 seeds) | Speed |
|--|---|---|---|---|
| **Positional** | Matches/exceeds 4/6 settings | Student **exceeds teacher +7.3pp** | All 3 seeds clear baseline (std **3.0** vs 6.0) | **12–24× faster** |
| Fullseq | Collapses | Degrades | **1/3 seeds degrades below baseline**; 2.0× higher std | — |

<br>

### The deeper finding:
High-KL tokens are **noise**, not signal. Position is a better proxy for supervision quality than any information-theoretic measure. The cascade effect means early-token improvements propagate through entire responses.

<br>

**Positional distillation is not an approximation — it's a principled focus on where teachers actually teach.**

<!--
[~1 min]
End with the one-sentence takeaway.
-->
