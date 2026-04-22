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
| **Positional** | Wins 5/6 settings | Student **exceeds teacher** | 0/3 collapse | **12× faster** |
| Fullseq | Collapses | Degrades | 2/3 collapse | — |

<br>

### The deeper finding:
High-KL tokens are **noise**, not signal. Position is a better proxy for supervision quality than any information-theoretic measure. The cascade effect means early-token improvements propagate through entire responses.

<br>

**Positional distillation is not an approximation — it's a principled focus on where teachers actually teach.**

<!--
[~1 min]
End with the one-sentence takeaway.
-->
