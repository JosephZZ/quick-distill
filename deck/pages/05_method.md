---
layout: center
---

# Method: Positional Distillation

### One line of code.

<br>

$$
\mathcal{L}_{\text{full}} = \sum_{t=1}^{T} D_{\text{KL}}(q \| p) \quad \longrightarrow \quad \mathcal{L}_{\text{pos}} = \sum_{t=1}^{N} D_{\text{KL}}(q \| p)
$$

<br>

```python
loss_mask[:, N:] = 0  # That's it.
```

<br>

| | Full-seq (T≈3584) | Pos-100 | Gain |
|--|-------------------|---------|------|
| **Time / step** | 133s | 11s | **12× faster** |
| **200 steps** | 7.4 hours | 37 min | **12× faster** |
| **GPU memory** | 17 GB | 9 GB | **1.9× less** |

<br>

Not an approximation — **a principled focus on the highest-quality signal.**

<!--
[~1 min]
The method is trivially simple. That's the point.
The contribution is understanding WHY this works.
-->
