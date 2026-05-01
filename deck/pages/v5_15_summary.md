# Summary

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### What we asked

> Among all generated tokens, **which ones** actually carry the distillation signal?

### What we tested

Three families of signal indicators at fixed budget K=100:
- **KL** — top-K by per-token KL divergence
- **Entropy** — top-K student / top-K teacher / format-mask
- **Position** — first K tokens (prefix-K)

### What we found

| Indicator | vs full-seq |
|-----------|------------:|
| Top-KL | **−3.75** (worst) |
| Entropy variants | ±1pp (≈ tie) |
| **Position** | **+3.5 to +4.3** (only one that exceeds) |

</div>
<div style="width: 50%;">

### Why position wins

- **Causal coverage**: the prefix sets the trajectory; aligning it auto-aligns the tail.
- **Contiguous-prefix shape**: matches the autoregressive structure of the model.
- **Mode-seeking** on a contiguous prefix: student can commit to the correct (but non-modal) teacher branch → **student surpasses teacher** on funcall (+5–7pp).

### Cost

- 1-line code change.
- ~10× wall-clock, ~4× memory.
- No new hyper-parameters (single elbow, N ≈ 50–150 across tasks).

### Default recipe

> **OPD + LoRA + prefix-100** is now the default we'd recommend for any small-to-mid-scale on-policy distillation run.

</div>
</div>

<br>

> Position isn't *a signal*. It's *a shape*. Pick the right shape and the rest follows.

<!--
[~1.5 min] Wrap up. Three numbers, two mechanisms, one default recipe.
"Shape, not signal" is the line we want them to leave with.
-->
