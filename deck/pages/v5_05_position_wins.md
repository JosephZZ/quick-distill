# §3 Position uniquely *exceeds* full-seq

### Prefix-K — the simplest possible selector — is the only one that beats full-seq

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 55%;">

### MATH-500, K=100, n1bs16 LoRA (avg@4)

| Selector | avg@4 | Δ vs full-seq |
|----------|------:|--------------:|
| No-distill baseline | 50.95 | −11.40 |
| Top-KL-100 | 58.60 | −3.75 |
| Top-ent-student-100 | 61.35 | −1.00 |
| Format-mask | 62.05 | −0.30 |
| Top-ent-teacher-100 | 62.20 | −0.15 |
| Full-seq | 62.35 | 0.00 |
| Random-100 | 63.05 | +0.70 |
| **Pos-100** | **65.85** | **+3.50** |
| **Pos-50 / Pos-150** | **66.65** | **+4.30** |

</div>
<div style="width: 45%;">

### Position is not an entropy proxy

If "first 100 tokens" were just a coarse high-entropy filter, top-entropy-100 would match prefix-100. It doesn't (61.35 vs 65.85, **−4.50pp**).

Quartile evidence (head-N tokens × surprise quartile):

| Head N | Q1 share | Q1 enrichment |
|-------:|---------:|--------------:|
| 50 | 61.0% | 2.4× |
| 100 | 53.2% | 2.1× |
| 200 | 42.6% | 1.7× |

Head-100 covers only **38.4% of >p95 surprise**. To cover 95% you'd need the first 846 positions.

**Position is correlated with entropy, not equivalent to it.**

</div>
</div>

<br>

### Why position wins where entropy ties

Entropy / KL / format-mask all select **scattered** tokens. Prefix-K selects a **contiguous prefix**. Two distinct effects:

1. **Causal coverage.** Aligning the planning prefix re-orients the entire trajectory; aligning scattered late tokens doesn't.
2. **Distribution-shape coupling.** A contiguous prefix matches the autoregressive structure of the model — the loss gradient lines up with how the network actually produces the next token, given the previously produced ones.

> Position isn't a *signal*; it's a *shape*. That's why it dominates.

<img src="/images/fig3_signal_indicators.png" style="max-width: 70%; max-height: 30vh; object-fit: contain; margin: 0 auto; display: block;" />

<!--
[~2 min] Punchline of Part 1.
Position is the only indicator that EXCEEDS full-seq.
Two reasons given: causal coverage (next slide will dig in) + contiguous-prefix shape.
This naturally motivates Part 2 (position works broadly) and Part 3 (here's why).
-->
