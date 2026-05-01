# Concurrent Work and Our Delta

Prefix truncation as an empirical recipe is **not new** — we explain *why* and validate breadth.

<br>

| | **Chen et al. 2025** | **Li et al. 2025** | **This work** |
|--|---|---|---|
| Setting | Off-policy KD | RL (PPO/GRPO) | **On-policy KD** |
| *Why* it works | Empirical only | Reward decay (RL) | **Signal decay + paradox** |
| Token-selection baselines | — | — | **7+ strategies** |
| Info.-quality paradox | — | — | ✅ |
| Heuristic for $N$ | — | — | **45% rule (descriptive)** |
| Family / task / scale validation | 1 / 1 / — | 1 / 1 / — | **3 / 3 / 3** |

<br>

### Our framing:

We take the prefix **as a known empirical recipe**, not a new discovery. Our contribution is:

1. **Empirical evidence consistent with *why*** the prefix is the right choice in on-policy KD (signal decay + paradox).
2. **Demonstrate prefix is *better*, not merely faster**, than full-seq in on-policy KD.
3. **Validate breadth**: 3 families × 3 tasks × 3 teacher scales — and report where it underperforms.

<!--
Address the obvious "isn't this just prefix truncation" reviewer objection up front.
-->
