# The Information-Quality Paradox

### If prefix works, can we do even better with smarter token selection?

We compare 7+ strategies, all selecting K=100 tokens from full-length responses:

<br>

| Strategy | How it selects | KL Coverage | Best avg@4 | Stable? |
|----------|---------------|-------------|------------|---------|
| **prefix** | First 100 tokens | **45.6%** | **65.85%** | ✅ |
| ent-teacher | Top-100 by teacher entropy | 50.5% | 63.30% | ❌ degrades |
| ent-student | Top-100 by student entropy | 67.0% | 62.70% | partial |
| ent-and | Top-100 by teacher × student ent | 57.3% | 54.50% | ? |
| ent-or | Top-100 by teacher + student ent | 64.3% | 55.20% | ? |
| **top-KL** | Top-100 by KL divergence | **93.2%** | **53.35%** | ❌ |
| random | 100 random tokens | 21.1% | 61.25% | ❌ collapses |

<br>

### The paradox: **93% KL coverage → worst result. 46% coverage → best result.**

<img src="/images/fig2_token_paradox.png" style="max-width: 80%; max-height: 35vh; object-fit: contain; margin: 0 auto; display: block;" />

<!--
[~2 min]
This is the intellectual core. Every information-theoretic method loses to simple prefix.
The paradox inverts standard intuition from active learning / curriculum design.
-->
