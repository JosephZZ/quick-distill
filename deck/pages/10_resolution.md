# Position Is Not Reducible to High Entropy

### "Is prefix just a cheap proxy for high-entropy selection?"

If yes, top-entropy-100 should match prefix-100. It doesn't (62.7% vs 65.85%).
The data show why — head and entropy correlate, but **position carries information beyond entropy**.

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### Surprise quartile distribution among first $N$ tokens

Baseline is 25% per quartile (uniform).

| Head $N$ | Q1 (top-25%) | Q4 (bottom-25%) | Q1 enrichment |
|:---:|:---:|:---:|:---:|
| 25  | **70.0%** | 7.4% | $2.8\times$ |
| 50  | **61.0%** | 13.1% | $2.4\times$ |
| 100 | 53.2% | 17.3% | $2.1\times$ |
| 200 | 42.6% | 23.1% | $1.7\times$ |
| 500 | 31.8% | 27.1% | $1.3\times$ |

**Head is enriched in high entropy — but not exclusively.** First 100 tokens still contain 17% Q4 (low-surprise) tokens.

</div>
<div style="width: 50%;">

### What head-100 covers, what it misses

| Quantity | First 100 |
|---|:---:|
| Tokens covered | 13% of total |
| High-surprise mass ($>p_{75}$) covered | 28.5% |
| Top-5% surprise covered | **38.4%** |
| Cumulative $\sum$surprise covered | 40.2% |

**Prefix-100 misses 60%+ of high-surprise mass.** Yet it still beats top-entropy-100, top-KL-100, random-100.

To cover 95% of $>p_{95}$ surprise tokens, you need the first **846 positions** — basically the whole response.

→ Position and entropy are **correlated, not identical**.
→ Top-entropy selection puts equal weight on a high-surprise late token (which lives on an already-broken trajectory) and a high-surprise early token (which sets the trajectory). Position discounts the former.

</div>
</div>

<br>

### Reading: position is a *causal* proxy, not an entropy proxy

Entropy ranks tokens by **local uncertainty**. Position ranks them by **causal influence on the rest of the generation**. The cascade slide showed why the latter dominates: aligning the planner auto-aligns the executor; aligning the executor doesn't auto-align the planner.

<!--
[~2 min]
This is the heart of Sale #2. Position vs entropy is THE conceptual contrast.
Quartile table makes "head ≠ high entropy" concrete with numbers.
Right column shows even high-entropy mass is too dispersed for entropy ranking to win.
-->
