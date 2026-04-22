# What Do Early vs Late Tokens Encode?

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

### Early tokens = **Reasoning strategy**

> "Let's use the quadratic formula..."
> "First, decompose into prime factors..."
> "We can set up a system of equations..."

The teacher's **value-add** is here: which approach to take, how to frame the problem.

### Late tokens = **Execution + formatting**

> "...= $\frac{8\sqrt{2}}{2} = 4\sqrt{2}$"
> "$\boxed{32}$"

Teacher and student already **agree** on arithmetic.

</div>
<div style="width: 50%;">

### Token classification confirms this:

| Category | Mean KL | What it is |
|----------|---------|-----------|
| LaTeX format | **3.99** | \(, \[, \\\\ |
| Planning | **1.75** | "To", "First" |
| Structural | 0.89 | **, : |
| Math operators | 0.38 | =, +, − |
| Numbers | **0.28** | 0–9 |

<br>

**Highest KL = format disagreements**

Not reasoning quality — just how to *present* the answer.

Training on these teaches style, not substance.

</div>
</div>

<!--
[~1.5 min]
Concrete evidence of what early vs late tokens ARE.
High KL tokens are LaTeX formatting, not math reasoning.
-->
