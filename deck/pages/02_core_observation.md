# Core Observation: Teacher Signal Quality Decays with Position

In on-policy KD, the teacher scores: $p_{\text{teacher}}(\cdot \mid x, y_{<t}^{\text{student}})$

As $t$ grows, the teacher is increasingly **conditioned on the student's prefix**.

<br>

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 45%;">

### Two regimes:

**Early positions** (t < 100)
- Teacher gives **independent judgment**
- "Which approach should we take?"
- High entropy, high disagreement

**Late positions** (t > 200)
- Teacher **rubber-stamps** student
- "Continue what you started"
- Low entropy, high agreement

</div>
<div style="width: 55%;">

### The evidence (Qwen 1.5B → 1.7B):

| Position | KL | Teacher Ent | Agreement |
|----------|-----|-----------|-----------|
| 0–50 | **1.91** | 0.26 | 75% |
| 50–100 | 0.94 | 0.21 | 82% |
| 100–200 | 0.65 | 0.16 | 86% |
| 300–500 | 0.43 | 0.17 | **89%** |

KL drops **4.4×**. Agreement rises from 75% to 89%.

</div>
</div>

<!--
[~2 min]
This is the core insight. The teacher is not uniformly useful.
Early = independent assessment. Late = rubber-stamping.
Three metrics all point the same way.
-->
