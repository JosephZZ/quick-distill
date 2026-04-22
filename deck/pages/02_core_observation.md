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

### The evidence:

<img src="/images/fig1_kl_decay.png" style="max-width: 100%; max-height: 55vh; object-fit: contain; margin: 0;" />

KL drops **4.4×**. Agreement rises from 75% to 89%.

</div>
</div>

<!--
[~2 min]
This is the core insight. The teacher is not uniformly useful.
Early = independent assessment. Late = rubber-stamping.
Three metrics all point the same way.
-->
