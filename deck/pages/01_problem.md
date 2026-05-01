# On-Policy Distillation: Setup and the Problem

### The standard pipeline

Student samples a response → teacher scores **every token** → student trains on full-sequence reverse KL.

$$
\mathcal{L}_{\text{full}} = \sum_{t=1}^{T} D_{\text{KL}}\!\big(p_\theta(\cdot \mid x, y_{<t}) \,\|\, p_T(\cdot \mid x, y_{<t})\big)
$$

**Implicit assumption**: every position $t$ contributes useful supervision.

<br>

### But full-sequence training keeps breaking:

| Setting | What happens |
|---------|-------------|
| Qwen math (3 seeds, n1bs16) | Mean **56.78 ± 6.0**; seed 123 **degrades below baseline** (50.45% vs 50.95%) |
| Gemma math | Drops **below baseline** (13.45% → 11.70%) |
| Llama function calling | Degrades from 55.3% → **32.0%** |
| Qwen coding | HumanEval drops from 39% → 32% |

<br>

### Two questions

1. Why does training on **more** tokens make things **worse**?
2. Are all positions equally worth distilling — and if not, which ones matter?

<!--
[~1.5 min]
Setup the standard OPD pipeline + reverse-KL loss in one line.
Then immediately confront the failure mode: full-seq is unstable across families.
Pose the question: which tokens carry the supervision signal?
-->
