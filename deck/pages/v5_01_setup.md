# On-Policy Distillation: Setup and the Question

### The standard pipeline

Student samples response → teacher scores **every token** → student trains on full-sequence reverse KL.

$$
\mathcal{L}_{\text{full}} = \sum_{t=1}^{T} D_{\text{KL}}\!\big(p_\theta(\cdot \mid x) \,\|\, p_T(\cdot \mid x)\big)_t
$$

**Implicit assumption**: every position $t$ contributes useful supervision.

<br>

### But full-sequence training keeps breaking:

| Setting | What happens |
|---------|-------------|
| Qwen math (3 seeds, n1bs16) | Mean **56.78 ± 6.0**; one seed **drops below baseline** (50.45 vs 50.95) |
| Gemma math | **13.45% → 11.70%** (below baseline) |
| Llama function calling | **55.3% → 32.0%** |
| Qwen coding | HumanEval **39% → 32%** |

<br>

### The question this paper answers

> **Among all generated tokens, which ones actually carry the distillation signal?**

Three plausible answers: high-**KL** tokens, high-**entropy** tokens, or **early-position** tokens. We test all three.

<!--
[~1.5 min] Setup OPD reverse KL loss + the failure mode that motivates the question.
The question itself frames the entire paper as a signal-indicator selection study.
-->
