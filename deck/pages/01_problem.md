# The Problem: Full-Sequence Distillation Is Fragile

On-policy KD: student generates → teacher scores every token → student trains on reverse KL.

**Standard assumption**: all token positions contribute equally useful supervision.

<br>

### But full-sequence training keeps breaking:

| Setting | What happens |
|---------|-------------|
| Qwen math (3 seeds) | 2/3 seeds **collapse** (65% → 38%) |
| Gemma math | Drops **below baseline** (13.45% → 11.70%) |
| Llama function calling | Degrades from 55.3% → **32.0%** |
| Qwen coding | HumanEval drops from 39% → 32% |

<br>

**Question**: Why does training on *more* tokens make things *worse*?

<!--
[~1.5 min]
Start with the failure mode. Everyone doing OPD has seen this — fullseq is unstable.
The natural assumption is all tokens are equal. We show this is wrong.
-->
