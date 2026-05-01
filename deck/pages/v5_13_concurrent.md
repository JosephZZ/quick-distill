# Concurrent work

### Where this paper sits

| Direction | Examples | Difference |
|-----------|----------|-----------|
| Token-importance reweighting | TIP (2025), entropy-weighted KD (2024), thoughts-to-tokens reweighting | All select by **per-token signal magnitude**. We show position dominates *all* per-token signals. |
| Curriculum / on-policy distillation | Qwen3 distillation report, mini-on-policy-RL distillation | Same OPD pipeline, no token-selection lens. |
| Sequence-level KD | Sequence-level KD (Kim & Rush 2016), MiniLLM | Use a single sequence-level loss; we ablate *which* tokens carry the gradient. |

### What's specifically novel here

1. **Position as the indicator, not a heuristic.** A first-class comparison against KL- and entropy-based selectors at fixed K.
2. **Negative controls** (middle / last) showing late-token training is *worse than no distillation*.
3. **Cascade evidence**: training the prefix automatically aligns the tail.
4. **Token-level quantification** of why student can surpass teacher under mode-seeking reverse-KL.

<!--
[~1 min] Briefly position concurrent work. We're not arguing nobody else has thought
about token selection — we're arguing nobody has compared selectors head-to-head
at fixed budget against full-seq with this pipeline.
-->
