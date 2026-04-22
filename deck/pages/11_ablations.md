# Ablations: Position Sweep + CoT + LoRA vs FullFT

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 33%;">

### Position Sweep (Qwen Math)

| N | avg@4 |
|---|-------|
| 5 | 56.50 |
| 10 | 59.50 |
| 20 | 60.20 |
| 50 | 62.45 |
| **100** | **64.25** |
| 150 | 65.70 |
| **200** | **66.75** |
| fullseq | 62→38 ⚠️ |

Diminishing returns past N=100. Fullseq collapses.

</div>
<div style="width: 33%;">

### CoT Thinking Mode

Qwen3-1.7B → Qwen3-4B with `enable_thinking=True`:

| Method | avg@4 |
|--------|-------|
| **pos-100** | **66.80%** |
| fullseq | 64.65% |

**Positional wins even in thinking mode** (+2.15pp)

### Funcall Position Sweep

Llama funcall (non-monotonic!):

| N | full_acc |
|---|---------|
| 50 | 44.0% |
| 100 | 40.5% |
| **150** | **59.0%** |
| 200 | 51.2% |
| 300 | 46.2% |
| fullseq | 32.0% |

</div>
<div style="width: 33%;">

### LoRA vs FullFT

| Method | Qwen Math |
|--------|-----------|
| LoRA pos-100 | **65.85%** |
| LoRA fullseq | 62→38% |
| FullFT pos-100 | 53.15% |
| FullFT fullseq | 57.10% |

**LoRA + positional = double regularization**

LoRA constrains *parameters*. Position constrains *signal*. Together: best of both.

</div>
</div>

<!--
[~2 min]
Three ablations. Position sweep shows diminishing returns.
CoT mode confirms generality. LoRA + positional is complementary.
-->
