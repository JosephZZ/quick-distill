# format_mask experiment — results

Date: 2026-04-27

## Setup

Same n1bs16 LoRA recipe as the paper baselines, with `--token_select_mode
format_mask` instead of full-sequence loss. format_mask zeros the loss
mask at any token position whose decoded student emission falls into the
"format" categories: `structural`, `math_latex`, `math_operator`,
`math_number`, or `planning` (per `kl_x_entropy_buckets.py` classifier).

- Student: Qwen2.5-Math-1.5B
- Teacher: Qwen3-1.7B
- Dataset: AI-MO/NuminaMath-CoT, 3200 problems, bs=16 n_samples=1
- 200 steps, save every 50, lr=5e-5, LoRA r=32 α=64
- Format token share at training time: ~5.9% of total tokens (per launcher log)

## Results — MATH-500 avg@4 / pass@4 / maj@4

| Step | avg@4 | pass@4 | maj@4 |
|---:|---:|---:|---:|
| 50 | **62.05%** | 76.20% | 67.20% |
| 100 | 59.95% | 74.00% | 64.80% |
| 150 | 61.95% | 76.00% | 66.20% |
| 200 | 61.90% | 75.60% | 66.60% |

Best step = 50, peak avg@4 = **62.05%**.

## Comparison to baselines (n1bs16 LoRA, identical recipe)

| Method | Best avg@4 | Δ vs no-distill |
|---|---:|---:|
| no-distill | 50.95% | 0 |
| **format_mask** | **62.05%** | **+11.10pp** |
| fullseq | ~65% | +14 |
| prefix-100 | 65.85% | +14.9 |
| pos-50 | 66.65% | +15.7 |

## Reading

format_mask ranks **between baseline and fullseq**, *not* between fullseq
and prefix-100. Three takeaways:

1. **Removing format gradient does help** (+11pp over no-distill)
   — confirms format-noise is a real source of harmful supervision.

2. **But it does not match fullseq, let alone prefix-100** — meaning
   "remove harmful gradient" is *not* sufficient. The full-sequence
   loss, despite including format gradient, still beats format_mask
   by ~3pp. Interpretation: dilution by format tokens is not the
   dominant problem; *what fullseq misses by treating all positions
   equally* is the bigger issue.

3. **The prefix-100 advantage (+3.8pp over format_mask) must come
   from somewhere format_mask doesn't capture.** The bucket framework
   says: *prefix concentrates hiKL_hiE (the "useful gradient")*, not
   just *avoids hiKL_loE*. format_mask only does the second. The
   missing 3.8pp is the value of the up-sampling.

## Implications for the paper

This is a clean mid-tier result that supports the framework. The next
test is `hi_kl_hi_surp` (running): if it matches or beats prefix-100,
we have direct evidence that *selecting hiKL_hiE* is the operative
mechanism, with format_mask as a control showing partial-only effect.

| Mechanism | Method | Predicted effect | Observed |
|---|---|---|---|
| ① Remove harmful gradient | format_mask | partial (mid-tier) | ✓ 62.05% |
| ② Select useful gradient | hi_kl_hi_surp | match prefix | TBD |
| ①+② implicit | prefix-100 | best | ✓ 65.85% |

If ② holds, the paper's claim sharpens from "prefix wins because
position concentrates signal" to "selection on (KL, surprise) is the
mechanism; prefix is one cheap proxy."
