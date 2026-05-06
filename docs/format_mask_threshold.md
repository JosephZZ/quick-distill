# Format-Mask Threshold Analysis

Data: `/zhi_backup/ziheng/quick-distillation/docs/kl_position_analysis_v2/raw_logprobs.jsonl`
Trajectories: 100, total tokens: 74549, total KL: 33266.2

## Per-category share

| Category | n_tokens | tok % | KL sum | KL % | mean KL/tok |
|---|---:|---:|---:|---:|---:|
| continuation | 26122 | 35.0% | 14628.3 | 44.0% | 0.560 |
| structural | 24725 | 33.2% | 6499.1 | 19.5% | 0.263 |
| planning | 4747 | 6.4% | 4585.2 | 13.8% | 0.966 |
| math_latex | 3206 | 4.3% | 4305.5 | 12.9% | 1.343 |
| math_number | 10528 | 14.1% | 2463.3 | 7.4% | 0.234 |
| math_operator | 5221 | 7.0% | 784.9 | 2.4% | 0.150 |

## Cumulative mask coverage (add categories in this order)

| Mask = | Masked tokens | Masked tok % | Masked KL | Masked KL % | Remaining KL % |
|---|---:|---:|---:|---:|---:|
| {structural} | 24725 | 33.2% | 6499.1 | 19.5% | 80.5% |
| {structural, math_latex} | 27931 | 37.5% | 10804.6 | 32.5% | 67.5% |
| {structural, math_latex, math_operator} | 33152 | 44.5% | 11589.5 | 34.8% | 65.2% |
| {structural, math_latex, math_operator, math_number} | 43680 | 58.6% | 14052.7 | 42.2% | 57.8% |
| {structural, math_latex, math_operator, math_number, planning} | 48427 | 65.0% | 18637.9 | 56.0% | 44.0% |
| {structural, math_latex, math_operator, math_number, planning, continuation} | 74549 | 100.0% | 33266.2 | 100.0% | 0.0% |

## Recommendation guides

- Aggressive mask (structural+latex+ops+numbers): captures most format-y tokens but may over-remove signal
- Conservative mask (structural+latex only): keeps numerical/operator content
- Pick the row whose masked-KL % matches the fraction of signal the user wants removed.

If the prefix-vs-fullseq gap is mostly about format noise, a mask that
removes ~50-80% of KL but leaves reasoning content should *match or exceed*
fullseq performance. If it does not, format-noise is not the operative variable.