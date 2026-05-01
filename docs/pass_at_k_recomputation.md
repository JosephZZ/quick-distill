# MATH-500 pass@k / maj@k / avg@k Recomputation (UCLACG)

Recomputed from raw `results.jsonl` files on UCLACG (128 eval directories).
All numbers are percentages on the MATH-500 test set (500 problems).
`avg`/`maj`/`pass` are per-problem mean / majority-vote / any-correct over the n_samples generations.

- Total eval dirs scanned: **128**
- With n_samples >= 4 (full pass@4 recoverable): **108**
- With n_samples == 1 (only avg=pass=maj): **20**
- Incomplete (total != 500): **0**

Schema note: results.jsonl uses `responses` (not `samples`) as the per-problem list of generations.
The `summary.json` `accuracy` field is the fraction of *problems* with at least one correct sample (i.e., pass@k), and `avg_accuracy` is the per-sample average.

## eval_results/llama31-8b-baseline

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `eval_results/llama31-8b-baseline` | 4 | 500 | 37.25 | 46.80 | 58.60 | acc=58.60, avg=37.25, maj=46.80, pass=58.60 | OK |

## eval_results/llama32-1b-baseline

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `eval_results/llama32-1b-baseline` | 4 | 500 | 15.20 | 20.40 | 32.60 | acc=32.60, avg=15.20, maj=20.40, pass=32.60 | OK |

## eval_results/llama32-3b-baseline

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `eval_results/llama32-3b-baseline` | 4 | 500 | 19.80 | 30.80 | 38.80 | acc=38.80, avg=19.80, maj=30.80, pass=38.80 | OK |

## format-mask-1.7b-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/format-mask-1.7b-math/step_100/eval` | 4 | 500 | 59.95 | 64.80 | 74.00 | acc=74.00, avg=59.95, maj=64.80, pass=74.00 | OK |
| `checkpoints/format-mask-1.7b-math/step_150/eval` | 4 | 500 | 61.95 | 66.20 | 76.00 | acc=76.00, avg=61.95, maj=66.20, pass=76.00 | OK |
| `checkpoints/format-mask-1.7b-math/step_200/eval` | 4 | 500 | 61.90 | 66.60 | 75.60 | acc=75.60, avg=61.90, maj=66.60, pass=75.60 | OK |
| `checkpoints/format-mask-1.7b-math/step_50/eval` | 4 | 500 | 62.05 | 67.20 | 76.20 | acc=76.20, avg=62.05, maj=67.20, pass=76.20 | OK |

## gemma-pos50-coding-retrain

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/gemma-pos50-coding-retrain/eval_step_200` | 4 | 500 | 23.15 | 26.20 | 35.60 | acc=35.60, avg=23.15, maj=26.40, pass=35.60 | MISMATCH: maj 26.40!=26.20 |

## gemma-pos50-math-retrain

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/gemma-pos50-math-retrain/eval_final` | 4 | 500 | 24.05 | 27.00 | 35.80 | acc=35.80, avg=24.05, maj=27.20, pass=35.80 | MISMATCH: maj 27.20!=27.00 |
| `checkpoints/gemma-pos50-math-retrain/eval_step_200` | 4 | 500 | 23.95 | 27.00 | 36.40 | acc=36.40, avg=23.95, maj=27.00, pass=36.40 | OK |

## hi-kl-hi-surp-1.7b-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/hi-kl-hi-surp-1.7b-math/eval_final` | 4 | 500 | 55.60 | 65.00 | 73.00 | acc=73.00, avg=55.60, maj=65.00, pass=73.00 | OK |
| `checkpoints/hi-kl-hi-surp-1.7b-math/step_100/eval` | 4 | 500 | 47.75 | 65.60 | 72.20 | acc=72.20, avg=47.75, maj=65.60, pass=72.20 | OK |
| `checkpoints/hi-kl-hi-surp-1.7b-math/step_150/eval` | 4 | 500 | 53.95 | 65.60 | 74.80 | acc=74.80, avg=53.95, maj=65.60, pass=74.80 | OK |
| `checkpoints/hi-kl-hi-surp-1.7b-math/step_200/eval` | 4 | 500 | 54.30 | 66.00 | 71.40 | acc=71.40, avg=54.30, maj=66.00, pass=71.40 | OK |
| `checkpoints/hi-kl-hi-surp-1.7b-math/step_50/eval` | 4 | 500 | 39.00 | 62.40 | 69.00 | acc=69.00, avg=39.00, maj=62.40, pass=69.00 | OK |

## hi-kl-hi-surp-topk100-1.7b-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/hi-kl-hi-surp-topk100-1.7b-math/eval_final` | 4 | 500 | 56.75 | 65.40 | 73.60 | acc=73.60, avg=56.75, maj=65.40, pass=73.60 | OK |

## scale-gemma2-2b-tgemma3-4b-math-fullseq

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq/eval_step_100` | 4 | 500 | 11.70 | 18.40 | 24.60 | acc=24.60, avg=11.70, maj=18.20, pass=24.60 | MISMATCH: maj 18.20!=18.40 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq/eval_step_150` | 4 | 500 | 11.70 | 18.40 | 24.60 | acc=24.60, avg=11.70, maj=18.20, pass=24.60 | MISMATCH: maj 18.20!=18.40 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq/eval_step_200` | 4 | 500 | 11.70 | 18.40 | 24.60 | acc=24.60, avg=11.70, maj=18.20, pass=24.60 | MISMATCH: maj 18.20!=18.40 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq/eval_step_50` | 4 | 500 | 11.70 | 18.40 | 24.60 | acc=24.60, avg=11.70, maj=18.20, pass=24.60 | MISMATCH: maj 18.20!=18.40 |

## scale-gemma2-2b-tgemma3-4b-math-pos50

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-pos50/eval_step_100` | 4 | 500 | 7.20 | 15.80 | 18.00 | acc=18.00, avg=7.20, maj=16.00, pass=18.00 | MISMATCH: maj 16.00!=15.80 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-pos50/eval_step_150` | 4 | 500 | 6.30 | 13.00 | 16.00 | acc=16.00, avg=6.30, maj=13.00, pass=16.00 | OK |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-pos50/eval_step_200` | 4 | 500 | 6.55 | 13.20 | 15.80 | acc=15.80, avg=6.55, maj=13.20, pass=15.80 | OK |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-pos50/eval_step_50` | 4 | 500 | 6.30 | 13.40 | 15.00 | acc=15.00, avg=6.30, maj=13.40, pass=15.00 | OK |

## topent-student-k200-1.7b-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/topent-student-k200-1.7b-math/eval_step_100` | 4 | 500 | 43.65 | 64.60 | 73.00 | acc=73.00, avg=43.65, maj=64.60, pass=73.00 | OK |
| `checkpoints/topent-student-k200-1.7b-math/eval_step_150` | 4 | 500 | 48.45 | 65.80 | 72.20 | acc=72.20, avg=48.45, maj=65.80, pass=72.20 | OK |
| `checkpoints/topent-student-k200-1.7b-math/eval_step_50` | 4 | 500 | 53.20 | 66.20 | 73.40 | acc=73.40, avg=53.20, maj=66.20, pass=73.40 | OK |

## v1-cot-lora-fullseq-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-cot-lora-fullseq-math/eval_final` | 4 | 500 | 64.65 | 70.80 | 75.80 | acc=75.80, avg=64.65, maj=70.80, pass=75.80 | OK |
| `checkpoints/v1-cot-lora-fullseq-math/eval_step_200` | 4 | 500 | 63.55 | 69.20 | 76.40 | acc=76.40, avg=63.55, maj=69.20, pass=76.40 | OK |

## v1-cot-lora-pos100-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-cot-lora-pos100-math/step_100/eval` | 1 | 500 | 64.20 | 64.20 | 64.20 | acc=64.20, avg=64.20, maj=64.20, pass=64.20 | OK |
| `checkpoints/v1-cot-lora-pos100-math/step_150/eval` | 1 | 500 | 66.80 | 66.80 | 66.80 | acc=66.80, avg=66.80, maj=66.80, pass=66.80 | OK |
| `checkpoints/v1-cot-lora-pos100-math/step_200/eval` | 1 | 500 | 63.00 | 63.00 | 63.00 | acc=63.00, avg=63.00, maj=63.00, pass=63.00 | OK |
| `checkpoints/v1-cot-lora-pos100-math/step_50/eval` | 1 | 500 | 65.00 | 65.00 | 65.00 | acc=65.00, avg=65.00, maj=65.00, pass=65.00 | OK |

## v1-ent-and-k100

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-ent-and-k100/eval_final` | 4 | 500 | 55.35 | 67.20 | 74.60 | acc=74.60, avg=55.35, maj=67.20, pass=74.60 | OK |
| `checkpoints/v1-ent-and-k100/eval_step_200` | 4 | 500 | 54.50 | 66.80 | 73.00 | acc=73.00, avg=54.50, maj=66.80, pass=73.00 | OK |

## v1-ent-or-k100

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-ent-or-k100/eval_final` | 4 | 500 | 53.65 | 64.80 | 72.00 | acc=72.00, avg=53.65, maj=64.80, pass=72.00 | OK |
| `checkpoints/v1-ent-or-k100/eval_step_200` | 4 | 500 | 55.20 | 66.20 | 74.00 | acc=74.00, avg=55.20, maj=66.20, pass=74.00 | OK |

## v1-equalcompute-fullseq-32steps

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-equalcompute-fullseq-32steps/eval_final` | 4 | 500 | 57.00 | 65.00 | 73.80 | acc=73.80, avg=57.00, maj=65.00, pass=73.80 | OK |

## v1-fullft-fullseq-m1.5b-t1.7b-coding

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-fullft-fullseq-m1.5b-t1.7b-coding/eval_final` | 4 | 500 | 51.10 | 60.00 | 70.60 | acc=70.60, avg=51.10, maj=60.00, pass=70.60 | OK |
| `checkpoints/v1-fullft-fullseq-m1.5b-t1.7b-coding/eval_step_200` | 4 | 500 | 51.10 | 60.00 | 70.60 | acc=70.60, avg=51.10, maj=60.00, pass=70.60 | OK |

## v1-fullft-fullseq-m1.5b-t1.7b-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-fullft-fullseq-m1.5b-t1.7b-math/eval_step_100` | 4 | 500 | 54.10 | 61.20 | 72.20 | acc=72.20, avg=54.10, maj=61.20, pass=72.20 | OK |
| `checkpoints/v1-fullft-fullseq-m1.5b-t1.7b-math/eval_step_150` | 4 | 500 | 54.35 | 62.80 | 73.60 | acc=73.60, avg=54.35, maj=62.80, pass=73.60 | OK |
| `checkpoints/v1-fullft-fullseq-m1.5b-t1.7b-math/eval_step_200` | 4 | 500 | 53.50 | 61.60 | 71.80 | acc=71.80, avg=53.50, maj=61.60, pass=71.80 | OK |
| `checkpoints/v1-fullft-fullseq-m1.5b-t1.7b-math/eval_step_50` | 4 | 500 | 52.70 | 59.80 | 71.80 | acc=71.80, avg=52.70, maj=59.80, pass=71.80 | OK |

## v1-gemma-fullft-fullseq-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-gemma-fullft-fullseq-math/step_100/eval` | 4 | 500 | 9.50 | 19.60 | 21.00 | acc=21.00, avg=9.50, maj=19.60, pass=21.00 | OK |
| `checkpoints/v1-gemma-fullft-fullseq-math/step_150/eval` | 4 | 500 | 13.00 | 22.40 | 24.20 | acc=24.20, avg=13.00, maj=22.40, pass=24.20 | OK |
| `checkpoints/v1-gemma-fullft-fullseq-math/step_200/eval` | 4 | 500 | 13.90 | 23.00 | 25.00 | acc=25.00, avg=13.90, maj=23.00, pass=25.00 | OK |
| `checkpoints/v1-gemma-fullft-fullseq-math/step_50/eval` | 4 | 500 | 11.15 | 17.60 | 20.20 | acc=20.20, avg=11.15, maj=17.60, pass=20.20 | OK |

## v1-gemma-fullft-pos100-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-gemma-fullft-pos100-math/eval_step_100` | 4 | 500 | 26.70 | 30.60 | 40.60 | acc=40.60, avg=26.70, maj=30.80, pass=40.60 | MISMATCH: maj 30.80!=30.60 |
| `checkpoints/v1-gemma-fullft-pos100-math/eval_step_150` | 4 | 500 | 25.45 | 28.80 | 38.20 | acc=38.20, avg=25.45, maj=28.80, pass=38.20 | OK |
| `checkpoints/v1-gemma-fullft-pos100-math/eval_step_200` | 4 | 500 | 26.65 | 30.80 | 40.40 | acc=40.40, avg=26.65, maj=30.80, pass=40.40 | OK |
| `checkpoints/v1-gemma-fullft-pos100-math/eval_step_50` | 4 | 500 | 25.20 | 29.60 | 40.60 | acc=40.60, avg=25.20, maj=30.00, pass=40.60 | MISMATCH: maj 30.00!=29.60 |

## v1-gemma-funcall-fullseq

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-gemma-funcall-fullseq/eval_final` | 4 | 500 | 23.10 | 26.00 | 35.60 | acc=35.60, avg=23.10, maj=26.00, pass=35.60 | OK |
| `checkpoints/v1-gemma-funcall-fullseq/eval_step_200` | 4 | 500 | 22.80 | 25.80 | 36.00 | acc=36.00, avg=22.80, maj=25.80, pass=36.00 | OK |

## v1-gemma-multiseed-pos50-s456

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-gemma-multiseed-pos50-s456/eval_final` | 4 | 500 | 25.90 | 30.00 | 37.80 | acc=37.80, avg=25.90, maj=30.00, pass=37.80 | OK |
| `checkpoints/v1-gemma-multiseed-pos50-s456/eval_step_200` | 4 | 500 | 25.50 | 28.60 | 37.40 | acc=37.40, avg=25.50, maj=28.60, pass=37.40 | OK |

## v1-gemma-pos100-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-gemma-pos100-math/eval_step_50` | 4 | 500 | 22.30 | 27.80 | 35.80 | acc=35.80, avg=22.30, maj=28.00, pass=35.80 | MISMATCH: maj 28.00!=27.80 |

## v1-gemma-pos50-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-gemma-pos50-math/eval_step_100` | 4 | 500 | 24.85 | 28.80 | 36.40 | acc=36.40, avg=24.85, maj=28.80, pass=36.40 | OK |
| `checkpoints/v1-gemma-pos50-math/eval_step_150` | 4 | 500 | 25.80 | 29.00 | 37.20 | acc=37.20, avg=25.80, maj=29.00, pass=37.20 | OK |
| `checkpoints/v1-gemma-pos50-math/eval_step_200` | 4 | 500 | 25.70 | 28.00 | 36.40 | acc=36.40, avg=25.70, maj=28.00, pass=36.40 | OK |
| `checkpoints/v1-gemma-pos50-math/eval_step_50` | 4 | 500 | 25.40 | 29.20 | 39.40 | acc=39.40, avg=25.40, maj=29.00, pass=39.40 | MISMATCH: maj 29.00!=29.20 |

## v1-llama-fullft-fullseq-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-fullft-fullseq-math/eval_final` | 4 | 500 | 18.95 | 25.20 | 37.20 | acc=37.20, avg=18.95, maj=25.20, pass=37.20 | OK |
| `checkpoints/v1-llama-fullft-fullseq-math/eval_step_200` | 4 | 500 | 20.10 | 27.20 | 40.20 | acc=40.20, avg=20.10, maj=27.20, pass=40.20 | OK |

## v1-llama-fullft-funcall-pos150

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-fullft-funcall-pos150/eval_final` | 4 | 500 | 17.00 | 25.80 | 35.00 | acc=35.00, avg=17.00, maj=25.80, pass=35.00 | OK |
| `checkpoints/v1-llama-fullft-funcall-pos150/eval_step_200` | 4 | 500 | 17.15 | 23.60 | 34.20 | acc=34.20, avg=17.15, maj=23.60, pass=34.20 | OK |

## v1-llama-fullft-pos100-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-fullft-pos100-math/eval_final` | 4 | 500 | 12.10 | 20.80 | 29.60 | acc=29.60, avg=12.10, maj=20.80, pass=29.60 | OK |
| `checkpoints/v1-llama-fullft-pos100-math/eval_step_200` | 4 | 500 | 12.80 | 22.00 | 30.00 | acc=30.00, avg=12.80, maj=22.00, pass=30.00 | OK |

## v1-llama-fullft-pos150-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-fullft-pos150-math/eval_final` | 4 | 500 | 15.45 | 23.80 | 32.20 | acc=32.20, avg=15.45, maj=23.80, pass=32.20 | OK |
| `checkpoints/v1-llama-fullft-pos150-math/eval_step_200` | 4 | 500 | 14.20 | 24.20 | 34.60 | acc=34.60, avg=14.20, maj=24.20, pass=34.60 | OK |

## v1-llama-fullft-pos200-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-fullft-pos200-math/eval_final` | 4 | 500 | 16.95 | 23.80 | 33.60 | acc=33.60, avg=16.95, maj=23.80, pass=33.60 | OK |
| `checkpoints/v1-llama-fullft-pos200-math/eval_step_200` | 4 | 500 | 16.80 | 24.60 | 35.20 | acc=35.20, avg=16.80, maj=24.60, pass=35.20 | OK |

## v1-llama-fullft-pos300-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-fullft-pos300-math/eval_final` | 4 | 500 | 18.15 | 26.00 | 34.40 | acc=34.40, avg=18.15, maj=26.00, pass=34.40 | OK |
| `checkpoints/v1-llama-fullft-pos300-math/eval_step_200` | 4 | 500 | 17.80 | 24.80 | 35.60 | acc=35.60, avg=17.80, maj=25.00, pass=35.60 | MISMATCH: maj 25.00!=24.80 |

## v1-llama-funcall-fullseq

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-funcall-fullseq/eval_final` | 4 | 500 | 18.95 | 28.20 | 37.80 | acc=37.80, avg=18.95, maj=28.20, pass=37.80 | OK |
| `checkpoints/v1-llama-funcall-fullseq/eval_step_200` | 4 | 500 | 17.30 | 26.60 | 39.00 | acc=39.00, avg=17.30, maj=26.60, pass=39.00 | OK |

## v1-llama-funcall-pos100

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-funcall-pos100/eval_final` | 4 | 500 | 13.15 | 22.40 | 32.60 | acc=32.60, avg=13.15, maj=22.40, pass=32.60 | OK |
| `checkpoints/v1-llama-funcall-pos100/eval_step_200` | 4 | 500 | 12.55 | 22.80 | 31.40 | acc=31.40, avg=12.55, maj=22.80, pass=31.40 | OK |

## v1-llama-funcall-pos150

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-funcall-pos150/eval_final` | 4 | 500 | 15.95 | 24.00 | 34.60 | acc=34.60, avg=15.95, maj=24.00, pass=34.60 | OK |
| `checkpoints/v1-llama-funcall-pos150/eval_step_200` | 4 | 500 | 15.75 | 24.60 | 35.40 | acc=35.40, avg=15.75, maj=24.60, pass=35.40 | OK |

## v1-llama-funcall-pos200

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-funcall-pos200/eval_final` | 4 | 500 | 16.20 | 25.40 | 37.20 | acc=37.20, avg=16.20, maj=25.40, pass=37.20 | OK |
| `checkpoints/v1-llama-funcall-pos200/eval_step_200` | 4 | 500 | 15.80 | 23.80 | 35.40 | acc=35.40, avg=15.80, maj=23.80, pass=35.40 | OK |

## v1-llama-funcall-pos50

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-funcall-pos50/eval_final` | 4 | 500 | 8.60 | 17.80 | 24.20 | acc=24.20, avg=8.60, maj=17.80, pass=24.20 | OK |
| `checkpoints/v1-llama-funcall-pos50/eval_step_200` | 4 | 500 | 8.90 | 19.80 | 25.20 | acc=25.20, avg=8.90, maj=19.80, pass=25.20 | OK |

## v1-llama-lora-fullseq-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-lora-fullseq-math/eval_final` | 4 | 500 | 20.30 | 27.60 | 37.40 | acc=37.40, avg=20.30, maj=27.60, pass=37.40 | OK |
| `checkpoints/v1-llama-lora-fullseq-math/eval_step_200` | 4 | 500 | 20.35 | 28.20 | 39.60 | acc=39.60, avg=20.35, maj=28.20, pass=39.60 | OK |

## v1-llama-lora-fullseq-math-fixed

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-lora-fullseq-math-fixed/eval_final` | 4 | 500 | 20.65 | 29.20 | 39.80 | acc=39.80, avg=20.65, maj=29.20, pass=39.80 | OK |
| `checkpoints/v1-llama-lora-fullseq-math-fixed/eval_step_200` | 4 | 500 | 19.50 | 27.40 | 36.60 | acc=36.60, avg=19.50, maj=27.40, pass=36.60 | OK |

## v1-llama-lora-pos100-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-lora-pos100-math/eval_final` | 4 | 500 | 17.80 | 26.60 | 38.40 | acc=38.40, avg=17.80, maj=26.60, pass=38.40 | OK |
| `checkpoints/v1-llama-lora-pos100-math/eval_step_200` | 4 | 500 | 18.90 | 27.60 | 39.00 | acc=39.00, avg=18.90, maj=27.60, pass=39.00 | OK |
| `checkpoints/v1-llama-lora-pos100-math/step_100/eval` | 4 | 500 | 17.90 | 26.60 | 38.60 | acc=38.60, avg=17.90, maj=26.60, pass=38.60 | OK |
| `checkpoints/v1-llama-lora-pos100-math/step_150/eval` | 4 | 500 | 16.45 | 25.20 | 34.60 | acc=34.60, avg=16.45, maj=25.20, pass=34.60 | OK |
| `checkpoints/v1-llama-lora-pos100-math/step_200/eval` | 4 | 500 | 17.35 | 27.20 | 36.40 | acc=36.40, avg=17.35, maj=27.20, pass=36.40 | OK |
| `checkpoints/v1-llama-lora-pos100-math/step_50/eval` | 4 | 500 | 12.90 | 24.20 | 31.80 | acc=31.80, avg=12.90, maj=24.20, pass=31.80 | OK |

## v1-llama-lora-pos100-math-fixed

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-lora-pos100-math-fixed/eval_final` | 4 | 500 | 13.15 | 23.20 | 30.40 | acc=30.40, avg=13.15, maj=23.20, pass=30.40 | OK |
| `checkpoints/v1-llama-lora-pos100-math-fixed/eval_step_200` | 4 | 500 | 12.35 | 22.20 | 30.80 | acc=30.80, avg=12.35, maj=22.20, pass=30.80 | OK |

## v1-llama-lora-pos150-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-lora-pos150-math/eval_final` | 4 | 500 | 18.75 | 25.80 | 37.40 | acc=37.40, avg=18.75, maj=25.80, pass=37.40 | OK |
| `checkpoints/v1-llama-lora-pos150-math/eval_step_200` | 4 | 500 | 18.40 | 24.80 | 36.20 | acc=36.20, avg=18.40, maj=24.80, pass=36.20 | OK |

## v1-llama-lora-pos200-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-lora-pos200-math/eval_final` | 4 | 500 | 19.15 | 25.80 | 37.20 | acc=37.20, avg=19.15, maj=25.80, pass=37.20 | OK |
| `checkpoints/v1-llama-lora-pos200-math/eval_step_200` | 4 | 500 | 22.45 | 30.40 | 42.60 | acc=42.60, avg=22.45, maj=30.40, pass=42.60 | OK |

## v1-llama-lora-pos300-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-llama-lora-pos300-math/eval_final` | 4 | 500 | 20.40 | 28.00 | 38.20 | acc=38.20, avg=20.40, maj=28.00, pass=38.20 | OK |
| `checkpoints/v1-llama-lora-pos300-math/eval_step_200` | 4 | 500 | 20.75 | 26.40 | 37.60 | acc=37.60, avg=20.75, maj=26.40, pass=37.60 | OK |

## v1-multiseed-fullseq-s123

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-multiseed-fullseq-s123/eval_final` | 4 | 500 | 48.95 | 66.00 | 72.60 | acc=72.60, avg=48.95, maj=66.00, pass=72.60 | OK |
| `checkpoints/v1-multiseed-fullseq-s123/eval_step_200` | 4 | 500 | 50.45 | 65.20 | 74.20 | acc=74.20, avg=50.45, maj=65.20, pass=74.20 | OK |
| `checkpoints/v1-multiseed-fullseq-s123/step_100/eval` | 1 | 500 | 44.00 | 44.00 | 44.00 | acc=44.00, avg=44.00, maj=44.00, pass=44.00 | OK |
| `checkpoints/v1-multiseed-fullseq-s123/step_150/eval` | 1 | 500 | 47.60 | 47.60 | 47.60 | acc=47.60, avg=47.60, maj=47.60, pass=47.60 | OK |
| `checkpoints/v1-multiseed-fullseq-s123/step_200/eval` | 1 | 500 | 49.40 | 49.40 | 49.40 | acc=49.40, avg=49.40, maj=49.40, pass=49.40 | OK |
| `checkpoints/v1-multiseed-fullseq-s123/step_50/eval` | 1 | 500 | 63.60 | 63.60 | 63.60 | acc=63.60, avg=63.60, maj=63.60, pass=63.60 | OK |

## v1-multiseed-fullseq-s456

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-multiseed-fullseq-s456/eval_final` | 4 | 500 | 57.85 | 67.80 | 74.00 | acc=74.00, avg=57.85, maj=67.80, pass=74.00 | OK |
| `checkpoints/v1-multiseed-fullseq-s456/eval_step_200` | 4 | 500 | 57.55 | 67.00 | 74.20 | acc=74.20, avg=57.55, maj=67.00, pass=74.20 | OK |
| `checkpoints/v1-multiseed-fullseq-s456/step_100/eval` | 1 | 500 | 50.20 | 50.20 | 50.20 | acc=50.20, avg=50.20, maj=50.20, pass=50.20 | OK |
| `checkpoints/v1-multiseed-fullseq-s456/step_150/eval` | 1 | 500 | 59.60 | 59.60 | 59.60 | acc=59.60, avg=59.60, maj=59.60, pass=59.60 | OK |
| `checkpoints/v1-multiseed-fullseq-s456/step_200/eval` | 1 | 500 | 61.80 | 61.80 | 61.80 | acc=61.80, avg=61.80, maj=61.80, pass=61.80 | OK |
| `checkpoints/v1-multiseed-fullseq-s456/step_50/eval` | 1 | 500 | 63.20 | 63.20 | 63.20 | acc=63.20, avg=63.20, maj=63.20, pass=63.20 | OK |

## v1-multiseed-pos100-s123

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-multiseed-pos100-s123/eval_final` | 4 | 500 | 61.10 | 65.80 | 76.80 | acc=76.80, avg=61.10, maj=65.80, pass=76.80 | OK |
| `checkpoints/v1-multiseed-pos100-s123/eval_step_200` | 4 | 500 | 61.00 | 65.80 | 76.00 | acc=76.00, avg=61.00, maj=65.80, pass=76.00 | OK |
| `checkpoints/v1-multiseed-pos100-s123/step_100/eval` | 1 | 500 | 55.60 | 55.60 | 55.60 | acc=55.60, avg=55.60, maj=55.60, pass=55.60 | OK |
| `checkpoints/v1-multiseed-pos100-s123/step_150/eval` | 1 | 500 | 62.00 | 62.00 | 62.00 | acc=62.00, avg=62.00, maj=62.00, pass=62.00 | OK |
| `checkpoints/v1-multiseed-pos100-s123/step_200/eval` | 1 | 500 | 62.80 | 62.80 | 62.80 | acc=62.80, avg=62.80, maj=62.80, pass=62.80 | OK |
| `checkpoints/v1-multiseed-pos100-s123/step_50/eval` | 1 | 500 | 60.00 | 60.00 | 60.00 | acc=60.00, avg=60.00, maj=60.00, pass=60.00 | OK |

## v1-multiseed-pos100-s456

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-multiseed-pos100-s456/eval_final` | 4 | 500 | 59.05 | 65.60 | 74.00 | acc=74.00, avg=59.05, maj=65.60, pass=74.00 | OK |
| `checkpoints/v1-multiseed-pos100-s456/eval_step_200` | 4 | 500 | 60.40 | 66.00 | 75.60 | acc=75.60, avg=60.40, maj=66.00, pass=75.60 | OK |
| `checkpoints/v1-multiseed-pos100-s456/step_100/eval` | 1 | 500 | 61.60 | 61.60 | 61.60 | acc=61.60, avg=61.60, maj=61.60, pass=61.60 | OK |
| `checkpoints/v1-multiseed-pos100-s456/step_150/eval` | 1 | 500 | 60.20 | 60.20 | 60.20 | acc=60.20, avg=60.20, maj=60.20, pass=60.20 | OK |
| `checkpoints/v1-multiseed-pos100-s456/step_200/eval` | 1 | 500 | 61.20 | 61.20 | 61.20 | acc=61.20, avg=61.20, maj=61.20, pass=61.20 | OK |
| `checkpoints/v1-multiseed-pos100-s456/step_50/eval` | 1 | 500 | 59.60 | 59.60 | 59.60 | acc=59.60, avg=59.60, maj=59.60, pass=59.60 | OK |

## v1-reopold-lora-math

| Eval dir | n | total | avg | maj | pass | summary.json | match |
|---|---|---|---|---|---|---|---|
| `checkpoints/v1-reopold-lora-math/step_100/eval` | 4 | 500 | 60.35 | 64.20 | 74.00 | acc=74.00, avg=60.35, maj=64.20, pass=74.00 | OK |
| `checkpoints/v1-reopold-lora-math/step_150/eval` | 4 | 500 | 56.75 | 66.20 | 73.20 | acc=73.20, avg=56.75, maj=66.20, pass=73.20 | OK |
| `checkpoints/v1-reopold-lora-math/step_50/eval` | 4 | 500 | 60.55 | 66.00 | 76.20 | acc=76.20, avg=60.55, maj=66.00, pass=76.20 | OK |

## Mismatches between recomputed and summary.json

| Path | Issue |
|---|---|
| `checkpoints/gemma-pos50-coding-retrain/eval_step_200/results.jsonl` | MISMATCH: maj 26.40!=26.20 |
| `checkpoints/gemma-pos50-math-retrain/eval_final/results.jsonl` | MISMATCH: maj 27.20!=27.00 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq/eval_step_100/results.jsonl` | MISMATCH: maj 18.20!=18.40 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq/eval_step_150/results.jsonl` | MISMATCH: maj 18.20!=18.40 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq/eval_step_200/results.jsonl` | MISMATCH: maj 18.20!=18.40 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-fullseq/eval_step_50/results.jsonl` | MISMATCH: maj 18.20!=18.40 |
| `checkpoints/scale-gemma2-2b-tgemma3-4b-math-pos50/eval_step_100/results.jsonl` | MISMATCH: maj 16.00!=15.80 |
| `checkpoints/v1-gemma-fullft-pos100-math/eval_step_100/results.jsonl` | MISMATCH: maj 30.80!=30.60 |
| `checkpoints/v1-gemma-fullft-pos100-math/eval_step_50/results.jsonl` | MISMATCH: maj 30.00!=29.60 |
| `checkpoints/v1-gemma-pos100-math/eval_step_50/results.jsonl` | MISMATCH: maj 28.00!=27.80 |
| `checkpoints/v1-gemma-pos50-math/eval_step_50/results.jsonl` | MISMATCH: maj 29.00!=29.20 |
| `checkpoints/v1-llama-fullft-pos300-math/eval_step_200/results.jsonl` | MISMATCH: maj 25.00!=24.80 |

## Files with no summary.json (0)


## n=1 evals (avg=maj=pass; cannot compute pass@4) — 20 files

- `checkpoints/v1-cot-lora-pos100-math/step_100/eval/results.jsonl` — score=64.20
- `checkpoints/v1-cot-lora-pos100-math/step_150/eval/results.jsonl` — score=66.80
- `checkpoints/v1-cot-lora-pos100-math/step_200/eval/results.jsonl` — score=63.00
- `checkpoints/v1-cot-lora-pos100-math/step_50/eval/results.jsonl` — score=65.00
- `checkpoints/v1-multiseed-fullseq-s123/step_100/eval/results.jsonl` — score=44.00
- `checkpoints/v1-multiseed-fullseq-s123/step_150/eval/results.jsonl` — score=47.60
- `checkpoints/v1-multiseed-fullseq-s123/step_200/eval/results.jsonl` — score=49.40
- `checkpoints/v1-multiseed-fullseq-s123/step_50/eval/results.jsonl` — score=63.60
- `checkpoints/v1-multiseed-fullseq-s456/step_100/eval/results.jsonl` — score=50.20
- `checkpoints/v1-multiseed-fullseq-s456/step_150/eval/results.jsonl` — score=59.60
- `checkpoints/v1-multiseed-fullseq-s456/step_200/eval/results.jsonl` — score=61.80
- `checkpoints/v1-multiseed-fullseq-s456/step_50/eval/results.jsonl` — score=63.20
- `checkpoints/v1-multiseed-pos100-s123/step_100/eval/results.jsonl` — score=55.60
- `checkpoints/v1-multiseed-pos100-s123/step_150/eval/results.jsonl` — score=62.00
- `checkpoints/v1-multiseed-pos100-s123/step_200/eval/results.jsonl` — score=62.80
- `checkpoints/v1-multiseed-pos100-s123/step_50/eval/results.jsonl` — score=60.00
- `checkpoints/v1-multiseed-pos100-s456/step_100/eval/results.jsonl` — score=61.60
- `checkpoints/v1-multiseed-pos100-s456/step_150/eval/results.jsonl` — score=60.20
- `checkpoints/v1-multiseed-pos100-s456/step_200/eval/results.jsonl` — score=61.20
- `checkpoints/v1-multiseed-pos100-s456/step_50/eval/results.jsonl` — score=59.60

## Canonical-doc spot checks

Spot-check selected best-step avg@4 against the n1bs16 canonical numbers in `CLAUDE.md` (note: those numbers are for the Qwen2.5-Math-1.5B → Qwen3-1.7B math run; the UCLACG checkpoints below are mostly the *llama* / *gemma* / *multiseed* variants and are NOT what the canonical table cites).

| Family | Best step | avg@4 (recomputed) | maj@4 | pass@4 |
|---|---|---|---|---|
| eval_results/llama31-8b-baseline | (unknown) | 37.25 | 46.80 | 58.60 |
| eval_results/llama32-1b-baseline | (unknown) | 15.20 | 20.40 | 32.60 |
| eval_results/llama32-3b-baseline | (unknown) | 19.80 | 30.80 | 38.80 |
| format-mask-1.7b-math | step_50 | 62.05 | 67.20 | 76.20 |
| gemma-pos50-coding-retrain | step_200 | 23.15 | 26.20 | 35.60 |
| gemma-pos50-math-retrain | eval_final | 24.05 | 27.00 | 35.80 |
| hi-kl-hi-surp-1.7b-math | eval_final | 55.60 | 65.00 | 73.00 |
| hi-kl-hi-surp-topk100-1.7b-math | eval_final | 56.75 | 65.40 | 73.60 |
| scale-gemma2-2b-tgemma3-4b-math-fullseq | step_100 | 11.70 | 18.40 | 24.60 |
| scale-gemma2-2b-tgemma3-4b-math-pos50 | step_100 | 7.20 | 15.80 | 18.00 |
| topent-student-k200-1.7b-math | step_50 | 53.20 | 66.20 | 73.40 |
| v1-cot-lora-fullseq-math | eval_final | 64.65 | 70.80 | 75.80 |
| v1-ent-and-k100 | eval_final | 55.35 | 67.20 | 74.60 |
| v1-ent-or-k100 | step_200 | 55.20 | 66.20 | 74.00 |
| v1-equalcompute-fullseq-32steps | eval_final | 57.00 | 65.00 | 73.80 |
| v1-fullft-fullseq-m1.5b-t1.7b-coding | eval_final | 51.10 | 60.00 | 70.60 |
| v1-fullft-fullseq-m1.5b-t1.7b-math | step_150 | 54.35 | 62.80 | 73.60 |
| v1-gemma-fullft-fullseq-math | step_200 | 13.90 | 23.00 | 25.00 |
| v1-gemma-fullft-pos100-math | step_100 | 26.70 | 30.60 | 40.60 |
| v1-gemma-funcall-fullseq | eval_final | 23.10 | 26.00 | 35.60 |
| v1-gemma-multiseed-pos50-s456 | eval_final | 25.90 | 30.00 | 37.80 |
| v1-gemma-pos100-math | step_50 | 22.30 | 27.80 | 35.80 |
| v1-gemma-pos50-math | step_150 | 25.80 | 29.00 | 37.20 |
| v1-llama-fullft-fullseq-math | step_200 | 20.10 | 27.20 | 40.20 |
| v1-llama-fullft-funcall-pos150 | step_200 | 17.15 | 23.60 | 34.20 |
| v1-llama-fullft-pos100-math | step_200 | 12.80 | 22.00 | 30.00 |
| v1-llama-fullft-pos150-math | eval_final | 15.45 | 23.80 | 32.20 |
| v1-llama-fullft-pos200-math | eval_final | 16.95 | 23.80 | 33.60 |
| v1-llama-fullft-pos300-math | eval_final | 18.15 | 26.00 | 34.40 |
| v1-llama-funcall-fullseq | eval_final | 18.95 | 28.20 | 37.80 |
| v1-llama-funcall-pos100 | eval_final | 13.15 | 22.40 | 32.60 |
| v1-llama-funcall-pos150 | eval_final | 15.95 | 24.00 | 34.60 |
| v1-llama-funcall-pos200 | eval_final | 16.20 | 25.40 | 37.20 |
| v1-llama-funcall-pos50 | step_200 | 8.90 | 19.80 | 25.20 |
| v1-llama-lora-fullseq-math | step_200 | 20.35 | 28.20 | 39.60 |
| v1-llama-lora-fullseq-math-fixed | eval_final | 20.65 | 29.20 | 39.80 |
| v1-llama-lora-pos100-math | step_200 | 18.90 | 27.60 | 39.00 |
| v1-llama-lora-pos100-math-fixed | eval_final | 13.15 | 23.20 | 30.40 |
| v1-llama-lora-pos150-math | eval_final | 18.75 | 25.80 | 37.40 |
| v1-llama-lora-pos200-math | step_200 | 22.45 | 30.40 | 42.60 |
| v1-llama-lora-pos300-math | step_200 | 20.75 | 26.40 | 37.60 |
| v1-multiseed-fullseq-s123 | step_200 | 50.45 | 65.20 | 74.20 |
| v1-multiseed-fullseq-s456 | eval_final | 57.85 | 67.80 | 74.00 |
| v1-multiseed-pos100-s123 | eval_final | 61.10 | 65.80 | 76.80 |
| v1-multiseed-pos100-s456 | step_200 | 60.40 | 66.00 | 75.60 |
| v1-reopold-lora-math | step_50 | 60.55 | 66.00 | 76.20 |
