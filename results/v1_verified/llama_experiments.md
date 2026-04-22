# Llama 3.2 1B → Llama 3.1 8B Experiments

## Baselines (MATH-500 avg@4)
| Model | avg@4 |
|-------|-------|
| Llama 3.2 1B-Instruct (student) | 15.20% |
| Llama 3.2 3B-Instruct | 19.80% |
| Llama 3.1 8B-Instruct (teacher) | 37.25% |

## LoRA Results (200 steps, 3200 problems, n=1, bs=16)

### pos-100
| Step | avg@4 |
|------|-------|
| 50 | 14.00% |
| 100 | 17.75% |
| 150 | 18.05% |
| **200** | **18.95%** |

### fullseq (max_new_tokens=2048)
| Step | avg@4 |
|------|-------|
| 200 | **20.35%** |

Note: intermediate step evals failed (script bug). Only step 200 eval available from built-in eval.

## FullFT Results (200 steps)

### pos-100
| Step | avg@4 |
|------|-------|
| 200 | 12.80% ⚠️ |

### fullseq (max_new_tokens=2048)
| Step | avg@4 |
|------|-------|
| 200 | **20.10%** |

## Summary: All Methods (fixed nothink bug)

### LoRA Math (MATH-500 avg@4, baseline 15.20%)

| Position | avg@4 | vs Baseline |
|----------|-------|-------------|
| pos-100 | 12.35% ⚠️ | -2.85pp |
| pos-150 | 18.75% | +3.55pp |
| **pos-200** | **22.45%** | **+7.25pp** |
| pos-300 | 20.75% | +5.55pp |
| fullseq | 20.65% | +5.45pp |

### FullFT Math (MATH-500 avg@4, baseline 15.20%)

| Position | avg@4 | vs Baseline |
|----------|-------|-------------|
| pos-100 | 12.80% ⚠️ | -2.40pp |
| pos-150 | 14.20% | -1.00pp |
| pos-200 | 16.80% | +1.60pp |
| pos-300 | 17.80% | +2.60pp |
| fullseq | 20.10% | +4.90pp |

## Pre-training KL Profile (50 problems, 512 tokens)

| Positions | Mean KL | Teacher Ent | Agreement |
|-----------|---------|-------------|-----------|
| 0-50 | 0.186 | 0.423 | 89.1% |
| 50-100 | 0.184 | 0.432 | 88.5% |
| 100-200 | 0.193 | 0.415 | 90.1% |
| 200-300 | 0.145 | 0.372 | 92.2% |

First 100 / rest ratio: **1.15×** (nearly flat — unlike Qwen's 2.81×)

## Key Findings
1. **Fullseq >> pos-100** in both LoRA (+8.3pp) and FullFT (+7.3pp) — opposite of Gemma/Qwen
2. **No fullseq degradation** — loss stable throughout, no collapse
3. **pos-100 HURTS** — drops below baseline (12.35% vs 15.20%)
4. **Root cause: flat KL profile** — Llama 1B→8B KL is uniformly low (0.18) at all positions, ratio=1.15×. No early-position signal concentration.
5. Compare: Qwen has KL=1.42 at first 100 tokens (ratio=2.81×), so pos-100 captures the densest signal
6. **Positional distillation works when KL is front-loaded, fails when KL is flat**
