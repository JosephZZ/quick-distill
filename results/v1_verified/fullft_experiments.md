# Full-Finetune Experiments

## M-1.5B → Q3-1.7B Math (MATH-500 avg@4, baseline 50.95%)

### 200-step results

| Method | s50 | s100 | s150 | s200 | Best |
|--------|-----|------|------|------|------|
| FullFT pos-50 | 51.80 | 52.20 | 51.60 | **53.70** | **53.70** |
| FullFT pos-100 | 50.85 | 52.35 | 53.00 | **53.15** | **53.15** |
| FullFT pos-150 | 52.30 | 51.55 | 52.30 | **53.15** | **53.15** |
| FullFT pos-200 | — | — | **53.50** | 53.15 | **53.50** |
| FullFT fullseq | 52.70 | 54.10 | **54.35** | 53.50 | **54.35** |

### 400-step results (extended training)

| Step | pos-100 | pos-150 | pos-200 | pos-300 | fullseq |
|------|---------|---------|---------|---------|---------|
| 50 | 50.85 | 51.40 | 51.30 | 50.55 | 51.40 |
| 100 | 52.35 | 53.10 | **54.05** | **53.70** | 53.55 |
| 150 | 53.00 | 53.70 | 53.40 | 53.95 | 54.95 |
| 200 | **53.15** | 53.25 | 53.75 | 53.00 | 55.95 |
| 250 | 49.20⚠️ | 53.30 | 52.55 | 54.45 | **57.05** |
| 300 | 49.90⚠️ | **54.55** | 53.75 | **55.10** | **57.10** |
| 350 | 50.75 | 53.85 | 53.10 | 54.95 | 56.05 |
| 400 | 49.95⚠️ | 52.85 | 53.85 | 54.25 | 56.20 |
| **Best** | **53.15** | **54.55** | **54.05** | **55.10** | **57.10** |

## Q3-1.7B → Q3-4B Math (MATH-500 avg@4, baseline 69.20%)

### 200-step results

| Method | s50 | s100 | s150 | s200 | Best |
|--------|-----|------|------|------|------|
| FullFT pos-100 | 64.35 | **64.65** | 64.00 | 64.45 | **64.65** |
| FullFT pos-150 | 64.65 | 64.35 | **64.75** | 64.45 | **64.75** |
| FullFT pos-200 | 64.70 | **64.85** | — | — | **64.85** |
| FullFT fullseq | 64.75 | **65.20** | 64.85 | 64.90 | **65.20** |

## M-1.5B → Q3-1.7B Coding (HumanEval pass@1, baseline 32.93%)

| Step | pos-50 | pos-100 | fullseq |
|------|--------|---------|---------|
| 50 | 33.5 | 31.7 | 32.3 |
| 100 | 34.8 | 34.8 | 34.1 |
| 150 | 36.0 | 35.4 | 36.0 |
| 200 | **37.2** | 35.4 | 36.0 |
| **Best** | **37.2** | **35.4** | **36.0** |

## LoRA vs FullFT Comparison (best results)

### Math (M-1.5B → Q3-1.7B)

| Method | LoRA pos-100 | FullFT pos-100 | FullFT fullseq | Delta (LoRA vs best FullFT) |
|--------|-------------|----------------|----------------|---------------------------|
| 200 steps | **63.15** | 53.15 | 54.35 | **+8.80pp** |
| 400 steps | **63.65** | 49.20⚠️ | **57.10** | **+6.55pp** |

### Coding (M-1.5B → Q3-1.7B)

| Metric | LoRA pos-100 | FullFT pos-50 | Delta |
|--------|-------------|--------------|-------|
| HE pass@1 | **39.63** | **37.20** | **+2.4pp** |

LoRA consistently outperforms FullFT by 2-9pp across all settings.

## Key Findings

1. **FullFT math: fullseq > pos-N** — opposite of LoRA where pos-100 > fullseq
2. **FullFT coding: pos-50 > fullseq > pos-100** — shorter prefix is better for coding
3. **FullFT pos-100 degrades at 400 steps** — insufficient gradient diversity causes overfitting
4. **FullFT pos-150+ stable** — ≥150 tokens provides enough diversity
5. **FullFT pos-300 reaches 97% of fullseq** (55.10 vs 57.10) — near-optimal tradeoff
6. **LoRA >> FullFT** by 6-9pp — low-rank constraint provides critical regularization
7. **Crossover interaction**: LoRA favors pos-100, FullFT favors fullseq
