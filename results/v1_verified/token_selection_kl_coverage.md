# Token Selection & KL Coverage Analysis

## Token Selection Comparison (Qwen Math, K=100, LoRA n1bs16)

| Strategy | KL Coverage | Ent Overlap | Best avg@4 | Stable? | Contiguous? |
|----------|------------|-------------|------------|---------|------------|
| **prefix** | **45.6%** | **33.5%** | **65.85%** | ✅ | ✅ |
| ent_teacher | 50.5% | 64.0% | 63.30% | ❌ degrades | ❌ |
| ent_student | 67.0% | 94.8% | 62.70% | partial | ❌ |
| ent_and | 57.3% | 78.9% | 54.50% | ? | ❌ |
| ent_or | 64.3% | 81.1% | 55.20% | ? | ❌ |
| top_kl | **93.2%** | 68.6% | **53.35%** | ❌ | ❌ |
| random | 21.1% | 19.8% | 61.25% | ❌ collapses | ❌ |
| middle | 15.8% | 16.9% | 61.80% | ✅ | ✅ |
| last | 13.9% | 13.2% | 62.30% | ❌ collapses | ✅ |

## KL Coverage Analysis (Llama 1B → 8B, K=100)

| Strategy | KL Coverage | Ent Overlap | Avg Pos | Contiguous? |
|----------|------------|-------------|---------|------------|
| prefix | 34.8% | 24.1% | 49.5 | Yes |
| top_kl | 91.7% | 55.1% | 176.2 | No |
| ent_student | 78.7% | 71.8% | 169.6 | No |
| ent_teacher | 72.4% | 61.6% | 176.5 | No |
| random | 31.9% | 19.7% | 177.0 | No |

## Pre-training KL Profiles

### Qwen (M-1.5B → Q3-1.7B)

| Positions | Mean KL | Teacher Ent | Agreement | N tokens |
|-----------|---------|-------------|-----------|----------|
| 0-50 | 1.905 | 0.259 | 75.4% | 2467 |
| 50-100 | 0.938 | 0.214 | 81.9% | 2450 |
| 100-200 | 0.649 | 0.161 | 86.3% | 4844 |
| 200-300 | 0.517 | 0.161 | 87.8% | 4497 |
| 300-500 | 0.427 | 0.172 | 88.8% | 8124 |

First 100 / rest ratio: **2.81×**

### Llama (3.2-1B → 3.1-8B)

| Positions | Mean KL | Teacher Ent | Agreement | N tokens |
|-----------|---------|-------------|-----------|----------|
| 0-50 | 0.185 | 0.423 | 89.1% | 2500 |
| 50-100 | 0.184 | 0.432 | 88.5% | 2494 |
| 100-200 | 0.193 | 0.415 | 90.1% | 4649 |
| 200-300 | 0.145 | 0.372 | 92.2% | 3480 |
| 300-500 | 0.148 | 0.387 | 91.7% | 5177 |

First 100 / rest ratio: **1.15×** (flat)

## Key Finding: Information-Quality Paradox

Top-KL selection (93.2% coverage) performs WORST (53.35%).
Prefix selection (45.6% coverage) performs BEST (65.85%).
More KL signal ≠ better distillation. High-KL tokens are format/style noise.
