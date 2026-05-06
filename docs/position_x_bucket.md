# Position × KL × Surprise Bucket Distribution

Data: `/zhi_backup/ziheng/quick-distillation/docs/kl_position_analysis_v2/raw_logprobs.jsonl`
Total tokens: 74549
Global thresholds: KL_p75=0.013, surp_p75=0.006

## Bucket share by position window

Each row: % of tokens IN that position window that fall into each bucket.

| Position | n | hiKL_hiE | hiKL_loE | loKL_hiE | loKL_loE | mean KL | mean surp |
|---|---:|---:|---:|---:|---:|---:|---:|
| **0-100** | 9991 | 46.5% | 3.1% | 6.7% | 43.7% | 1.41 | 0.32 |
| **100-300** | 18967 | 24.6% | 4.2% | 4.6% | 66.6% | 0.41 | 0.11 |
| **300-500** | 15595 | 17.2% | 4.8% | 4.0% | 73.9% | 0.32 | 0.07 |
| **500+** | 29996 | 11.9% | 4.1% | 3.0% | 81.1% | 0.21 | 0.05 |

## Cumulative bucket coverage by position window

Each row: % of ALL tokens in that bucket (across full sequences) that
fall in this position window. Tells us where each bucket's mass lives.

| Position | hiKL_hiE share | hiKL_loE share | loKL_hiE share | loKL_loE share |
|---|---:|---:|---:|---:|
| **0-100** | 29.9% | 10.0% | 21.8% | 8.3% |
| **100-300** | 30.0% | 26.0% | 28.3% | 23.9% |
| **300-500** | 17.3% | 24.4% | 20.4% | 21.8% |
| **500+** | 22.9% | 39.6% | 29.5% | 46.0% |

## Direct test: prefix-100 vs full sequence

Baseline (full sequence) bucket shares are p75-by-construction:
  hiKL_hiE ≈ 21%, hiKL_loE ≈ 4%, loKL_hiE ≈ 4%, loKL_loE ≈ 71%

Prefix-100 actual:
  hiKL_hiE = 46.5% (enrichment vs baseline 21%: 2.22x)
  hiKL_loE = 3.1% (enrichment vs baseline ~4%: 0.77x)
  loKL_hiE = 6.7% (enrichment vs baseline ~4%: 1.67x)
  loKL_loE = 43.7% (enrichment vs baseline ~71%: 0.62x)

Reading guide:
  - If prefix hiKL_hiE > baseline hiKL_hiE  → prefix concentrates 'useful gradient'
  - If prefix hiKL_loE < baseline hiKL_loE  → prefix avoids 'harmful gradient'
  - Both true → prefix-100 is a principled positional proxy for KL×entropy selection