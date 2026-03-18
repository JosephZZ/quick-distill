# KL Divergence Position Analysis Report

## Overview

Analysis of token-position-wise KL divergence between teacher (Qwen3-1.7B) and student (Qwen2.5-Math-1.5B) during on-policy distillation. This report covers per-position KL distribution, cumulative KL patterns, and their relationship to distillation performance. Token classification analysis is covered in a separate report.

## Per-Position KL Distribution

### Phase-wise Summary

| Phase | Position Range | Mean KL | Description |
|-------|---------------|---------|-------------|
| Ultra-early | 0-4 | 1.714 | Strategy decision phase |
| Early | 5-19 | 0.934 | Rapid decline |
| Mid-early | 20-49 | 0.896 | Gradual decline |
| Mid | 50-99 | 0.525 | Stable low values |
| Mid-late | 100-149 | 0.449 | Continued decline |
| Late | 150-199 | 0.383 | Most stable / converged |

### Decay Ratios

- Ultra-early vs Late: 4.48x difference
- Position 0 vs Position 192 (min): 8.51x difference (3.057 vs 0.359)
- First 10 vs Last 10 positions: 4.26x difference

### Detailed Position Values (2,000 trajectories)

| Pos | KL | Pos | KL | Pos | KL |
|-----|-------|-----|-------|-----|-------|
| 0 | 3.057 | 10 | 1.036 | 20 | 1.031 |
| 1 | 1.979 | 11 | 1.007 | 21 | 1.018 |
| 2 | 1.104 | 12 | 0.843 | 22 | 0.891 |
| 3 | 1.510 | 13 | 1.007 | 23 | 0.954 |
| 4 | 0.921 | 14 | 0.839 | 24 | 0.946 |
| 5 | 0.962 | 15 | 1.027 | 25 | 1.050 |
| 6 | 0.946 | 16 | 0.764 | 26 | 1.083 |
| 7 | 0.846 | 17 | 1.011 | 27 | 1.095 |
| 8 | 0.853 | 18 | 0.987 | 28 | 1.064 |
| 9 | 0.936 | 19 | 0.943 | 29 | 1.207 |

### Extended Position Values (59,936 trajectories)

| Pos | KL | Pos | KL |
|-----|-------|-----|-------|
| 30 | ~1.1 | 100 | ~0.45 |
| 40 | ~0.9 | 120 | ~0.43 |
| 50 | 0.693 | 140 | ~0.42 |
| 60 | 0.599 | 150 | 0.410 |
| 70 | 0.535 | 160 | 0.396 |
| 75 | 0.507 | 170 | 0.386 |
| 80 | 0.479 | 175 | 0.374 |
| 90 | 0.454 | 180 | 0.375 |
| | | 190 | 0.366 |
| | | 192 | 0.359 |

## Cumulative KL Distribution

### Per-Position Mean |log p_student - log p_teacher|

| Position | Mean |diff| | # Samples |
|----------|-------------|-----------|
| 0 | 8.222 | 59,936 |
| 1 | 1.670 | 59,936 |
| 3 | 1.522 | 59,936 |
| 5 | 2.057 | 59,936 |
| 10 | 1.272 | 59,936 |
| 20 | 1.170 | 59,929 |
| 50 | 1.238 | 59,880 |
| 100 | 0.823 | 59,434 |
| 150 | 0.646 | 58,201 |
| 200 | 0.575 | 55,629 |
| 300 | 0.565 | 44,405 |
| 400 | 0.569 | 28,067 |
| 500 | 0.628 | 12,210 |

### Per-Trajectory Cumulative KL Distribution

59,936 trajectories, mean sequence length 386.

| First N tokens | Mean % of total | Median % | P25 | P75 |
|----------------|-----------------|----------|------|------|
| 5 | 5.2% | 4.1% | 2.7% | 6.3% |
| 10 | 7.8% | 6.7% | 4.1% | 10.2% |
| 20 | 12.2% | 10.9% | 7.5% | 15.2% |
| 50 | 26.2% | 24.1% | 18.6% | 31.2% |
| 100 | 43.9% | 41.4% | 33.5% | 51.3% |
| 200 | 66.1% | 63.7% | 53.6% | 76.3% |

## KL Signal vs Distillation Performance

Cross-reference with actual distillation experiment results (baseline avg@4 = 50.95%):

| Pos Limit | % of KL signal | Best avg@4 | Delta vs baseline | % of max gain captured |
|-----------|----------------|------------|-------------------|------------------------|
| 5 | 5.2% | 56.50% (+5.55) | | 35% |
| 10 | 7.8% | 59.50% (+8.55) | | 54% |
| 20 | 12.2% | 60.20% (+9.25) | | 58% |
| 50 | 26.2% | 62.45% (+11.50) | | 73% |
| 200 | 66.1% | 66.75% (+15.80) | | 100% |

**Key insight:** First 50 tokens contain only 26% of total KL signal but capture 73% of achievable performance gain.

## Distilled vs Raw Model KL Comparison

Per-position KL comparison (100 problems x 1 trajectory, teacher=Qwen3-1.7B):

| Range | Raw | Pos-200tok s100 | Pos-200tok s200 | Full-seq s50 | Full-seq s200 |
|---------|-------|-----------------|-----------------|--------------|---------------|
| 0-50 | 2.064 | 1.015 | 1.015 | 1.109 | 1.036 |
| 50-100 | 0.759 | 0.321 | 0.322 | 0.437 | 0.312 |
| 100-150 | 0.495 | 0.246 | 0.228 | 0.287 | 0.236 |
| 150-200 | 0.387 | 0.237 | 0.248 | 0.314 | 0.236 |
| 200-300 | 0.382 | 0.238 | 0.242 | 0.264 | 0.273 |
| 300-400 | 0.331 | 0.216 | 0.229 | 0.291 | 0.308 |
| 400-500 | 0.317 | 0.289 | 0.277 | 0.232 | 0.202 |
| 500-600 | 0.260 | 0.259 | 0.283 | 0.241 | 0.174 |
| 600-700 | 0.252 | 0.243 | 0.205 | 0.248 | 0.132 |

Key observations:
- **First 200 positions:** all distilled models show ~50-60% lower KL than raw.
- **Pos-200tok step 100 vs step 200:** nearly identical KL profiles, indicating training converges early.
- **Cascade effect confirmed:** 200-400 range KL is also lower for distilled models, even though pos-200tok only trains on first 200 tokens.
- **Full-seq step 200 low KL at tail (500+):** reflects model learning to repeat boxed in teacher-like style.

## Key Findings

1. **Position 0 dominates:** KL=3.057, far above all others. Initial strategy/approach selection is where teacher and student diverge most.
2. **Rapid decay in first 5 tokens:** 3.057 to ~0.92 (3.3x reduction). Once reasoning direction is established, models converge.
3. **Position 3 bump:** secondary peak (KL=1.510) suggests a second decision point early in the sequence, possibly related to problem decomposition or sub-strategy selection.
4. **After position 50, KL stabilizes** around 0.4-0.5 and continues to slowly decrease.
5. **Late-stage slight uptick (positions 25-29):** a minor increase in KL around positions 25-29 could reflect transition points in reasoning structure (e.g., shifting from setup to computation).
6. **Early KL is disproportionately efficient:** 26% of KL signal captures 73% of performance gain (11.50/15.80 pp). Baseline avg@4 = 50.95%.
7. **Position 0 |diff| dominance:** Position 0's |diff| = 8.222 is 10x that of position 100 (0.823) and 14.3x that of position 200 (0.575).

## Implications for Distillation

- **Position-weighted loss** could accelerate alignment by focusing on high-KL early tokens.
- **Curriculum approach:** first get reasoning start right, then fine-tune later distributions.
- **Dead-zone strategies** for mid/late tokens have limited benefit -- student already approximates teacher well there.
- **The first token is critical:** equal-weight loss across positions is sub-optimal.

## Data & Methodology

- **Data source:** qwen3-1.7B-logprobs.jsonl
- **Sample sizes:** 2,000 (small-scale) and 59,936 (large-scale) trajectories
- **Validity rate:** 100% (all student/teacher log-prob lengths match)
- **Analysis range:** first 200 positions
- **Metric:** forward KL divergence between teacher and student distributions at each position
- **Mean sequence length:** 386 tokens
- Every position has 55k+ samples in the large-scale analysis (sufficient statistical power)
- Position 0 analyzed with full 2,000/59,936 samples (maximum coverage for most critical position)
- **Plots:** `kl_position_analysis_v3/kl_comparison_all.png`

## File Paths

- Analysis logs: `logs/math_qwen2.5-1.5B_qwen3-1.7B_kl-analysis/`
- Analysis data: `docs/kl_position_analysis_v2/`, `docs/kl_position_analysis_v3/`
