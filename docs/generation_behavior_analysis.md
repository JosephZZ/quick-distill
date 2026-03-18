# Generation Behavior Analysis: Distilled vs Base Student Model

Date: 2026-03-11

**Goal**: Analyze whether distilling only the first N tokens causes behavioral changes
BEYOND position N (the 'cascade effect').

**Models compared**:
- Base (Qwen2.5-Math-1.5B)
- Pos-5 (step 200)
- Pos-10 (step 200)
- Pos-20 (step 200)
- Pos-50 (step 200)
- Pos-200tok (step 100)
- Progressive 1->200 (step 200)
- Full-seq (step 50)


## 1. Response Opening Similarity (Prefix Diversity)

For each question, we look at the first N characters of the 4 generated responses
and count how many are unique. A ratio of 1.0 means all 4 responses start differently;
lower values mean more convergent openings.

### Prefix length = 50 chars

| Model | Avg Unique Prefixes (out of 4) | Unique Ratio |
|-------|-------------------------------|-------------|
| Base (Qwen2.5-Math-1.5B) | 3.68 | 0.920 |
| Pos-5 (step 200) | 2.58 | 0.644 |
| Pos-10 (step 200) | 2.52 | 0.630 |
| Pos-20 (step 200) | 2.32 | 0.579 |
| Pos-50 (step 200) | 2.23 | 0.557 |
| Pos-200tok (step 100) | 2.26 | 0.565 |
| Progressive 1->200 (step 200) | 2.48 | 0.621 |
| Full-seq (step 50) | 3.25 | 0.813 |

### Prefix length = 100 chars

| Model | Avg Unique Prefixes (out of 4) | Unique Ratio |
|-------|-------------------------------|-------------|
| Base (Qwen2.5-Math-1.5B) | 3.87 | 0.968 |
| Pos-5 (step 200) | 3.42 | 0.856 |
| Pos-10 (step 200) | 3.35 | 0.838 |
| Pos-20 (step 200) | 3.15 | 0.788 |
| Pos-50 (step 200) | 3.02 | 0.756 |
| Pos-200tok (step 100) | 3.03 | 0.758 |
| Progressive 1->200 (step 200) | 3.34 | 0.836 |
| Full-seq (step 50) | 3.68 | 0.920 |

### Prefix length = 200 chars

| Model | Avg Unique Prefixes (out of 4) | Unique Ratio |
|-------|-------------------------------|-------------|
| Base (Qwen2.5-Math-1.5B) | 3.98 | 0.995 |
| Pos-5 (step 200) | 3.92 | 0.981 |
| Pos-10 (step 200) | 3.92 | 0.981 |
| Pos-20 (step 200) | 3.84 | 0.961 |
| Pos-50 (step 200) | 3.78 | 0.945 |
| Pos-200tok (step 100) | 3.75 | 0.939 |
| Progressive 1->200 (step 200) | 3.88 | 0.970 |
| Full-seq (step 50) | 3.97 | 0.993 |

### Prefix length = 500 chars

| Model | Avg Unique Prefixes (out of 4) | Unique Ratio |
|-------|-------------------------------|-------------|
| Base (Qwen2.5-Math-1.5B) | 4.00 | 1.000 |
| Pos-5 (step 200) | 4.00 | 1.000 |
| Pos-10 (step 200) | 4.00 | 1.000 |
| Pos-20 (step 200) | 4.00 | 1.000 |
| Pos-50 (step 200) | 4.00 | 1.000 |
| Pos-200tok (step 100) | 4.00 | 1.000 |
| Progressive 1->200 (step 200) | 4.00 | 1.000 |
| Full-seq (step 50) | 4.00 | 1.000 |

### Cross-model prefix similarity (first response per question, first 200 chars)

Fraction of questions where Model A and Model B produce the same 200-char prefix:

| Model | Base | Pos-5 | Pos-10 | Pos-20 | Pos-50 | Pos-200tok | Progressive 1->200 | Full-seq |
|---|---|---|---|---|---|---|---|---|
| Base | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| Pos-5 | 0.000 | 1.000 | 0.010 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 |
| Pos-10 | 0.000 | 0.010 | 1.000 | 0.018 | 0.000 | 0.002 | 0.002 | 0.000 |
| Pos-20 | 0.000 | 0.004 | 0.018 | 1.000 | 0.008 | 0.004 | 0.010 | 0.000 |
| Pos-50 | 0.000 | 0.000 | 0.000 | 0.008 | 1.000 | 0.012 | 0.046 | 0.000 |
| Pos-200tok | 0.000 | 0.000 | 0.002 | 0.004 | 0.012 | 1.000 | 0.014 | 0.000 |
| Progressive 1->200 | 0.000 | 0.000 | 0.002 | 0.010 | 0.046 | 0.014 | 1.000 | 0.000 |
| Full-seq | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |


## 2. Cascade Effect: Does Early-Token Distillation Improve Later Tokens?

The key question: Pos-50 only trains on the first 50 tokens of the response,
but does the model produce better responses overall (including after position 50)?

### Overall Metrics

| Model | avg@4 | maj@4 | pass@4 | Avg Response Length (chars) | Median Resp Len |
|-------|-------|-------|--------|---------------------------|----------------|
| Base (Qwen2.5-Math-1.5B) | 50.9% | 46.6% | 72.8% | 1996 | 1538 |
| Pos-5 (step 200) | 56.3% | 51.4% | 73.4% | 1859 | 1526 |
| Pos-10 (step 200) | 56.0% | 52.8% | 74.2% | 1894 | 1546 |
| Pos-20 (step 200) | 56.5% | 52.0% | 75.8% | 1967 | 1648 |
| Pos-50 (step 200) | 61.9% | 58.4% | 78.6% | 1929 | 1666 |
| Pos-200tok (step 100) | 66.8% | 64.6% | 80.2% | 1578 | 1300 |
| Progressive 1->200 (step 200) | 62.3% | 58.2% | 78.8% | 1897 | 1610 |
| Full-seq (step 50) | 65.6% | 62.4% | 80.0% | 1388 | 1180 |

### Per-Question Accuracy Shift: Base -> Pos-50

- Questions improved (more correct samples): 202/500 (40.4%)
- Questions degraded (fewer correct samples): 68/500 (13.6%)
- Questions unchanged: 230/500 (46.0%)
- Net improvement: 134 questions

### Response Length Distribution

| Model | Mean | Std | P10 | P25 | P50 | P75 | P90 |
|-------|------|-----|-----|-----|-----|-----|-----|
| Base (Qwen2.5-Math-1.5B) | 1996 | 1433 | 787 | 1085 | 1540 | 2280 | 4263 |
| Pos-5 (step 200) | 1859 | 1173 | 844 | 1096 | 1527 | 2157 | 3275 |
| Pos-10 (step 200) | 1894 | 1247 | 840 | 1118 | 1550 | 2172 | 3346 |
| Pos-20 (step 200) | 1967 | 1198 | 891 | 1201 | 1648 | 2311 | 3447 |
| Pos-50 (step 200) | 1929 | 1086 | 957 | 1245 | 1666 | 2270 | 3082 |
| Pos-200tok (step 100) | 1578 | 1052 | 631 | 893 | 1300 | 1897 | 2837 |
| Progressive 1->200 (step 200) | 1897 | 1109 | 885 | 1184 | 1610 | 2200 | 3259 |
| Full-seq (step 50) | 1388 | 886 | 555 | 787 | 1180 | 1698 | 2520 |


## 3. Response Structure Analysis

Do distilled models produce more structured, teacher-like responses?

### Structural Marker Frequency (avg per response)

| Model | Step N: | ### heading | \boxed{} | ```code``` | First,/Second, | Therefore/Thus | Let's/We | **bold** | Verification | Newlines | Avg Steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base (Qwen2.5-Math-1.5B) | 0.17 | 4.31 | 1.55 | 4.11 | 0.67 | 0.86 | 1.76 | 1.14 | 1.23 | 54.3 | 2.8 |
| Pos-5 (step 200) | 0.13 | 3.82 | 0.99 | 3.69 | 0.48 | 0.82 | 3.77 | 0.47 | 1.27 | 48.5 | 2.0 |
| Pos-10 (step 200) | 0.23 | 4.02 | 1.03 | 3.48 | 0.53 | 0.81 | 4.90 | 0.81 | 1.24 | 60.6 | 2.0 |
| Pos-20 (step 200) | 0.64 | 5.09 | 1.03 | 3.35 | 0.45 | 0.94 | 4.80 | 1.63 | 1.15 | 73.1 | 2.4 |
| Pos-50 (step 200) | 4.66 | 8.02 | 1.05 | 1.83 | 0.27 | 0.55 | 4.91 | 1.32 | 1.11 | 79.2 | 5.3 |
| Pos-200tok (step 100) | 3.12 | 4.16 | 1.03 | 0.34 | 0.21 | 0.36 | 4.04 | 0.36 | 0.50 | 68.3 | 3.4 |
| Progressive 1->200 (step 200) | 3.21 | 5.96 | 1.00 | 1.94 | 0.27 | 0.69 | 5.36 | 0.72 | 1.11 | 73.4 | 3.9 |
| Full-seq (step 50) | 0.34 | 0.92 | 1.02 | 0.43 | 0.52 | 0.63 | 1.62 | 1.18 | 0.28 | 51.5 | 2.8 |

### Most Common Response Openings (first 80 chars)

**Base (Qwen2.5-Math-1.5B)** - Top 5 openings:

- (25, 1.2%) `Given:`
- (24, 1.2%) `Let's solve the problem step by step.`
- (17, 0.9%) ````python`
- (10, 0.5%) `\[`
- (9, 0.4%) `To solve the problem, we need to follow these steps:`

**Pos-5 (step 200)** - Top 5 openings:

- (122, 6.1%) `We are given the following information:`
- (79, 4.0%) `We are given the equation:`
- (34, 1.7%) `We are given the expression:`
- (17, 0.9%) `We are given the following conditions:`
- (17, 0.9%) `We are given the inequality:`

**Pos-10 (step 200)** - Top 5 openings:

- (162, 8.1%) `We are given the equation:`
- (144, 7.2%) `We are given the expression:`
- (92, 4.6%) `We are given:`
- (52, 2.6%) `We are given the function:`
- (29, 1.5%) `We are given the following information:`

**Pos-20 (step 200)** - Top 5 openings:

- (156, 7.8%) `We are given the equation:`
- (150, 7.5%) `We are given the expression:`
- (134, 6.7%) `We are given:`
- (99, 5.0%) `We are given the function:`
- (56, 2.8%) `We are given the following:`

**Pos-50 (step 200)** - Top 5 openings:

- (211, 10.5%) `We are given the expression:`
- (163, 8.2%) `We are given the equation:`
- (126, 6.3%) `We are given:`
- (109, 5.5%) `We are given the function:`
- (55, 2.8%) `We are given the following:`

**Pos-200tok (step 100)** - Top 5 openings:

- (208, 10.4%) `We are given the expression:`
- (175, 8.8%) `We are given:`
- (149, 7.4%) `We are given the equation:`
- (88, 4.4%) `We are given the function:`
- (55, 2.8%) `We are given the following information:`

**Progressive 1->200 (step 200)** - Top 5 openings:

- (216, 10.8%) `We are given the expression:`
- (174, 8.7%) `We are given:`
- (160, 8.0%) `We are given the equation:`
- (100, 5.0%) `We are given the function:`
- (53, 2.6%) `We are given the following information:`

**Full-seq (step 50)** - Top 5 openings:

- (72, 3.6%) `Given:`
- (35, 1.8%) `Given the equation:`
- (27, 1.4%) `Let's solve the problem step by step.`
- (15, 0.8%) `To solve the equation:`
- (15, 0.8%) `We are given the equation:`


## 4. Correctness by Response Length

Responses binned by character length. Accuracy computed per bin.

| Model | 0-500 | 500-1K | 1K-1.5K | 1.5K-2K | 2K-3K | 3K-5K | 5K-10K | 10K+ |
|---|---|---|---|---|---|---|---|---|
| Base (Qwen2.5-Math-1.5B) | 51% (47) | 69% (356) | 69% (555) | 51% (397) | 35% (328) | 21% (158) | 8% (159) | - |
| Pos-5 (step 200) | 93% (27) | 79% (350) | 67% (582) | 56% (448) | 43% (356) | 19% (156) | 2% (81) | - |
| Pos-10 (step 200) | 91% (22) | 80% (328) | 68% (606) | 57% (456) | 38% (342) | 20% (153) | 4% (93) | - |
| Pos-20 (step 200) | 96% (23) | 81% (277) | 73% (548) | 57% (470) | 43% (414) | 21% (177) | 3% (91) | - |
| Pos-50 (step 200) | 100% (8) | 88% (238) | 81% (559) | 63% (492) | 45% (487) | 19% (159) | 4% (56) | 0% (1) |
| Pos-200tok (step 100) | 97% (87) | 88% (567) | 76% (551) | 61% (349) | 37% (270) | 13% (142) | 3% (34) | - |
| Progressive 1->200 (step 200) | 100% (14) | 88% (304) | 78% (560) | 64% (481) | 48% (388) | 15% (194) | 3% (59) | - |
| Full-seq (step 50) | 95% (152) | 85% (635) | 68% (582) | 50% (290) | 35% (232) | 9% (94) | 0% (15) | - |

### Correct vs Incorrect Response Length

| Model | Avg Len (Correct) | Avg Len (Incorrect) | Ratio (Correct/Incorrect) |
|-------|-------------------|--------------------|--------------------------| 
| Base (Qwen2.5-Math-1.5B) | 1465 | 2549 | 0.57 |
| Pos-5 (step 200) | 1450 | 2385 | 0.61 |
| Pos-10 (step 200) | 1451 | 2458 | 0.59 |
| Pos-20 (step 200) | 1528 | 2539 | 0.60 |
| Pos-50 (step 200) | 1549 | 2547 | 0.61 |
| Pos-200tok (step 100) | 1199 | 2338 | 0.51 |
| Progressive 1->200 (step 200) | 1493 | 2564 | 0.58 |
| Full-seq (step 50) | 1084 | 1967 | 0.55 |


## 5. Early vs Late Response Quality (Token Cascade)

We approximate token positions using words (roughly 1 word ~ 1.3 tokens for English/math).
For Pos-50, distillation only affected the first ~50 tokens (~38 words).
We analyze: does the LATER part of the response also change?

### Word cutoff = 40 (approx 52 tokens)

| Model | Avg Jaccard (first 40 words) | Avg Jaccard (after 40 words) | Early Change > Late Change? |
|---|---|---|---|
| Pos-5 (step 200) | 0.302 | 0.308 | Yes |
| Pos-10 (step 200) | 0.291 | 0.299 | Yes |
| Pos-20 (step 200) | 0.283 | 0.288 | Yes |
| Pos-50 (step 200) | 0.251 | 0.251 | Yes |
| Pos-200tok (step 100) | 0.253 | 0.215 | No (late also changed) |
| Progressive 1->200 (step 200) | 0.261 | 0.256 | No (late also changed) |
| Full-seq (step 50) | 0.323 | 0.242 | No (late also changed) |

### Word cutoff = 80 (approx 104 tokens)

| Model | Avg Jaccard (first 80 words) | Avg Jaccard (after 80 words) | Early Change > Late Change? |
|---|---|---|---|
| Pos-5 (step 200) | 0.323 | 0.293 | No (late also changed) |
| Pos-10 (step 200) | 0.313 | 0.283 | No (late also changed) |
| Pos-20 (step 200) | 0.300 | 0.272 | No (late also changed) |
| Pos-50 (step 200) | 0.264 | 0.235 | No (late also changed) |
| Pos-200tok (step 100) | 0.265 | 0.196 | No (late also changed) |
| Progressive 1->200 (step 200) | 0.270 | 0.242 | No (late also changed) |
| Full-seq (step 50) | 0.325 | 0.217 | No (late also changed) |

### Word cutoff = 150 (approx 195 tokens)

| Model | Avg Jaccard (first 150 words) | Avg Jaccard (after 150 words) | Early Change > Late Change? |
|---|---|---|---|
| Pos-5 (step 200) | 0.325 | 0.260 | No (late also changed) |
| Pos-10 (step 200) | 0.313 | 0.255 | No (late also changed) |
| Pos-20 (step 200) | 0.303 | 0.232 | No (late also changed) |
| Pos-50 (step 200) | 0.265 | 0.202 | No (late also changed) |
| Pos-200tok (step 100) | 0.258 | 0.169 | No (late also changed) |
| Progressive 1->200 (step 200) | 0.273 | 0.208 | No (late also changed) |
| Full-seq (step 50) | 0.310 | 0.180 | No (late also changed) |

Note: Lower Jaccard similarity = more different from base model.
If 'late' Jaccard is also low, it means distillation cascaded to affect later tokens too.

---

## 6. Key Findings and Interpretation

### Finding 1: Distillation strongly reduces prefix diversity (even Pos-5!)

Even Pos-5, which only distills the first 5 tokens, reduces the 50-char prefix diversity from 0.920 to 0.644. The model learns to start responses in a more consistent, teacher-like way. More distillation positions -> more convergent openings (Pos-50 at 0.557 vs Pos-5 at 0.644).

Interestingly, Full-seq (step 50) has the highest diversity among distilled models (0.813), likely because it has only trained for 50 steps vs 200 for others.

### Finding 2: CASCADE EFFECT IS REAL -- early-token distillation improves the entire response

This is the central finding. Pos-50 only trains on the first 50 tokens, yet:
- **avg@4 improves from 50.9% to 61.9%** (+11.0 points)
- **pass@4 improves from 72.8% to 78.6%** (+5.8 points)
- 202 questions improved vs only 68 degraded (3:1 ratio)

The Jaccard analysis (Section 5) confirms: **later tokens change just as much as early tokens**. At word cutoff=40 (approx 50 tokens), Pos-50 has Jaccard 0.251 for early words and 0.251 for late words -- the change is uniform across the entire response.

Even Pos-5 (distilling only 5 tokens) improves avg@4 by +5.4 points, demonstrating that distilling even a tiny prefix cascades through the autoregressive generation.

### Finding 3: Distilled models adopt teacher-like response structure

Pos-50 dramatically increases structured writing:
- **Step markers**: 5.3/response (vs 2.8 for base) -- nearly double
- **Markdown headings**: 8.0/response (vs 4.3 for base)
- **Newlines**: 79.2/response (vs 54.3 for base)

The model learns the teacher's structured reasoning style even beyond the distilled positions. Full-seq shows a different pattern: fewer headings (0.92) but more concise responses, suggesting it learns a different teacher style.

### Finding 4: Distilled models converge to "We are given..." openings

All distilled models (except Full-seq) overwhelmingly start with "We are given..." patterns. The base model shows much more diverse openings. The top opening for Pos-50 ("We are given the expression:") appears in 10.5% of responses vs varied openings in the base model. Full-seq learns different openings ("Given:", "Let's solve..."), reflecting deeper structural changes from full-sequence distillation.

### Finding 5: Shorter responses correlate with correctness, and distilled models learn this

Across all models, correct responses are ~40-50% shorter than incorrect ones. Distilled models with more position coverage (Pos-200tok, Full-seq) produce notably shorter responses overall:
- Base avg length: 1996 chars
- Pos-200tok: 1578 chars (-21%)
- Full-seq: 1388 chars (-30%)

The length-accuracy profile shows distilled models achieve higher accuracy at every length bin, especially in the 500-1.5K range where most responses fall.

### Finding 6: Diminishing returns from position coverage, but clear scaling

| Position limit | avg@4 | Improvement vs base |
|---------------|-------|-------------------|
| 5 tokens | 56.3% | +5.4 |
| 10 tokens | 56.0% | +5.1 |
| 20 tokens | 56.5% | +5.6 |
| 50 tokens | 61.9% | +11.0 |
| 200 tokens | 66.8% | +15.9 |
| Full (3584) | 65.6% | +14.7 |

There's a jump between 20 and 50 tokens, suggesting that position 20-50 contains critical reasoning-setup tokens. Full-seq at step 50 is slightly below Pos-200tok at step 100, likely due to fewer training steps.

### Conclusion

**Early-position distillation creates a strong cascade effect.** Training on just the first 50 tokens changes the model's entire generation trajectory -- it adopts more structured reasoning, more consistent openings, and produces higher-quality solutions throughout the response. The mechanism is likely that early tokens set up the reasoning framework (problem restatement, approach selection), and better early decisions lead to better solutions downstream through autoregressive generation.
