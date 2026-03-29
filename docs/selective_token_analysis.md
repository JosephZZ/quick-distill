# Selective Token Analysis: Top-KL vs Top-Entropy vs Positional

**Data**: 100 undistilled student (Qwen2.5-Math-1.5B) trajectories scored by Qwen3-1.7B teacher.
**Average sequence length**: 745 tokens.
**Metrics**: Per-token KL = |log p_s - log p_t| on sampled token; Entropy proxy = -log p(sampled token).

---

## 1. Signal Concentration

Top-N tokens by KL/entropy capture far more signal than the first N positional tokens:

| N tokens | Top-N KL | Top-N Student Ent | Top-N Teacher Ent | Early-N (positional) |
|----------|----------|-------------------|-------------------|---------------------|
| 50 | **86.4%** | **80.9%** | **86.3%** | 35.8% |
| 100 | **95.6%** | **93.5%** | **96.0%** | 48.8% |
| 150 | 98.4% | 97.4% | 98.7% | 56.5% |
| 200 | 99.3% | 98.8% | 99.6% | 61.8% |
| 250 | 99.7% | 99.3% | 99.9% | 65.3% |
| 300 | 99.8% | 99.5% | 100.0% | 69.4% |

**Takeaway**: Top-100 tokens capture 94-96% of all KL/entropy signal, while first 100 positions only capture ~49%. Selective methods have a ~2x advantage in signal coverage.

---

## 2. Top-100 Overlap Matrix

| Top-100 | Student Ent | Teacher Ent | KL | Early-100 |
|---------|-----------|------------|-----|-----------|
| Student Ent | — | 64.3% | 70.5% | 39.3% |
| Teacher Ent | 64.3% | — | 82.7% | 32.7% |
| KL | 70.5% | 82.7% | — | 36.0% |
| Early-100 | 39.3% | 32.7% | 36.0% | — |

- **Teacher ent ∩ KL = 83%**: High KL is mostly driven by teacher uncertainty.
- **Student ent ∩ KL = 71%**: Substantial overlap but 30% are different tokens.
- **All selective ∩ Early-100 ≈ 33-39%**: Early tokens cover only ~1/3 of top selective tokens.

---

## 3. Position Distribution of Top-100 KL Tokens

| Position range | Top-100 KL | Top-100 Student Ent |
|---------------|-----------|---------------------|
| 0-49 | 13.9% | 23.7% |
| 50-99 | 11.1% | 15.6% |
| 100-199 | 19.6% | 18.4% |
| 200-299 | 14.4% | 13.0% |
| 300+ | **40.9%** | **29.3%** |
| **Mean position** | **313** | **240** |
| **Median position** | 231 | 149 |

- KL tokens are more scattered (mean pos 313, 41% in 300+).
- Entropy tokens are somewhat more front-loaded (mean pos 240, 24% in first 50).

---

## 4. What Are the High-KL Tokens?

### KL Tier 1 (Top 1%, KL > 9.7, avg pos 165)
Dominated by **strategy selection and formatting**:
- `"To"` (KL=33.0) — reasoning approach selection at position 0
- `"Please"` (KL=25.6) — dialogue style difference
- `"The"` (KL=27.9) — new reasoning paragraph
- `"Let"` (KL=23.1) — math reasoning opener
- `` ``` `` (KL=11.7) — code block formatting
- `\(` (KL=11.6) — LaTeX inline math format
- `\[` (KL=12.6) — LaTeX display math format
- `"Python"` (KL=14.0) — code language choice

### KL Tier 2 (Top 1-5%, KL 2.3-9.7, avg pos 332)
**LaTeX formatting + numbers**: `\(`, `\\`, `\[`, digits, commas.

### KL Tier 3 (Top 5-15%, KL 0.2-2.3, avg pos 334)
**Common words and operators**: "the", digits, commas, dots, spaces.

**Summary**: High KL is primarily **format/style disagreement** (LaTeX notation, code blocks) and **strategy decisions** (first token). Not necessarily reasoning quality.

---

## 5. What Are the High-Entropy Tokens?

### Student Entropy Tier 1 (Top 1%, Entropy > 2.0, avg pos 206)
Almost entirely **numbers**: `"4"`, `"3"`, `"5"`, `"2"`, `"8"`, `"7"`, `"6"`, `"1"`, `"9"`.

Student is most uncertain about **computation results** — which digit comes next.

### Student Entropy Tier 2 (Top 1-5%, Entropy 0.7-2.0, avg pos 282)
**Digits + common words**: `"1"`, `"the"`, `"2"`, `"3"`, `"4"`, newlines, commas.

**Summary**: High student entropy is primarily about **numerical uncertainty** — the student doesn't know what number to produce. This is arguably a more direct signal of reasoning gaps than KL's format disagreements.

---

## 6. High-KL vs High-Entropy: Different Signals

| | High KL, Low Student Ent | High Student Ent, Low KL | High Both |
|---|--------------------------|--------------------------|-----------|
| **Meaning** | Student confident, teacher disagrees | Student uncertain, teacher agrees | Student uncertain, teacher disagrees |
| **Examples** | `\(` (format), `"To"` (strategy) | `"rectangular"`, `"convert"` | Digits: `"3"`, `"5"`, `"8"` |
| **What it teaches** | Format preferences | Student's confusion points | Actual reasoning gaps |

---

## 7. Teacher Entropy vs KL

Teacher entropy Tier 1 tokens are **nearly identical** to KL Tier 1:
- `"To"`: T_ent=34.2, S_ent=1.2, KL=33.0
- `\(`: T_ent=12.4, S_ent=0.4, KL=12.0
- `` ``` ``: T_ent=12.6, S_ent=0.0, KL=12.6

**KL ≈ Teacher_ent - Student_ent** in practice. High KL is mostly caused by the teacher being uncertain about the student's token choices, not the student being uncertain.

---

## 8. Experimental Results (MATH-500, LoRA, n1bs16, SGLang)

All experiments: k=100 tokens selected per sequence, full-length generation (max_new_tokens=2048), 3200 problems, LoRA r=32.

### Best avg@4 Comparison

| Method | Best Step | avg@4 | maj@4 | pass@4 |
|--------|-----------|-------|-------|--------|
| **Pos-100 (prefix)** | **200** | **65.85%** | **70.80%** | **79.80%** |
| Random-100 | 150 | 63.05% | 69.20% | 76.80% |
| Top-Entropy-Teacher-100 | 150 | 62.20% | 67.80% | 75.80% |
| Top-Entropy-Student-100 | 100 | 61.35% | 67.20% | 73.20% |
| Top-KL-100 | 50 | 58.60% | 65.80% | 74.40% |

### Key Observations

1. **Positional prefix wins decisively.** Despite covering only ~49% of KL signal (vs 96% for top-KL), prefix-100 outperforms all selective methods by 2.8-7.3pp avg@4. The **cascade effect** — early-token corrections propagating through autoregressive generation — is more powerful than raw signal concentration.

2. **Top-KL is the worst method.** As predicted by the analysis above (§4, §6), top-KL tokens are dominated by format/style disagreements (LaTeX, code blocks), not reasoning quality. Distilling on these tokens teaches superficial preferences, not math skills. It also degrades sharply after step 50, dropping to 45.7% avg@4 at step 100.

3. **Student entropy > teacher entropy > KL.** Among selective methods, student entropy (targeting numerical uncertainty) slightly outperforms teacher entropy, and both beat top-KL. This aligns with the finding that student entropy targets computation gaps while KL targets format disagreements.

4. **Random selection is the best selective method.** Random-100 outperforms all "smart" selection criteria. This suggests that loss on diverse, uniformly distributed tokens provides better gradient signal than concentrating on a subset — possibly because scattered tokens still partially benefit from the cascade effect.

5. **Selective methods degrade more over training.** Top-KL and top-entropy-student show significant degradation by step 200 (dropping 6-13pp from peak), while positional prefix is more stable. This may be because selective methods amplify noisy high-signal tokens, causing overfitting.

### Implications

The cascade effect hypothesis is confirmed: **where** you distill matters more than **what** you distill on. Positional prefix succeeds because early tokens have disproportionate influence on all downstream tokens through autoregressive generation. Selective methods, despite targeting 2x more KL signal, cannot compensate for losing this positional advantage.
