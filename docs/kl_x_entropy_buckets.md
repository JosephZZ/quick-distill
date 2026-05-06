# KL × Entropy(surprise) Bucket Analysis

Data: /zhi_backup/ziheng/quick-distillation/docs/kl_position_analysis_v2/raw_logprobs.jsonl, total tokens: 74549
KL stats:        median=0.001  p75=0.013  max=45.709
Surprise stats:  median=0.000  p75=0.006  max=10.764
(surprise = -log p(sampled token | student); proxy for distribution entropy)

## Bucket sizes (split at KL>p75 and surprise>p75)

| Bucket | n | % of total |
|---|---:|---:|
| hiKL_hiE | 15571 | 20.9% |
| hiKL_loE | 3066 | 4.1% |
| loKL_hiE | 3066 | 4.1% |
| loKL_loE | 52846 | 70.9% |

## Category composition per bucket

| Bucket | planning | structural | math_latex | math_operator | math_number | continuation | mean KL | mean surp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **hiKL_hiE** | 12.2% | 28.7% | 4.3% | 4.1% | 10.7% | 40.0% | 1.81 | 0.50 |
| **hiKL_loE** | 10.2% | 30.6% | 17.8% | 2.1% | 4.1% | 35.3% | 1.65 | 0.00 |
| **loKL_hiE** | 6.1% | 49.9% | 1.6% | 3.8% | 7.4% | 31.1% | 0.01 | 0.03 |
| **loKL_loE** | 4.4% | 33.7% | 3.7% | 8.3% | 16.1% | 33.8% | 0.00 | 0.00 |

## Top-frequency tokens within high-KL buckets

### hiKL_hiE  (n=15571)

| Token | count | % of bucket | mean KL | mean surp | category |
|---|---:|---:|---:|---:|---|
| '\n' | 632 | 4.1% | 0.57 | 0.17 | structural |
| ' the' | 573 | 3.7% | 0.62 | 0.36 | planning |
| ' ' | 338 | 2.2% | 0.55 | 0.27 | structural |
| '1' | 337 | 2.2% | 1.37 | 0.58 | math_number |
| '2' | 300 | 1.9% | 1.10 | 0.67 | math_number |
| ' \\(' | 288 | 1.8% | 5.93 | 0.30 | math_latex |
| ' \\' | 274 | 1.8% | 1.64 | 0.36 | structural |
| ',' | 272 | 1.7% | 1.09 | 0.40 | structural |
| ' $' | 234 | 1.5% | 1.08 | 0.41 | structural |
| '3' | 220 | 1.4% | 1.64 | 0.72 | math_number |
| '.' | 211 | 1.4% | 0.95 | 0.36 | structural |
| '0' | 199 | 1.3% | 0.81 | 0.44 | math_number |
| ' is' | 198 | 1.3% | 0.47 | 0.36 | continuation |
| '\\' | 188 | 1.2% | 1.55 | 0.38 | structural |
| ' -' | 170 | 1.1% | 1.00 | 0.43 | math_operator |
| ' of' | 165 | 1.1% | 0.98 | 0.35 | continuation |
| '5' | 147 | 0.9% | 1.47 | 0.81 | math_number |
| '4' | 146 | 0.9% | 1.23 | 1.02 | math_number |
| ' and' | 133 | 0.9% | 1.41 | 0.44 | continuation |
| ' we' | 126 | 0.8% | 1.03 | 0.39 | planning |
| ' =' | 119 | 0.8% | 0.93 | 0.28 | math_operator |
| ' to' | 115 | 0.7% | 0.91 | 0.43 | planning |
| ' +' | 112 | 0.7% | 0.75 | 0.52 | math_operator |
| ' a' | 108 | 0.7% | 1.00 | 0.46 | continuation |
| '?' | 106 | 0.7% | 0.62 | 0.23 | structural |
| 'x' | 104 | 0.7% | 0.82 | 0.40 | continuation |
| ' can' | 99 | 0.6% | 1.18 | 0.40 | continuation |
| 'The' | 95 | 0.6% | 4.93 | 0.60 | planning |
| ').' | 94 | 0.6% | 1.05 | 0.36 | structural |
| '6' | 93 | 0.6% | 2.01 | 0.76 | math_number |

### hiKL_loE  (n=3066)

| Token | count | % of bucket | mean KL | mean surp | category |
|---|---:|---:|---:|---:|---|
| ' \\(' | 370 | 12.1% | 4.30 | 0.00 | math_latex |
| ' the' | 131 | 4.3% | 0.21 | 0.00 | planning |
| ' \\' | 110 | 3.6% | 0.86 | 0.00 | structural |
| '\\' | 93 | 3.0% | 1.04 | 0.00 | structural |
| '```' | 87 | 2.8% | 6.06 | 0.00 | structural |
| '`\n' | 83 | 2.7% | 0.59 | 0.00 | structural |
| '\\[' | 80 | 2.6% | 2.87 | 0.00 | math_latex |
| ' of' | 56 | 1.8% | 0.20 | 0.00 | continuation |
| "'s" | 51 | 1.7% | 1.21 | 0.00 | structural |
| ',' | 49 | 1.6% | 1.11 | 0.00 | structural |
| '(\\' | 44 | 1.4% | 4.72 | 0.00 | structural |
| '#' | 42 | 1.4% | 1.18 | 0.00 | structural |
| 'output' | 42 | 1.4% | 2.87 | 0.00 | continuation |
| '\n' | 41 | 1.3% | 2.16 | 0.00 | structural |
| 'The' | 41 | 1.3% | 0.65 | 0.00 | planning |
| ' we' | 36 | 1.2% | 0.32 | 0.00 | planning |
| '1' | 35 | 1.1% | 1.76 | 0.00 | math_number |
| '.Receive' | 33 | 1.1% | 0.60 | 0.00 | continuation |
| '\\)' | 31 | 1.0% | 0.70 | 0.00 | math_latex |
| ' oud' | 28 | 0.9% | 0.48 | 0.00 | continuation |
| ' ' | 27 | 0.9% | 1.09 | 0.00 | structural |
| '逄' | 26 | 0.8% | 5.90 | 0.00 | continuation |
| ' Python' | 25 | 0.8% | 5.75 | 0.00 | continuation |
| 'print' | 25 | 0.8% | 2.29 | 0.00 | math_latex |
| ' Propel' | 25 | 0.8% | 3.03 | 0.00 | continuation |
| ' and' | 24 | 0.8% | 0.33 | 0.00 | continuation |
| ' this' | 21 | 0.7% | 0.62 | 0.00 | continuation |
| '.' | 21 | 0.7% | 0.56 | 0.00 | structural |
| '2' | 20 | 0.7% | 0.54 | 0.00 | math_number |
| ' to' | 19 | 0.6% | 0.50 | 0.00 | planning |

## Reading guide

- **hiKL_hiE** = student is uncertain AND disagrees with teacher → genuine 'pivot' positions where reasoning happens
- **hiKL_loE** = student is confident but wrong (per teacher) → 'overconfident format / habit' positions; this is where prefix-100 likely helps most
- **loKL_hiE** = student uncertain but teacher agrees with the sample → coverage tokens, low gradient
- **loKL_loE** = both agree, both confident → easy positions, dropping these costs little