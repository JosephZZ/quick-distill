# Token-Level KL Divergence Classification Analysis

Data: `/CGLab/ziheng/projects/dft-distill/output/qwen3-1.7B-logprobs.jsonl`
Trajectories analyzed: 10000
Tokenizer: Qwen/Qwen2.5-Math-1.5B

## 1. Position 0 Deep Dive

Position 0 has KL=8.2 which is ~10x higher than other positions. What tokens appear here?

Total trajectories with position 0: 10000
Mean KL at position 0: 7.7052
Median KL at position 0: 7.4836
Unique tokens at position 0: 101

### Top 30 tokens at position 0 (by frequency)

| Token | Count | % | Mean KL | Median KL | Category |
|-------|-------|---|---------|-----------|----------|
| 'To' | 7510 | 75.1% | 7.32 | 7.52 | planning |
| 'Let' | 1005 | 10.1% | 1.60 | 1.09 | planning |
| 'First' | 342 | 3.4% | 14.03 | 13.86 | planning |
| 'Step' | 208 | 2.1% | 14.60 | 14.30 | planning |
| 'The' | 179 | 1.8% | 13.84 | 15.39 | planning |
| '**' | 140 | 1.4% | 18.61 | 18.76 | structural |
| 'We' | 115 | 1.1% | 2.08 | 2.01 | planning |
| 'There' | 73 | 0.7% | 19.29 | 19.91 | continuation |
| 'He' | 67 | 0.7% | 21.24 | 21.53 | continuation |
| 'Given' | 52 | 0.5% | 5.60 | 5.78 | planning |
| 'Each' | 18 | 0.2% | 10.15 | 13.63 | continuation |
| 'For' | 12 | 0.1% | 18.69 | 18.57 | planning |
| 'T' | 11 | 0.1% | 11.35 | 11.67 | continuation |
| 'Bl' | 11 | 0.1% | 3.30 | 3.31 | continuation |
| 'Rachel' | 10 | 0.1% | 8.26 | 12.75 | continuation |
| 'Carl' | 10 | 0.1% | 11.94 | 11.43 | continuation |
| '1' | 10 | 0.1% | 27.50 | 26.87 | math_number |
| 'Maria' | 8 | 0.1% | 2.63 | 2.58 | continuation |
| 'Sam' | 8 | 0.1% | 0.85 | 0.66 | continuation |
| 'Pa' | 8 | 0.1% | 9.17 | 9.10 | continuation |
| 'Gr' | 7 | 0.1% | 23.41 | 23.32 | continuation |
| 'Jo' | 7 | 0.1% | 2.86 | 2.21 | continuation |
| 'Jul' | 7 | 0.1% | 1.53 | 1.70 | continuation |
| 'Since' | 7 | 0.1% | 20.59 | 20.88 | planning |
| 'B' | 6 | 0.1% | 2.90 | 1.13 | continuation |
| 'Brit' | 6 | 0.1% | 3.31 | 3.33 | continuation |
| 'At' | 6 | 0.1% | 18.60 | 19.09 | continuation |
| 'Solution' | 6 | 0.1% | 27.12 | 26.75 | planning |
| 'Mary' | 5 | 0.1% | 3.32 | 3.32 | continuation |
| 'They' | 5 | 0.1% | 24.74 | 25.39 | continuation |

### Category breakdown at position 0

| Category | Count | % | Mean KL |
|----------|-------|---|---------|
| planning | 9438 | 94.4% | 7.20 |
| continuation | 409 | 4.1% | 15.00 |
| structural | 143 | 1.4% | 18.59 |
| math_number | 10 | 0.1% | 27.50 |

## 2. Token Content at High-KL Positions (0-49)

What tokens appear at each of the first 20 positions?

**Position 0** (mean KL=7.71, n=10000): `'To'` (7510/10000), `'Let'` (1005/10000), `'First'` (342/10000), `'Step'` (208/10000), `'The'` (179/10000)
**Position 1** (mean KL=2.75, n=10000): `' determine'` (3811/10000), `' find'` (2408/10000), `"'s"` (944/10000), `' solve'` (619/10000), `','` (226/10000)
**Position 2** (mean KL=1.46, n=10000): `' the'` (4780/10000), `' how'` (1584/10000), `' denote'` (709/10000), `' out'` (244/10000), `' '` (210/10000)
**Position 3** (mean KL=1.62, n=10000): `' many'` (1252/10000), `' the'` (876/10000), `' total'` (695/10000), `' number'` (535/10000), `' value'` (500/10000)
**Position 4** (mean KL=1.40, n=10000): `' of'` (2028/10000), `' number'` (1015/10000), `','` (345/10000), `' the'` (319/10000), `' many'` (198/10000)
**Position 5** (mean KL=1.81, n=10000): `' of'` (1539/10000), `' the'` (964/10000), `' \\('` (316/10000), `' we'` (283/10000), `' \\'` (182/10000)
**Position 6** (mean KL=1.35, n=10000): `' the'` (608/10000), `' of'` (476/10000), `' need'` (182/10000), `' '` (181/10000), `'2'` (162/10000)
**Position 7** (mean KL=1.43, n=10000): `','` (467/10000), `' the'` (380/10000), `' of'` (279/10000), `' to'` (267/10000), `' \\('` (250/10000)
**Position 8** (mean KL=1.12, n=10000): `','` (503/10000), `' we'` (445/10000), `' the'` (436/10000), `' of'` (357/10000), `' '` (319/10000)
**Position 9** (mean KL=1.25, n=10000): `','` (619/10000), `' the'` (585/10000), `' we'` (451/10000), `' need'` (281/10000), `' '` (259/10000)
**Position 10** (mean KL=1.32, n=10000): `' we'` (583/10000), `','` (514/10000), `' to'` (407/10000), `' the'` (366/10000), `' need'` (343/10000)
**Position 11** (mean KL=1.44, n=10000): `','` (769/10000), `' the'` (481/10000), `' we'` (460/10000), `' to'` (418/10000), `' need'` (384/10000)
**Position 12** (mean KL=1.33, n=10000): `' we'` (723/10000), `' the'` (546/10000), `','` (508/10000), `' to'` (489/10000), `' '` (352/10000)
**Position 13** (mean KL=1.38, n=10000): `' the'` (591/10000), `' need'` (569/10000), `' we'` (482/10000), `','` (462/10000), `' to'` (372/10000)
**Position 14** (mean KL=1.20, n=10000): `' to'` (640/10000), `' the'` (465/10000), `' we'` (370/10000), `' need'` (340/10000), `' '` (318/10000)
**Position 15** (mean KL=1.35, n=9998): `' the'` (521/9998), `','` (425/9998), `' to'` (419/9998), `' '` (322/9998), `' follow'` (298/9998)
**Position 16** (mean KL=1.24, n=9997): `' the'` (665/9997), `'1'` (374/9997), `' '` (355/9997), `' to'` (340/9997), `':\n\n'` (293/9997)
**Position 17** (mean KL=1.34, n=9995): `' the'` (552/9995), `'1'` (446/9995), `' '` (394/9995), `'.'` (339/9995), `' steps'` (291/9995)
**Position 18** (mean KL=1.34, n=9995): `' the'` (533/9995), `'1'` (394/9995), `'.'` (390/9995), `' '` (353/9995), `':\n\n'` (332/9995)
**Position 19** (mean KL=1.39, n=9995): `' the'` (640/9995), `'1'` (500/9995), `'.'` (349/9995), `' '` (337/9995), `','` (295/9995)

## 3. Token Type Distribution by Position Range

| Position Range | n_tokens | planning % | structural % | math_number % | math_operator % | math_latex % | continuation % | Mean KL |
|----------------|----------|------------|--------------|---------------|-----------------|--------------|----------------|---------|
| 0-5 | 50000 | 32.8% | 7.8% | 1.7% | 0.3% | 1.4% | 56.1% | 2.99 |
| 5-20 | 149980 | 15.9% | 19.4% | 9.5% | 2.5% | 3.1% | 49.6% | 1.35 |
| 20-50 | 299209 | 12.2% | 25.0% | 11.3% | 3.3% | 2.7% | 45.5% | 1.41 |
| 50-100 | 488165 | 9.7% | 30.8% | 15.2% | 5.3% | 3.8% | 35.1% | 1.01 |
| 100-200 | 808602 | 8.8% | 34.8% | 17.3% | 5.8% | 4.8% | 28.5% | 0.73 |
| 200-500 | 219080 | 7.9% | 39.3% | 18.1% | 4.5% | 5.9% | 24.4% | 0.78 |

### Mean KL by Category (all positions)

| Category | n_tokens | Mean KL | Median KL | Std KL |
|----------|----------|---------|-----------|--------|
| planning | 212717 | 1.754 | 0.081 | 3.494 |
| structural | 625605 | 0.893 | 0.001 | 3.010 |
| math_number | 302991 | 0.281 | 0.000 | 1.786 |
| math_operator | 95618 | 0.384 | 0.000 | 2.122 |
| math_latex | 84081 | 3.985 | 0.018 | 6.579 |
| continuation | 694024 | 0.921 | 0.008 | 2.603 |

### Mean KL by Category and Position Range

| Category | 0-4 | 5-19 | 20-49 | 50-99 | 100-199 | 200-499 |
|----------|-----|------|-------|-------|---------|---------|
| planning | 4.50 | 0.79 | 1.49 | 1.66 | 1.49 | 2.37 |
| structural | 3.26 | 1.46 | 1.60 | 0.93 | 0.60 | 0.86 |
| math_number | 1.49 | 0.60 | 0.74 | 0.28 | 0.17 | 0.13 |
| math_operator | 7.30 | 1.84 | 0.81 | 0.37 | 0.21 | 0.14 |
| math_latex | 8.84 | 9.23 | 6.50 | 4.95 | 2.97 | 1.87 |
| continuation | 1.94 | 1.12 | 1.19 | 0.89 | 0.70 | 0.48 |

## 4. High-KL Tokens (Top Tokens by Mean KL)

Tokens appearing at least 50 times, ranked by mean KL divergence:

### Top 40 highest-KL tokens

| Rank | Token | Category | Count | Mean KL | Median KL |
|------|-------|----------|-------|---------|-----------|
| 1 | 'Solution' | planning | 152 | 21.925 | 22.679 |
| 2 | 'Analysis' | continuation | 125 | 16.485 | 17.316 |
| 3 | '\\[' | math_latex | 7152 | 13.212 | 9.987 |
| 4 | ' examines' | continuation | 74 | 11.511 | 11.482 |
| 5 | 'He' | continuation | 150 | 10.798 | 8.638 |
| 6 | ' \\(' | math_latex | 21243 | 10.296 | 9.250 |
| 7 | 'First' | planning | 1706 | 9.978 | 11.247 |
| 8 | ' tests' | continuation | 52 | 8.708 | 8.740 |
| 9 | ' \\\\' | math_latex | 82 | 8.681 | 3.715 |
| 10 | '-a' | structural | 57 | 8.599 | 9.497 |
| 11 | 'There' | continuation | 201 | 8.282 | 2.934 |
| 12 | 'Therefore' | planning | 4913 | 7.953 | 7.773 |
| 13 | ' By' | planning | 74 | 7.876 | 7.629 |
| 14 | ' Identify' | continuation | 1345 | 7.779 | 7.239 |
| 15 | ' our' | continuation | 138 | 7.032 | 5.787 |
| 16 | 'To' | planning | 8806 | 6.900 | 6.926 |
| 17 | 'When' | continuation | 89 | 6.616 | 4.948 |
| 18 | 'Ident' | continuation | 1041 | 6.429 | 5.208 |
| 19 | 'This' | continuation | 621 | 6.343 | 2.446 |
| 20 | ',\n\n' | structural | 134 | 6.320 | 6.591 |
| 21 | '=\\' | structural | 84 | 6.277 | 5.377 |
| 22 | 'Thus' | planning | 2547 | 6.239 | 6.006 |
| 23 | ' First' | planning | 259 | 6.102 | 4.418 |
| 24 | '}=' | structural | 66 | 6.085 | 3.278 |
| 25 | ' According' | continuation | 525 | 6.081 | 4.672 |
| 26 | ' Virginia' | continuation | 65 | 6.027 | 3.765 |
| 27 | '$\\' | structural | 77 | 5.837 | 5.791 |
| 28 | ' \n' | structural | 189 | 5.753 | 5.251 |
| 29 | ' Thus' | planning | 158 | 5.706 | 4.505 |
| 30 | ' Matt' | continuation | 51 | 5.704 | 4.878 |
| 31 | ' *' | math_operator | 1330 | 5.580 | 0.235 |
| 32 | 'Calcul' | continuation | 92 | 5.481 | 4.774 |
| 33 | ' To' | planning | 1174 | 5.370 | 4.186 |
| 34 | ' Adding' | continuation | 70 | 5.349 | 3.819 |
| 35 | ' Math' | continuation | 55 | 5.263 | 4.308 |
| 36 | 'The' | planning | 2626 | 5.250 | 4.195 |
| 37 | '%.' | structural | 65 | 5.232 | 3.624 |
| 38 | ' When' | continuation | 126 | 5.097 | 3.839 |
| 39 | 'From' | planning | 132 | 5.075 | 3.915 |
| 40 | 'Next' | planning | 1753 | 5.031 | 4.757 |

### Top 40 lowest-KL tokens (teacher-student agreement)

| Rank | Token | Category | Count | Mean KL | Median KL |
|------|-------|----------|-------|---------|-----------|
| 1 | 'loor' | continuation | 94 | 0.000 | 0.000 |
| 2 | 'alie' | continuation | 56 | 0.000 | 0.000 |
| 3 | 'ina' | continuation | 86 | 0.000 | 0.000 |
| 4 | 'ulu' | continuation | 63 | 0.000 | 0.000 |
| 5 | 'atically' | continuation | 149 | 0.000 | 0.000 |
| 6 | 'aches' | continuation | 775 | 0.000 | 0.000 |
| 7 | 'ola' | continuation | 120 | 0.000 | 0.000 |
| 8 | 'ved' | continuation | 58 | 0.000 | 0.000 |
| 9 | 'ies' | continuation | 200 | 0.000 | 0.000 |
| 10 | 'em' | continuation | 145 | 0.000 | 0.000 |
| 11 | 'ssa' | continuation | 65 | 0.000 | 0.000 |
| 12 | 'ceil' | continuation | 104 | 0.000 | 0.000 |
| 13 | 'son' | continuation | 75 | 0.000 | 0.000 |
| 14 | ' vase' | continuation | 62 | 0.000 | 0.000 |
| 15 | 'o' | continuation | 300 | 0.000 | 0.000 |
| 16 | 'clidean' | continuation | 51 | 0.000 | 0.000 |
| 17 | 'ls' | continuation | 50 | 0.000 | 0.000 |
| 18 | 'anna' | continuation | 120 | 0.000 | 0.000 |
| 19 | 'abytes' | continuation | 59 | 0.000 | 0.000 |
| 20 | 'affle' | continuation | 59 | 0.000 | 0.000 |
| 21 | 'ears' | continuation | 99 | 0.000 | 0.000 |
| 22 | 'ient' | continuation | 155 | 0.000 | 0.000 |
| 23 | 'ash' | continuation | 190 | 0.000 | 0.000 |
| 24 | ' deviations' | continuation | 75 | 0.000 | 0.000 |
| 25 | 'diamond' | continuation | 59 | 0.000 | 0.000 |
| 26 | ' kits' | continuation | 87 | 0.000 | 0.000 |
| 27 | 'les' | continuation | 247 | 0.000 | 0.000 |
| 28 | 'tr' | continuation | 67 | 0.000 | 0.000 |
| 29 | 'ponents' | continuation | 256 | 0.000 | 0.000 |
| 30 | ' Je' | continuation | 98 | 0.000 | 0.000 |
| 31 | 'ns' | continuation | 92 | 0.000 | 0.000 |
| 32 | 'olly' | continuation | 74 | 0.000 | 0.000 |
| 33 | 'ders' | continuation | 83 | 0.000 | 0.000 |
| 34 | 'po' | continuation | 74 | 0.000 | 0.000 |
| 35 | 'ant' | continuation | 185 | 0.000 | 0.000 |
| 36 | 'ots' | continuation | 77 | 0.000 | 0.000 |
| 37 | 'aldo' | continuation | 232 | 0.000 | 0.000 |
| 38 | 'als' | continuation | 175 | 0.000 | 0.000 |
| 39 | 'ilt' | continuation | 70 | 0.000 | 0.000 |
| 40 | 'ram' | continuation | 156 | 0.000 | 0.000 |

## 5. Highest-KL Tokens by Category

### planning (top 15)

| Token | Count | Mean KL | Median KL |
|-------|-------|---------|-----------|
| 'Solution' | 152 | 21.925 | 22.679 |
| 'First' | 1706 | 9.978 | 11.247 |
| 'Therefore' | 4913 | 7.953 | 7.773 |
| ' By' | 74 | 7.876 | 7.629 |
| 'To' | 8806 | 6.900 | 6.926 |
| 'Thus' | 2547 | 6.239 | 6.006 |
| ' First' | 259 | 6.102 | 4.418 |
| ' Thus' | 158 | 5.706 | 4.505 |
| ' To' | 1174 | 5.370 | 4.186 |
| 'The' | 2626 | 5.250 | 4.195 |
| 'From' | 132 | 5.075 | 3.915 |
| 'Next' | 1753 | 5.031 | 4.757 |
| ' calculating' | 64 | 4.777 | 3.501 |
| 'Since' | 743 | 4.425 | 3.002 |
| ' We' | 1838 | 4.369 | 3.385 |

### structural (top 15)

| Token | Count | Mean KL | Median KL |
|-------|-------|---------|-----------|
| '-a' | 57 | 8.599 | 9.497 |
| ',\n\n' | 134 | 6.320 | 6.591 |
| '=\\' | 84 | 6.277 | 5.377 |
| '}=' | 66 | 6.085 | 3.278 |
| '$\\' | 77 | 5.837 | 5.791 |
| ' \n' | 189 | 5.753 | 5.251 |
| '%.' | 65 | 5.232 | 3.624 |
| ' $(' | 221 | 3.943 | 1.584 |
| '-x' | 92 | 3.932 | 0.313 |
| '**' | 1030 | 3.850 | 0.301 |
| '**\n\n' | 326 | 3.833 | 0.629 |
| ' (\\' | 415 | 3.767 | 0.293 |
| ' $\\' | 1768 | 3.604 | 1.391 |
| '-\\' | 78 | 3.566 | 0.021 |
| '}-' | 144 | 3.502 | 0.291 |

### math_number (top 15)

| Token | Count | Mean KL | Median KL |
|-------|-------|---------|-----------|
| '1' | 54168 | 0.661 | 0.000 |
| '4' | 26372 | 0.279 | 0.000 |
| '6' | 16826 | 0.261 | 0.000 |
| '3' | 34048 | 0.232 | 0.000 |
| '8' | 14772 | 0.208 | 0.000 |
| '9' | 10854 | 0.207 | 0.000 |
| '7' | 12945 | 0.206 | 0.000 |
| '2' | 55367 | 0.206 | 0.000 |
| '5' | 30104 | 0.187 | 0.000 |
| '0' | 47497 | 0.096 | 0.000 |

### math_operator (top 15)

| Token | Count | Mean KL | Median KL |
|-------|-------|---------|-----------|
| ' *' | 1330 | 5.580 | 0.235 |
| ' /' | 832 | 2.690 | 0.061 |
| '*' | 636 | 2.081 | 0.000 |
| '=' | 789 | 2.041 | 0.014 |
| '+' | 1200 | 1.141 | 0.004 |
| '/' | 1279 | 1.114 | 0.001 |
| '-' | 4685 | 0.883 | 0.002 |
| ' >' | 363 | 0.857 | 0.003 |
| ' ×' | 341 | 0.380 | 0.013 |
| ' -' | 19536 | 0.350 | 0.000 |
| ' +' | 14598 | 0.292 | 0.000 |
| ' <' | 599 | 0.223 | 0.000 |
| '_' | 1178 | 0.175 | 0.000 |
| ' =' | 39829 | 0.107 | 0.000 |
| '^' | 8355 | 0.105 | 0.000 |

### math_latex (top 15)

| Token | Count | Mean KL | Median KL |
|-------|-------|---------|-----------|
| '\\[' | 7152 | 13.212 | 9.987 |
| ' \\(' | 21243 | 10.296 | 9.250 |
| ' \\\\' | 82 | 8.681 | 3.715 |
| ' context' | 69 | 1.569 | 0.189 |
| ' sum' | 1738 | 1.419 | 0.163 |
| ' intersection' | 120 | 1.292 | 0.278 |
| ' eliminate' | 141 | 1.224 | 0.154 |
| '\\)' | 3991 | 1.153 | 0.001 |
| ' \\$' | 303 | 1.050 | 0.007 |
| '\\\\' | 52 | 1.029 | 0.019 |
| ' fractions' | 183 | 0.960 | 0.145 |
| ' fraction' | 607 | 0.908 | 0.017 |
| ' interval' | 168 | 0.833 | 0.040 |
| ' integer' | 926 | 0.729 | 0.008 |
| ' into' | 1770 | 0.697 | 0.009 |

### continuation (top 15)

| Token | Count | Mean KL | Median KL |
|-------|-------|---------|-----------|
| 'Analysis' | 125 | 16.485 | 17.316 |
| ' examines' | 74 | 11.511 | 11.482 |
| 'He' | 150 | 10.798 | 8.638 |
| ' tests' | 52 | 8.708 | 8.740 |
| 'There' | 201 | 8.282 | 2.934 |
| ' Identify' | 1345 | 7.779 | 7.239 |
| ' our' | 138 | 7.032 | 5.787 |
| 'When' | 89 | 6.616 | 4.948 |
| 'Ident' | 1041 | 6.429 | 5.208 |
| 'This' | 621 | 6.343 | 2.446 |
| ' According' | 525 | 6.081 | 4.672 |
| ' Virginia' | 65 | 6.027 | 3.765 |
| ' Matt' | 51 | 5.704 | 4.878 |
| 'Calcul' | 92 | 5.481 | 4.774 |
| ' Adding' | 70 | 5.349 | 3.819 |

## 6. Positions 1-4 Token Distributions

### Position 1 (mean KL=2.75, n=10000)

| Token | Count | % | Mean KL | Category |
|-------|-------|---|---------|----------|
| ' determine' | 3811 | 38.1% | 3.64 | continuation |
| ' find' | 2408 | 24.1% | 2.19 | continuation |
| "'s" | 944 | 9.4% | 0.73 | structural |
| ' solve' | 619 | 6.2% | 0.68 | continuation |
| ',' | 226 | 2.3% | 0.34 | structural |
| ' ' | 212 | 2.1% | 0.01 | structural |
| ' calculate' | 178 | 1.8% | 0.82 | continuation |
| 'Analysis' | 121 | 1.2% | 16.58 | continuation |
| ' evaluate' | 111 | 1.1% | 0.05 | continuation |
| ' compute' | 108 | 1.1% | 0.21 | continuation |
| ' the' | 103 | 1.0% | 1.65 | planning |
| ' simplify' | 97 | 1.0% | 0.15 | continuation |
| ' express' | 87 | 0.9% | 0.18 | continuation |
| ' are' | 84 | 0.8% | 0.36 | continuation |
| ' start' | 76 | 0.8% | 14.30 | continuation |
| ' convert' | 54 | 0.5% | 0.82 | continuation |
| ' factor' | 30 | 0.3% | 0.00 | continuation |
| ' estimate' | 25 | 0.2% | 0.11 | continuation |
| ' prove' | 24 | 0.2% | 0.11 | continuation |
| ' that' | 24 | 0.2% | 4.39 | continuation |

### Position 2 (mean KL=1.46, n=10000)

| Token | Count | % | Mean KL | Category |
|-------|-------|---|---------|----------|
| ' the' | 4780 | 47.8% | 0.56 | planning |
| ' how' | 1584 | 15.8% | 0.13 | continuation |
| ' denote' | 709 | 7.1% | 2.18 | continuation |
| ' out' | 244 | 2.4% | 1.35 | continuation |
| ' ' | 210 | 2.1% | 4.09 | structural |
| '1' | 209 | 2.1% | 0.00 | math_number |
| ' break' | 177 | 1.8% | 0.55 | continuation |
| ' for' | 132 | 1.3% | 4.66 | planning |
| ' we' | 119 | 1.2% | 2.51 | planning |
| '**\n\n' | 115 | 1.1% | 10.15 | structural |
| ' this' | 108 | 1.1% | 0.75 | continuation |
| ' $' | 106 | 1.1% | 2.36 | structural |
| ' \\' | 106 | 1.1% | 15.83 | structural |
| ' \\(' | 100 | 1.0% | 16.65 | math_latex |
| ' $\\' | 93 | 0.9% | 1.20 | structural |
| ' of' | 66 | 0.7% | 0.28 | continuation |
| ' let' | 52 | 0.5% | 1.01 | planning |
| ' with' | 50 | 0.5% | 0.30 | continuation |
| ' whether' | 41 | 0.4% | 0.31 | continuation |
| ' by' | 40 | 0.4% | 1.69 | planning |

### Position 3 (mean KL=1.62, n=10000)

| Token | Count | % | Mean KL | Category |
|-------|-------|---|---------|----------|
| ' many' | 1252 | 12.5% | 0.05 | continuation |
| ' the' | 876 | 8.8% | 0.29 | planning |
| ' total' | 695 | 7.0% | 0.48 | continuation |
| ' number' | 535 | 5.3% | 0.16 | continuation |
| ' value' | 500 | 5.0% | 0.07 | continuation |
| ' much' | 297 | 3.0% | 0.07 | continuation |
| ' problem' | 296 | 3.0% | 0.70 | continuation |
| ' how' | 263 | 2.6% | 0.16 | continuation |
| ' expression' | 242 | 2.4% | 0.18 | continuation |
| ':' | 208 | 2.1% | 0.00 | structural |
| ' down' | 152 | 1.5% | 0.29 | continuation |
| ' ' | 141 | 1.4% | 3.88 | structural |
| ' sum' | 124 | 1.2% | 1.74 | math_latex |
| 'This' | 114 | 1.1% | 21.79 | continuation |
| ' equation' | 101 | 1.0% | 0.16 | continuation |
| ' least' | 100 | 1.0% | 6.04 | continuation |
| "'s" | 91 | 0.9% | 0.03 | structural |
| '(\\' | 90 | 0.9% | 0.02 | structural |
| ' of' | 89 | 0.9% | 0.29 | continuation |
| ' need' | 89 | 0.9% | 5.98 | continuation |

### Position 4 (mean KL=1.40, n=10000)

| Token | Count | % | Mean KL | Category |
|-------|-------|---|---------|----------|
| ' of' | 2028 | 20.3% | 0.17 | continuation |
| ' number' | 1015 | 10.2% | 0.19 | continuation |
| ',' | 345 | 3.5% | 0.38 | structural |
| ' the' | 319 | 3.2% | 0.23 | planning |
| ' many' | 198 | 2.0% | 0.10 | continuation |
| ' \\(' | 159 | 1.6% | 19.13 | math_latex |
| ' \\' | 140 | 1.4% | 17.67 | structural |
| ' to' | 116 | 1.2% | 0.20 | planning |
| ' money' | 108 | 1.1% | 0.26 | continuation |
| ' ' | 107 | 1.1% | 4.25 | structural |
| ' amount' | 105 | 1.1% | 0.84 | continuation |
| ' question' | 102 | 1.0% | 3.14 | continuation |
| ' Determine' | 101 | 1.0% | 4.84 | continuation |
| ' that' | 96 | 1.0% | 0.79 | continuation |
| ' when' | 92 | 0.9% | 0.21 | continuation |
| ' more' | 87 | 0.9% | 0.00 | continuation |
| ' $' | 86 | 0.9% | 3.11 | structural |
| ' total' | 73 | 0.7% | 0.95 | continuation |
| ' cost' | 68 | 0.7% | 1.00 | continuation |
| ' value' | 67 | 0.7% | 0.69 | continuation |

## 7. Summary Statistics

- Mean sequence length: 201.5
- Median sequence length: 211.0
- Min/Max sequence length: 15/299

- Total tokens analyzed: 2015036
- Overall mean KL: 1.0064
- Overall median KL: 0.0010

- KL in first 1 positions: 3.8% of total
- KL in first 5 positions: 7.4% of total
- KL in first 10 positions: 10.8% of total
- KL in first 20 positions: 17.4% of total
- KL in first 50 positions: 38.2% of total

## 8. Key Findings

### Finding 1: Position 0 is dominated by "To" (75%) — a planning token the teacher strongly disagrees with
- 75.1% of trajectories start with "To" (as in "To find/determine/solve..."), with mean KL=7.32
- The teacher model prefers different opening strategies: when the student says "First" (KL=14.0), "Step" (KL=14.6), "**" (KL=18.6), or "Solution" (KL=27.5), the disagreement is even larger
- The student has a strong prior for "To solve/find..." openings; the teacher's distribution is more spread across strategies

### Finding 2: LaTeX delimiters are the single highest-KL token type
- `\(` (mean KL=10.3, 21K occurrences) and `\[` (mean KL=13.2, 7K occurrences) dominate KL across ALL positions
- This means the teacher and student fundamentally disagree on when/whether to use inline math formatting
- LaTeX as a category has mean KL=3.99, roughly 4x higher than any other category
- This likely reflects different training data distributions for math formatting conventions

### Finding 3: KL concentrates on "decision points", not computations
- **Low KL tokens** (near 0): word fragments ("aches", "ponent", "aldo"), numbers, operators (`=`, `^`, `+`)
- **High KL tokens**: sentence starters ("Therefore" KL=7.95, "Thus" KL=6.24, "First" KL=9.98), structural transitions
- The teacher-student disagreement is about HOW to structure reasoning, not WHAT to compute
- Math content tokens (numbers, operators) have the lowest KL (mean 0.28 and 0.38)

### Finding 4: Category composition shifts dramatically with position
- Positions 0-4: 33% planning tokens, only 2% math numbers
- Positions 200-500: 8% planning, 18% math numbers, 39% structural
- Early positions are "strategic" (what approach to take); later positions are "mechanical" (executing computations)
- This explains the positional KL decay: early = strategic disagreement, late = mechanical agreement

### Finding 5: The teacher particularly disagrees on response structure/format
- Structural tokens like `**\n\n`, `,\n\n`, `$\` have high KL (3.8-6.3)
- The teacher has different preferences for how to format and section mathematical reasoning
- Combined with the LaTeX finding, this suggests ~50%+ of KL comes from formatting/structure, not mathematical content

### Implications for Distillation
1. **Positional loss makes sense**: The first 50 positions carry 38% of total KL but are only ~25% of tokens; focusing loss here captures the strategic disagreements
2. **Format-blind distillation might help**: Much KL is "wasted" on formatting disagreements (LaTeX delimiters, bold markers). A distillation loss that ignores formatting tokens could be more efficient
3. **The teacher's value is in reasoning strategy**: The teacher disagrees most on which reasoning approach to use ("To" vs "First" vs "Step"), not on the math itself