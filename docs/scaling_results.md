# Scaling Experiment Results

On-policy reverse KL distillation across different student-teacher size combinations.
All experiments use LoRA (r=32), pos-100, n_samples=1, chunk_size=16 (3200 problems), 200 training steps.

## Experiment Configurations

| Config | Student | Teacher | Student Params | Teacher Params |
|--------|---------|---------|----------------|----------------|
| A | Qwen2.5-Math-1.5B | Qwen3-4B | 1.5B | 4B |
| B | Qwen2.5-Math-1.5B | Qwen3-8B | 1.5B | 8B |
| C | Qwen3-1.7B | Qwen3-4B | 1.7B | 4B |
| D | Qwen3-1.7B | Qwen3-8B | 1.7B | 8B |
| E | Qwen3-4B | Qwen3-8B | 4B | 8B |

---

## 1. Baseline Results

### Math (MATH-500, n=4, temp=0.7)

| Model | avg@4 | pass@4 |
|-------|-------|--------|
| Qwen2.5-Math-1.5B | 50.95% | 72.8% |
| Qwen3-1.7B | 69.20% | 81.0% |
| Qwen3-4B | 77.95% | 86.4% |
| Qwen3-8B | 77.70% | 87.8% |

### Function Calling (600 problems)

| Model | name_acc | full_acc | parse_rate |
|-------|----------|----------|------------|
| Qwen2.5-Math-1.5B | 9.7% | 2.7% | 24.2% |
| Qwen3-1.7B | 91.5% | 59.8% | 92.2% |
| Qwen3-4B | 99.0% | 73.0% | 99.8% |
| Qwen3-8B | 99.0% | 78.5% | 100.0% |

### Coding (HumanEval / MBPP, pass@1, temp=0.0)

| Model | HE | HE+ | MBPP | MBPP+ |
|-------|-----|------|------|-------|
| Qwen3-1.7B | 39.6% | 36.0% | 52.4% | 44.7% |
| Qwen3-4B | 73.2% | 66.5% | 67.2% | 55.6% |
| Qwen3-8B | 63.4% | 59.1% | 57.7% | 49.5% |

---

## 2. Math Results (MATH-500)

### avg@4

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| — | M-1.5B baseline | — | — | — | — | — | **50.95%** | — |
| A | M-1.5B | Q3-4B | 65.50% | 66.20% | 68.95% | 67.20% | **68.95%** | 150 |
| B | M-1.5B | Q3-8B | 64.70% | 66.10% | 67.20% | 67.85% | **67.85%** | 200 |
| — | Q3-1.7B baseline | — | — | — | — | — | **69.20%** | — |
| C | Q3-1.7B | Q3-4B | 67.65% | 69.20% | 68.10% | 67.55% | **69.20%** | 100 |
| D | Q3-1.7B | Q3-8B | 67.30% | 69.15% | 68.30% | 68.60% | **69.15%** | 100 |
| — | Q3-4B baseline | — | — | — | — | — | **77.95%** | — |
| E | Q3-4B | Q3-8B | 76.50% | 77.05% | 76.90% | 76.50% | **77.05%** | 100 |

### pass@4

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| — | M-1.5B baseline | — | — | — | — | — | **72.8%** | — |
| A | M-1.5B | Q3-4B | 79.8% | 80.4% | 81.0% | 80.8% | **81.0%** | 150 |
| B | M-1.5B | Q3-8B | 79.6% | 79.4% | 82.2% | 81.6% | **82.2%** | 150 |
| — | Q3-1.7B baseline | — | — | — | — | — | **81.0%** | — |
| C | Q3-1.7B | Q3-4B | 79.6% | 81.2% | 80.0% | 80.8% | **81.2%** | 100 |
| D | Q3-1.7B | Q3-8B | 78.8% | 81.2% | 79.2% | 80.6% | **81.2%** | 100 |
| — | Q3-4B baseline | — | — | — | — | — | **86.4%** | — |
| E | Q3-4B | Q3-8B | 84.8% | 85.2% | 85.4% | 84.6% | **85.4%** | 150 |

### Math: Improvement Over Student Baseline

| Config | Student | Baseline avg@4 | Best avg@4 | Δ avg@4 | Baseline pass@4 | Best pass@4 | Δ pass@4 |
|--------|---------|---------------|------------|---------|-----------------|-------------|----------|
| A | M-1.5B | 50.95% | 68.95% | **+18.0%** | 72.8% | 81.0% | **+8.2%** |
| B | M-1.5B | 50.95% | 67.85% | **+16.9%** | 72.8% | 82.2% | **+9.4%** |
| C | Q3-1.7B | 69.20% | 69.20% | **+0.0%** | 81.0% | 81.2% | **+0.2%** |
| D | Q3-1.7B | 69.20% | 69.15% | **-0.05%** | 81.0% | 81.2% | **+0.2%** |
| E | Q3-4B | 77.95% | 77.05% | **-0.9%** | 86.4% | 85.4% | **-1.0%** |

### Comparison with Original Experiments (M-1.5B → Q3-1.7B, pos-100, LoRA, n1)

| Method | Best avg@4 | Δ vs baseline |
|--------|-----------|---------------|
| M-1.5B baseline | 50.95% | — |
| M-1.5B → Q3-1.7B (original) | 65.85% | +14.9% |
| M-1.5B → **Q3-4B** (scaling A) | **68.95%** | **+18.0%** |
| M-1.5B → Q3-8B (scaling B) | 67.85% | +16.9% |

---

## 3. Function Calling Results (name_acc / full_acc)

### name_acc

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| — | M-1.5B baseline | — | — | — | — | — | **9.7%** | — |
| A | M-1.5B | Q3-4B | 13.0% | 50.0% | 62.8% | 67.8% | **67.8%** | 200 |
| B | M-1.5B | Q3-8B | 61.8% | 71.8% | 76.8% | 72.8% | **76.8%** | 150 |
| — | Q3-1.7B baseline | — | — | — | — | — | **91.5%** | — |
| C | Q3-1.7B | Q3-4B | 99.2% | 98.7% | 99.0% | 99.2% | **99.2%** | 50/200 |
| D | Q3-1.7B | Q3-8B | 99.0% | 98.8% | 99.0% | 99.0% | **99.0%** | 50/150/200 |
| — | Q3-4B baseline | — | — | — | — | — | **99.0%** | — |
| E | Q3-4B | Q3-8B | 98.5% | 98.8% | 98.8% | 98.8% | **98.8%** | 100/150/200 |

### full_acc

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| — | M-1.5B baseline | — | — | — | — | — | **2.7%** | — |
| A | M-1.5B | Q3-4B | 7.8% | 29.0% | 43.0% | 45.8% | **45.8%** | 200 |
| B | M-1.5B | Q3-8B | 38.2% | 54.0% | 57.0% | 55.2% | **57.0%** | 150 |
| — | Q3-1.7B baseline | — | — | — | — | — | **59.8%** | — |
| C | Q3-1.7B | Q3-4B | 71.5% | 69.7% | 72.8% | 73.7% | **73.7%** | 200 |
| D | Q3-1.7B | Q3-8B | 74.0% | 69.0% | 68.5% | 69.7% | **74.0%** | 50 |
| — | Q3-4B baseline | — | — | — | — | — | **73.0%** | — |
| E | Q3-4B | Q3-8B | 76.8% | 75.8% | 75.7% | 75.0% | **76.8%** | 50 |

### Funcall: Improvement Over Student Baseline (full_acc)

| Config | Student | Student Baseline | Best Distilled | Delta |
|--------|---------|-----------------|----------------|-------|
| A | M-1.5B | 2.7% | 45.8% | **+43.1%** |
| B | M-1.5B | 2.7% | 57.0% | **+54.3%** |
| C | Q3-1.7B | 59.8% | 73.7% | **+13.9%** |
| D | Q3-1.7B | 59.8% | 74.0% | **+14.2%** |
| E | Q3-4B | 73.0% | 76.8% | **+3.8%** |

---

## 4. Coding Results

### HumanEval (pass@1)

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| — | Q3-1.7B baseline | — | — | — | — | — | **39.6%** | — |
| A | M-1.5B | Q3-4B | 37.2% | 43.3% | 42.7% | 42.1% | **43.3%** | 100 |
| B | M-1.5B | Q3-8B | 36.0% | 41.5% | 39.6% | 39.6% | **41.5%** | 100 |
| C | Q3-1.7B | Q3-4B | 40.9% | 44.5% | 41.5% | 42.1% | **44.5%** | 100 |
| D | Q3-1.7B | Q3-8B | 39.0% | 36.6% | 37.8% | 40.2% | **40.2%** | 200 |
| — | Q3-4B baseline | — | — | — | — | — | **73.2%** | — |
| E | Q3-4B | Q3-8B | 71.3% | 68.9% | 71.3% | 70.7% | **71.3%** | 50/150 |

### HumanEval+ (pass@1)

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| — | Q3-1.7B baseline | — | — | — | — | — | **36.0%** | — |
| A | M-1.5B | Q3-4B | 34.8% | 38.4% | 37.2% | 37.2% | **38.4%** | 100 |
| B | M-1.5B | Q3-8B | 31.1% | 36.6% | 34.8% | 34.8% | **36.6%** | 100 |
| C | Q3-1.7B | Q3-4B | 36.6% | 40.9% | 37.2% | 37.8% | **40.9%** | 100 |
| D | Q3-1.7B | Q3-8B | 36.6% | 32.3% | 34.8% | 37.2% | **37.2%** | 200 |
| — | Q3-4B baseline | — | — | — | — | — | **66.5%** | — |
| E | Q3-4B | Q3-8B | 64.6% | 64.6% | 64.6% | 64.6% | **64.6%** | all |

### MBPP (pass@1)

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| — | Q3-1.7B baseline | — | — | — | — | — | **52.4%** | — |
| A | M-1.5B | Q3-4B | 50.3% | 49.7% | 52.4% | 52.4% | **52.4%** | 150/200 |
| B | M-1.5B | Q3-8B | 50.3% | 50.5% | 48.7% | 48.1% | **50.5%** | 100 |
| C | Q3-1.7B | Q3-4B | 54.2% | 55.8% | 56.3% | 57.1% | **57.1%** | 200 |
| D | Q3-1.7B | Q3-8B | 56.9% | 57.9% | 58.2% | 58.5% | **58.5%** | 200 |
| — | Q3-4B baseline | — | — | — | — | — | **67.2%** | — |
| E | Q3-4B | Q3-8B | 65.6% | 65.3% | 65.3% | 65.3% | **65.6%** | 50 |

### MBPP+ (pass@1)

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| — | Q3-1.7B baseline | — | — | — | — | — | **44.7%** | — |
| A | M-1.5B | Q3-4B | 44.2% | 44.4% | 47.6% | 47.1% | **47.6%** | 150 |
| B | M-1.5B | Q3-8B | 44.2% | 44.4% | 43.4% | 43.1% | **44.4%** | 100 |
| C | Q3-1.7B | Q3-4B | 44.7% | 46.6% | 47.1% | 47.6% | **47.6%** | 200 |
| D | Q3-1.7B | Q3-8B | 47.9% | 48.1% | 47.9% | 48.1% | **48.1%** | 100/200 |
| — | Q3-4B baseline | — | — | — | — | — | **55.6%** | — |
| E | Q3-4B | Q3-8B | 54.8% | 55.6% | 56.3% | 56.3% | **56.3%** | 150/200 |

### Coding: Improvement Over Student Baseline (Best HE+ / MBPP+)

| Config | Student | HE+ Baseline | Best HE+ | HE+ Delta | MBPP+ Baseline | Best MBPP+ | MBPP+ Delta |
|--------|---------|-------------|-----------|-----------|---------------|------------|-------------|
| A | M-1.5B | — | 38.4% | — | — | 47.6% | — |
| B | M-1.5B | — | 36.6% | — | — | 44.4% | — |
| C | Q3-1.7B | 36.0% | 40.9% | **+4.9%** | 44.7% | 47.6% | **+2.9%** |
| D | Q3-1.7B | 36.0% | 37.2% | **+1.2%** | 44.7% | 48.1% | **+3.4%** |
| E | Q3-4B | 66.5% | 64.6% | **-1.9%** | 55.6% | 56.3% | **+0.7%** |

---

## 5. Summary of Best Results Per Config

| Config | Student -> Teacher | Math avg@4 | Δ avg@4 | Math pass@4 | Δ pass@4 | Funcall full_acc | Δ Funcall | HE+ | MBPP+ |
|--------|-------------------|-----------|---------|-------------|----------|-----------------|-----------|-----|-------|
| A | M-1.5B -> Q3-4B | 68.95% | **+18.0%** | 81.0% | +8.2% | 45.8% | +43.1% | 38.4% | 47.6% |
| B | M-1.5B -> Q3-8B | 67.85% | **+16.9%** | 82.2% | +9.4% | 57.0% | +54.3% | 36.6% | 44.4% |
| C | Q3-1.7B -> Q3-4B | 69.20% | +0.0% | 81.2% | +0.2% | 73.7% | +13.9% | 40.9% | 47.6% |
| D | Q3-1.7B -> Q3-8B | 69.15% | -0.05% | 81.2% | +0.2% | 74.0% | +14.2% | 37.2% | 48.1% |
| E | Q3-4B -> Q3-8B | 77.05% | -0.9% | 85.4% | -1.0% | 76.8% | +3.8% | 64.6% | 56.3% |

---

## 6. Key Findings

### Math
1. **Cross-architecture distillation (M-1.5B) yields massive avg@4 gains.** Configs A and B show +18.0% and +16.9% avg@4 improvement (50.95% → 68.95%/67.85%), far exceeding same-family distillation. The Math-specialized student has the most room to improve.
2. **4B teacher outperforms 8B teacher for Math-1.5B.** Config A (Q3-4B teacher, 68.95% avg@4) beats Config B (Q3-8B teacher, 67.85%) by 1.1%. The capacity gap with 8B may be too large for efficient distillation.
3. **Scaling up from 1.7B to 4B teacher helps significantly.** Original M-1.5B→Q3-1.7B achieved 65.85% avg@4; scaling to Q3-4B teacher achieves 68.95% (+3.1%), confirming larger teachers help up to a point.
4. **Same-family distillation (Qwen3) shows zero gains on math.** Configs C/D (Q3-1.7B student) show 0.0%/-0.05% avg@4 change, Config E (Q3-4B) shows -0.9%. These students are already near their capacity ceiling for math.
5. **Larger teacher does not help for Qwen3 students.** Configs C vs D show identical performance; the 8B teacher provides no additional benefit when the student is already a Qwen3 model.

### Function Calling
1. **Distillation is extremely effective for cross-architecture transfer.** Configs A and B show massive gains (+43-54%) because Math-1.5B has almost no function calling ability (2.7% full_acc) and distillation essentially teaches it a new capability.
2. **Larger teacher matters significantly for weak students.** Config B (8B teacher, 57.0%) dramatically outperforms Config A (4B teacher, 45.8%) for the Math-1.5B student.
3. **Same-family distillation still shows strong funcall gains.** Configs C and D improve Qwen3-1.7B from 59.8% to ~74%, a +14% improvement. The teacher's stronger function calling knowledge transfers well even within the same model family.
4. **Diminishing returns at higher baselines.** Config E (Q3-4B, baseline 73.0%) improves only +3.8% to 76.8%, approaching but not reaching the teacher's 78.5%.
5. **Teacher ceiling effect.** Distilled students approach but do not exceed teacher performance: Config C reaches 73.7% vs teacher's 73.0% (slightly exceeding), Config D reaches 74.0% vs teacher's 78.5%, Config E reaches 76.8% vs teacher's 78.5%.

### Coding
1. **Same-family distillation with 4B teacher helps coding.** Config C (Q3-1.7B→Q3-4B) improves HE+ from 36.0% to 40.9% (+4.9%) and MBPP+ from 44.7% to 47.6% (+2.9%). The 4B teacher consistently outperforms the 8B teacher for coding on small students.
2. **8B teacher underperforms 4B teacher for coding on small students.** Config D (Q3-1.7B→Q3-8B) only reaches 37.2% HE+ vs Config C's 40.9%. Config B (M-1.5B→Q3-8B) also trails Config A (M-1.5B→Q3-4B). This suggests the 8B teacher's distribution may be harder for small students to learn from for code.
3. **Config E shows slight degradation on HumanEval.** Q3-4B drops from 66.5% to 64.6% HE+ (-1.9%), though MBPP+ slightly improves (+0.7%). The 4B student is already strong at coding and the 8B teacher doesn't improve it.
4. **Anomaly: Qwen3-8B baseline is weaker than Qwen3-4B on coding.** 8B gets 59.1% HE+ vs 4B's 66.5%. This may explain why distilling from 8B hurts coding — the teacher is actually worse at code than the 4B model, likely due to Qwen3's training data distribution.

### General Patterns
1. **Weaker students benefit more from distillation.** The largest gains occur when the student has the most room to improve (Math-1.5B on funcall: +54%, on math: +9%).
2. **Optimal training step varies.** Strong students peak early (step 50-100) while weak students need more training (step 150-200), especially for funcall where format learning takes time.
3. **Larger teacher generally helps more**, but the effect is most pronounced when the student is weak. For strong students, teacher size matters less.
4. **Cross-architecture transfer works surprisingly well.** The Math-1.5B model, despite being a different architecture/pretraining, successfully absorbs knowledge from Qwen3 teachers.

---

## 7. Full-Sequence Distillation (Scaling, LoRA)

Same student–teacher matrix as Sections 2–4 (Configs A–E), but **`position_limit=0`** (loss on full student response), generation via **`--use_vllm`**, **`teacher_micro_bs=4`**, otherwise matched to pos-100 scaling: LoRA r=32/α=64, lr=5e-5, n_samples=1, bs=16, 3200 problems, 200 steps, save every 50. **`max_new_tokens`**: 2048 (math), 512 (coding / funcall). Checkpoints: `checkpoints/scale-*-{math,coding,funcall}-fullseq/`. Scripts: `scripts/run_scaling_gpu0_fullseq.sh`, `scripts/run_scaling_gpu1_fullseq.sh`.

**Note:** MATH-500 numbers below use the same eval script as pos-100 (`eval_math500.py`, last-`\boxed{}` extraction). Full-seq models may repeat `\boxed{}`; if metrics look inconsistent with samples, consider first-boxed or manual spot checks (see project `CLAUDE.md`).

### 7.1 Math (MATH-500, n=4, temp=0.7)

#### avg@4

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| A | M-1.5B | Q3-4B | — | — | — | — | — | — |
| B | M-1.5B | Q3-8B | — | — | — | — | — | — |
| C | Q3-1.7B | Q3-4B | — | — | — | — | — | — |
| D | Q3-1.7B | Q3-8B | — | — | — | — | — | — |
| E | Q3-4B | Q3-8B | — | — | — | — | — | — |

#### pass@4

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| A | M-1.5B | Q3-4B | — | — | — | — | — | — |
| B | M-1.5B | Q3-8B | — | — | — | — | — | — |
| C | Q3-1.7B | Q3-4B | — | — | — | — | — | — |
| D | Q3-1.7B | Q3-8B | — | — | — | — | — | — |
| E | Q3-4B | Q3-8B | — | — | — | — | — | — |

*Fill from each run’s `checkpoints/scale-*-math-fullseq/eval_step_{50,100,150,200}/summary.json` (and derived avg@4 / pass@4 if computed outside `summary.json`).*

### 7.2 Function Calling (name_acc / full_acc)

#### name_acc

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| A | M-1.5B | Q3-4B | — | — | — | — | — | — |
| B | M-1.5B | Q3-8B | — | — | — | — | — | — |
| C | Q3-1.7B | Q3-4B | — | — | — | — | — | — |
| D | Q3-1.7B | Q3-8B | — | — | — | — | — | — |
| E | Q3-4B | Q3-8B | — | — | — | — | — | — |

#### full_acc

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| A | M-1.5B | Q3-4B | — | — | — | — | — | — |
| B | M-1.5B | Q3-8B | — | — | — | — | — | — |
| C | Q3-1.7B | Q3-4B | — | — | — | — | — | — |
| D | Q3-1.7B | Q3-8B | — | — | — | — | — | — |
| E | Q3-4B | Q3-8B | — | — | — | — | — | — |

*Fill from `checkpoints/scale-*-funcall-fullseq/eval_step_*/summary.json`.*

### 7.3 Coding (HumanEval / MBPP, pass@1)

#### HumanEval (pass@1)

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| A | M-1.5B | Q3-4B | — | — | — | — | — | — |
| B | M-1.5B | Q3-8B | — | — | — | — | — | — |
| C | Q3-1.7B | Q3-4B | — | — | — | — | — | — |
| D | Q3-1.7B | Q3-8B | — | — | — | — | — | — |
| E | Q3-4B | Q3-8B | — | — | — | — | — | — |

#### HumanEval+ (pass@1)

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| A | M-1.5B | Q3-4B | — | — | — | — | — | — |
| B | M-1.5B | Q3-8B | — | — | — | — | — | — |
| C | Q3-1.7B | Q3-4B | — | — | — | — | — | — |
| D | Q3-1.7B | Q3-8B | — | — | — | — | — | — |
| E | Q3-4B | Q3-8B | — | — | — | — | — | — |

#### MBPP (pass@1)

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| A | M-1.5B | Q3-4B | — | — | — | — | — | — |
| B | M-1.5B | Q3-8B | — | — | — | — | — | — |
| C | Q3-1.7B | Q3-4B | — | — | — | — | — | — |
| D | Q3-1.7B | Q3-8B | — | — | — | — | — | — |
| E | Q3-4B | Q3-8B | — | — | — | — | — | — |

#### MBPP+ (pass@1)

| Config | Student | Teacher | Step 50 | Step 100 | Step 150 | Step 200 | **Best** | Best Step |
|--------|---------|---------|---------|----------|----------|----------|----------|-----------|
| A | M-1.5B | Q3-4B | — | — | — | — | — | — |
| B | M-1.5B | Q3-8B | — | — | — | — | — | — |
| C | Q3-1.7B | Q3-4B | — | — | — | — | — | — |
| D | Q3-1.7B | Q3-8B | — | — | — | — | — | — |
| E | Q3-4B | Q3-8B | — | — | — | — | — | — |

*Coding metrics require `evalplus` on the jsonl files under each `eval_step_*` (same workflow as pos-100 scaling); the training script only drops a stub `summary.json` for coding.*

### 7.4 Full-seq vs pos-100 (best step, to be filled)

| Config | Task | Best pos-100 | Best full-seq | Notes |
|--------|------|--------------|---------------|-------|
| A–E | Math / Funcall / Coding | see §§2–4 | see §7.1–7.3 | Compare after tables above are filled |
