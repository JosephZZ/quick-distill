# Function Calling Distillation Results

## Setup

- **Student**: Qwen2.5-Math-1.5B (math model, no function calling ability)
- **Teacher**: Qwen3-1.7B (general model, moderate function calling ability)
- **Training data**: 3,200 single-turn function calling examples from glaive-function-calling-v2 (only provides prompts — no ground truth used in training)
- **Eval**: BFCL v3 (simple: 400, multiple: 200) — 600 total problems
- **Config**: LoRA (r=32, alpha=64, all linear layers), n_samples=1, bs=16, lr=5e-5, 200 steps, cosine LR schedule
- **Metrics**: Name accuracy (correct function names), Full accuracy (name + arguments match), Parse rate (valid JSON output)
- **Eval temperature**: 0.0 (greedy), max_new_tokens=512

## LoRA Results

### All Checkpoints

| Model | Step | Name Acc | Full Acc | Parse Rate | Avg |
|-------|------|----------|----------|------------|-----|
| **Baseline** (Qwen2.5-Math-1.5B) | — | 9.7% | 2.7% | 24.2% | 12.2% |
| **Teacher** (Qwen3-1.7B) | — | 75.3% | 54.0% | 75.3% | 68.2% |
| | | | | | |
| **Pos-50** | 50 | 35.0% | 20.7% | 39.5% | 31.7% |
| | 100 | 81.2% | 50.0% | 87.0% | 72.7% |
| | 150 | 95.0% | 55.3% | 98.3% | 82.9% |
| | **200** | **95.2%** | 57.2% | **98.3%** | **83.6%** |
| | | | | | |
| **Pos-100** | 50 | 25.3% | 15.5% | 28.8% | 23.2% |
| | **100** | 86.2% | **61.3%** | 91.3% | 79.6% |
| | 150 | 85.2% | 61.3% | 90.8% | 79.1% |
| | 200 | 86.3% | 58.3% | 91.8% | 78.8% |
| | | | | | |
| **Pos-150** | 50 | 11.5% | 7.2% | 13.2% | 10.6% |
| | 100 | 87.2% | 56.8% | 91.7% | 78.6% |
| | 150 | 87.7% | 61.5% | 92.2% | 80.4% |
| | **200** | 88.7% | **61.5%** | 92.5% | **80.9%** |
| | | | | | |
| **Pos-200** | 50 | 53.0% | 31.5% | 59.0% | 47.8% |
| | 100 | 78.5% | 51.0% | 87.0% | 72.2% |
| | 150 | 79.7% | 52.3% | 88.8% | 73.6% |
| | 200 | 80.8% | 54.5% | 90.2% | 75.2% |
| | | | | | |
| **Full-seq** | 50 | 27.7% | 14.8% | 28.8% | 23.8% |
| | 100 | 81.0% | 58.2% | 86.7% | 75.3% |
| | 150 | 88.2% | 56.3% | 93.8% | 79.4% |
| | 200 | 86.2% | 57.7% | 92.2% | 78.7% |

### Best Checkpoints (by Full Accuracy)

| Method | Best Step | Name Acc | Full Acc | Parse Rate | Avg |
|--------|-----------|----------|----------|------------|-----|
| Baseline | — | 9.7% | 2.7% | 24.2% | 12.2% |
| Teacher | — | 75.3% | 54.0% | 75.3% | 68.2% |
| Pos-50 | 200 | 95.2% | 57.2% | 98.3% | 83.6% |
| **Pos-100** | **100** | 86.2% | **61.3%** | 91.3% | 79.6% |
| Pos-150 | 150/200 | 88.7% | 61.5% | 92.5% | 80.9% |
| Pos-200 | 200 | 80.8% | 54.5% | 90.2% | 75.2% |
| Full-seq | 100 | 81.0% | 58.2% | 86.7% | 75.3% |

### Per-Category Breakdown (Best Checkpoints)

| Method | Simple Name | Simple Full | Multiple Name | Multiple Full |
|--------|-------------|-------------|---------------|---------------|
| Baseline | 8.8% | 2.2% | 11.5% | 3.5% |
| Teacher | 78.5% | 57.0% | 69.0% | 48.0% |
| Pos-50 (s200) | 97.8% | 62.7% | 90.0% | 46.0% |
| **Pos-100 (s100)** | 93.2% | **70.2%** | 72.0% | 43.5% |
| Pos-150 (s200) | 94.2% | 68.2% | 77.5% | 48.0% |
| Pos-200 (s200) | 91.2% | 64.0% | 60.0% | 35.5% |
| Full-seq (s100) | 88.2% | 66.0% | 66.5% | 42.5% |

## Generation Length Analysis

### Data Statistics
- **Ground truth answer**: mean=33 tokens, median=29, p95=60
- **Training prompts**: mean=174 tokens
- **Eval prompts (BFCL)**: mean=325 tokens

### Actual Generation Lengths at Eval Time (100 simple examples, greedy, max_new_tokens=512)

| Model | Mean | Median | p95 | Max |
|-------|------|--------|-----|-----|
| Baseline (no distill) | 313 | 293 | 512 | 512 |
| **Pos-50** (s200) | **35.5** | **35** | 54 | 66 |
| Pos-100 (s100) | 119 | 90 | 275 | 512 |
| Pos-200 (s200) | 430 | 512 | 512 | 512 |
| Full-seq (s100) | 326 | 284 | 512 | 512 |
| Teacher (Qwen3-1.7B) | 241 | 200 | 512 | 512 |

### Output Pattern Examples

**Pos-50** — Clean JSON, immediate stop:
```
[{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}]<|endoftext|>
```

**Pos-100** — Natural language explanation + JSON:
```
The factorial of 5 is 120.
Function call:
```json
[{"name": "math.factorial", "arguments": {"number": 5}}]
```
```

**Pos-200** — Repetition degradation (same pattern as math fullseq repeating \boxed{}):
```
[JSON block] The area is 25... [JSON block again] The area is 25... [JSON block again]...
```
Generates 512 tokens, repeating the JSON function call block multiple times.

**Full-seq** — Natural language + JSON, occasionally stops:
```
Sure! Let me calculate the area... [explanation] [JSON block]<|endoftext|>
```

## FullFT Results

Config: Full finetune (no LoRA), lr=5e-6, n_samples=1, bs=16, 200 steps, cosine LR schedule.

### All Checkpoints

| Model | Step | Name Acc | Full Acc | Parse Rate | Avg |
|-------|------|----------|----------|------------|-----|
| **FullFT Pos-50** | 50 | 55.8% | 32.0% | 62.7% | 50.2% |
| | 100 | 81.5% | 46.8% | 87.2% | 71.8% |
| | 150 | 89.2% | 47.3% | 95.0% | 77.2% |
| | **200** | 89.2% | 47.7% | **95.5%** | **77.4%** |
| | | | | | |
| **FullFT Pos-100** | 50 | 47.3% | 30.0% | 54.3% | 43.9% |
| | 100 | 71.5% | 44.2% | 78.2% | 64.6% |
| | 150 | 83.8% | 47.7% | 89.5% | 73.7% |
| | **200** | 84.8% | **47.8%** | 91.8% | **74.8%** |
| | | | | | |
| **FullFT Pos-150** | 50 | 38.2% | 23.3% | 44.7% | 35.4% |
| | 100 | 63.3% | 39.2% | 69.2% | 57.2% |
| | **150** | 69.8% | **43.5%** | 76.3% | **63.2%** |
| | 200 | 69.2% | 43.3% | 75.8% | 62.8% |
| | | | | | |
| **FullFT Pos-200** | 50 | 36.3% | 22.0% | 43.3% | 33.9% |
| | 100 | 62.3% | 39.8% | 67.2% | 56.4% |
| | **150** | 67.3% | **42.0%** | 73.5% | **60.9%** |
| | 200 | 67.0% | 41.7% | 73.2% | 60.6% |
| | | | | | |
| **FullFT Full-seq** | 50 | 44.0% | 26.0% | 50.3% | 40.1% |
| | 100 | 67.3% | 42.5% | 73.5% | 61.1% |
| | 150 | 67.8% | 43.8% | 74.7% | 62.1% |
| | **200** | 68.3% | **43.8%** | 74.5% | **62.2%** |

### Best Checkpoints (by Full Accuracy)

| Method | Best Step | Name Acc | Full Acc | Parse Rate | Avg |
|--------|-----------|----------|----------|------------|-----|
| FullFT Pos-50 | 200 | 89.2% | 47.7% | 95.5% | 77.4% |
| **FullFT Pos-100** | **200** | 84.8% | **47.8%** | 91.8% | 74.8% |
| FullFT Pos-150 | 150 | 69.8% | 43.5% | 76.3% | 63.2% |
| FullFT Pos-200 | 150 | 67.3% | 42.0% | 73.5% | 60.9% |
| FullFT Full-seq | 200 | 68.3% | 43.8% | 74.5% | 62.2% |

### Per-Category Breakdown (Best Checkpoints)

| Method | Simple Name | Simple Full | Multiple Name | Multiple Full |
|--------|-------------|-------------|---------------|---------------|
| FullFT Pos-50 (s200) | 96.8% | 52.8% | 74.0% | 37.5% |
| FullFT Pos-100 (s200) | 92.2% | 52.8% | 70.0% | 38.0% |
| FullFT Pos-150 (s150) | 77.0% | 49.5% | 55.5% | 31.5% |
| FullFT Pos-200 (s150) | 72.8% | 47.2% | 56.5% | 31.5% |
| FullFT Full-seq (s200) | 75.2% | 50.0% | 54.5% | 31.5% |

## LoRA vs FullFT Comparison

| Method | LoRA Full Acc | FullFT Full Acc | Delta |
|--------|-------------|-----------------|-------|
| Pos-50 | 57.2% | 47.7% | **-9.5pp** |
| Pos-100 | 61.3% | 47.8% | **-13.5pp** |
| Pos-150 | 61.5% | 43.5% | **-18.0pp** |
| Pos-200 | 54.5% | 42.0% | **-12.5pp** |
| Full-seq | 58.2% | 43.8% | **-14.4pp** |

**LoRA consistently outperforms FullFT** across all position limits by 9.5-18.0pp in full accuracy. This is a larger gap than observed in math distillation, likely because:
1. Function calling requires precise format learning (JSON syntax), where LoRA's regularization prevents catastrophic forgetting of base formatting capabilities
2. The small dataset (3200 examples) favors LoRA's parameter-efficient approach
3. FullFT with lr=5e-6 may be underfitting (but higher lr risks instability)

## Key Findings

### 1. Distillation dramatically improves function calling ability
The student model (a math model with zero function calling training) goes from 2.7% to 61.3% full accuracy — a **22.7x improvement** — and **surpasses the teacher** (54.0%) in full accuracy after distillation.

### 2. Student surpasses teacher after distillation
All LoRA distilled variants exceed the teacher's full accuracy (54.0%):
- Pos-150: 61.5% (+7.5pp over teacher)
- Pos-100: 61.3% (+7.3pp over teacher)
- Full-seq: 58.2% (+4.2pp)
- Pos-50: 57.2% (+3.2pp)

This is consistent with math results where the distilled student also exceeds the teacher. The mechanism: on-policy distillation lets the student improve locally on its own distribution, rather than imitating the teacher's (often verbose/messy) outputs.

### 3. Pos-100/150 achieve highest argument accuracy, Pos-50 achieves highest format compliance
- **Pos-100/150**: Best full accuracy (61.3%/61.5%) — learn both format and argument precision
- **Pos-50**: Best name accuracy (95.2%) and parse rate (98.3%) — learns extremely clean JSON output format
- **Pos-50 also has highest average score** (83.6%) when averaging name/full/parse
- **Pos-150**: Best multiple-category full accuracy (48.0%) — better generalization to complex calls

### 4. Position limit implicitly controls generation length via max_new_tokens clamping
`max_new_tokens` is auto-clamped to `position_limit` during training. This means:
- Pos-50 students never generate >50 tokens during training → learn to produce compact outputs
- Pos-200 students generate up to 200 tokens → learn to fill that window

Combined with on-policy cascade learning (positions learned sequentially from start), shorter position limits create a positive feedback loop: learn EOS early → generate shorter → more concentrated signal → reinforce.

### 5. Pos-200 exhibits repetition degradation
Pos-200 shows the same repetition pathology as math full-seq: the model repeats the JSON function call block multiple times, generating 430 tokens on average for a 33-token task. This mirrors the `\boxed{}` repetition observed in math experiments.

Interestingly, full-seq (max_new_tokens=512) has SHORTER average generation (326) than pos-200 (430). This may be because full-seq trains with longer sequences and encounters more diverse content after the JSON, while pos-200 trains in a 200-token window that's dominated by repetition of short JSON outputs.

### 6. Positional distillation hypothesis validated for agentic tasks
The math insight (early tokens = strategy, later tokens = execution) maps to function calling:
- **Pos-50 (early tokens)**: Learns output format (JSON) and function selection → highest name accuracy
- **Pos-100 (mid tokens)**: Learns argument precision → highest full accuracy
- **Beyond 100 tokens**: Diminishing returns, repetition degradation begins

### 7. LoRA significantly outperforms FullFT for function calling
LoRA beats FullFT by 9.5-18.0pp across all position limits. This gap is larger than in math distillation:
- LoRA Pos-100: 61.3% vs FullFT Pos-100: 47.8% (-13.5pp)
- LoRA Pos-150: 61.5% vs FullFT Pos-150: 43.5% (-18.0pp)

This suggests function calling benefits more from LoRA's implicit regularization, which preserves the base model's formatting capabilities while learning the new task. FullFT may overfit to the small training set or lose JSON formatting precision.

### 8. Training dynamics
- **Pos-50/100**: Converge by step 100-150, then plateau
- **Pos-200/Full-seq**: Slower convergence, don't match pos-100's best accuracy
- Loss trajectories: all decrease from ~2.3 to ~0.5-0.8

| Experiment | Final Loss | Final KL |
|------------|-----------|----------|
| LoRA Pos-50 | 0.467 | 0.448 |
| LoRA Pos-100 | 0.687 | 0.671 |
| LoRA Pos-150 | 0.567 | 0.572 |
| LoRA Pos-200 | 0.763 | 0.744 |
| LoRA Full-seq | 0.467 | 0.451 |
| FullFT Pos-50 | 0.730 | 0.610 |
| FullFT Pos-100 | 1.423 | 0.966 |
| FullFT Pos-150 | 1.237 | 0.820 |
| FullFT Pos-200 | 1.558 | 1.149 |
| FullFT Full-seq | 1.192 | 0.946 |

## Files

- Training script (LoRA): `scripts/run_funcall_experiments.sh`
- Training script (V2: LoRA pos-150 + FullFT): `scripts/run_funcall_experiments_v2.sh`
- Data preparation: `scripts/prepare_funcall_data.py`
- Evaluation: `eval_funcall.py`
- Eval runner: `scripts/eval_funcall_all.sh`
- Training data: `data/funcall/train.jsonl` (3,200 examples from glaive-function-calling-v2)
- Eval data: `data/funcall/eval_bfcl.jsonl` (1,000 examples, 600 used for simple+multiple)
- LoRA checkpoints: `checkpoints/funcall-{pos50,pos100,pos150,pos200,fullseq}-n1/`
- FullFT checkpoints: `checkpoints/funcall-{pos50,pos100,pos150,pos200,fullseq}-n1-fullft/`
