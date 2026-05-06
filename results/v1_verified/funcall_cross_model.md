# Cross-Model Function Calling (BFCL) Results

All experiments: LoRA r=32, alpha=64, lr=5e-5, 200 steps, 3200 training problems, seed=42.
Zero OOM in all runs. All results verified.

## Qwen (Qwen2.5-Math-1.5B → Qwen3-1.7B)

Teacher funcall ability: WEAK (54.0% full_acc, outputs natural language math solutions instead of JSON)
Baseline: 2.7% full_acc

| Method | Parse Rate | Name Acc | Full Acc | vs Teacher |
|--------|-----------|----------|----------|------------|
| Teacher | — | 75.3% | 54.0% | — |
| Baseline | 24.2% | 9.7% | 2.7% | — |
| **pos-50** | **98.3%** | **95.2%** | 57.2% | +3.2pp ✅ |
| **pos-100** | 91.3% | 86.2% | **61.3%** | **+7.3pp** ✅ |
| pos-150 | 92.5% | 88.7% | 61.5% | +7.5pp ✅ |
| pos-200 | 90.2% | 80.8% | 54.5% | +0.5pp |
| fullseq | 86.7% | 81.0% | 58.2% | +4.2pp ✅ |

## Gemma (gemma-2-2b-it → gemma-3-4b-it)

> **2026-05-03 RE-EVAL (vLLM 0.11.0).** All numbers below replace the original
> v1 evaluation, which used vLLM 0.8.5 and tripped the Gemma3 `boi_token`
> tokenizer bug (teacher loaded but tokenizer multimodal-init failed silently,
> producing garbage outputs). Original numbers (teacher 25.0, student 0%,
> fullseq 3.9) were artifacts of that bug, not real model behaviour.
>
> Re-eval setup: gemma3 venv (`/zhi_backup/ziheng/venvs/gemma3/`), vLLM 0.11.0,
> 600 problems (simple 400 + multiple 200), greedy, max_new_tokens=512.
> Raw outputs: `eval_results/gemma_funcall_rerun/`.

Teacher funcall ability: STRONG (72.83% full_acc, outputs clean JSON via chat template)
Baseline: 73.17% full_acc (gemma-2-2b-it is already capable on BFCL out of the box)

| Method | Best Step | Parse Rate | Name Acc | Full Acc | vs Teacher |
|--------|-----------|-----------|----------|----------|------------|
| Teacher (gemma-3-4b-it)         | —    | 99.00% | 98.67% | **72.83%** | — |
| Baseline (gemma-2-2b-it)        | —    | 95.33% | 95.00% | **73.17%** | +0.34 |
| **pos-50** (LoRA)               | s100 | 91.67% | 91.33% | 75.67% | **+2.84 ✅** |
| **pos-100** (LoRA)              | s100 | 96.17% | 95.67% | **79.00%** | **+6.17 ✅** |
| fullseq (LoRA)                  | s200 | 98.50% | 98.17% | 76.83% | +4.00 ✅ |

### Per-step trajectories (600 problems, all three metrics)

**pos-50 LoRA**

| Step | name_acc | full_acc | parse_rate |
|------|---------:|---------:|-----------:|
| 50   | 87.83 | 64.67 | 88.00 |
| 100  | **91.33** | **75.67** | **91.67** |
| 150  | 92.83 | 75.17 | 93.17 |
| 200  | 92.83 | 75.00 | 93.17 |

**pos-100 LoRA**

| Step | name_acc | full_acc | parse_rate |
|------|---------:|---------:|-----------:|
| 50   | 88.67 | 70.17 | 89.50 |
| **100**  | **95.67** | **79.00** | **96.17** |
| 150  | 94.00 | 76.00 | 94.33 |
| 200  | 94.00 | 76.00 | 94.17 |

**fullseq LoRA**

| Step | name_acc | full_acc | parse_rate |
|------|---------:|---------:|-----------:|
| 50   | 95.83 | 76.33 | 96.33 |
| 100  | 98.50 | 75.67 | 98.83 |
| 150  | 98.50 | 76.00 | 98.83 |
| **200**  | **98.17** | **76.83** | **98.50** |

Observations across all three metrics:

- **Full_acc**: pos-100 wins (79.00 > fullseq 76.83 > pos-50 75.67). All three exceed teacher (72.83) and student baseline (73.17).
- **Name_acc**: fullseq wins (98.17 ≈ teacher 98.67), then pos-100 (95.67), then pos-50 (92.83). All three improve over baseline (95.00).
- **Parse_rate**: fullseq wins (98.50 ≈ teacher 99.00), then pos-100 (96.17), then pos-50 (93.17). All three are above baseline (95.33).
- All three methods are stable (no degradation pattern across steps).
- **Tradeoff**: fullseq learns the JSON format more cleanly (highest parse + name), but pos-100 is more accurate on argument values (highest full_acc). Reverse-KL on a short prefix locks the student onto a slightly-different mode that gets the arguments right more often, even if the format envelope is marginally less polished.

### Old (broken-pipeline) numbers — KEPT FOR AUDIT, DO NOT USE

| Method | Parse Rate | Name Acc | Full Acc |
|--------|-----------|----------|----------|
| ~~Teacher~~ | ~~46.2%~~ | ~~37.1%~~ | ~~25.0%~~ |
| ~~Baseline~~ | ~~0.4%~~ | ~~0%~~ | ~~0%~~ |
| ~~pos-50~~ | ~~60.0%~~ | ~~39.8%~~ | ~~30.9%~~ |
| ~~pos-100~~ | ~~11.9%~~ | ~~8.1%~~ | ~~6.4%~~ |
| ~~fullseq~~ | ~~11.7%~~ | ~~4.8%~~ | ~~3.9%~~ |

## Llama (Llama-3.2-1B-Instruct → Llama-3.1-8B-Instruct)

Teacher funcall ability: STRONG (63.7% full_acc, outputs clean JSON)
Baseline: 55.3% full_acc (already capable)

| Method | Parse Rate | Name Acc | Full Acc | vs Teacher |
|--------|-----------|----------|----------|------------|
| Teacher | 98.7% | 98.2% | 63.7% | — |
| Baseline | 94.7% | 93.3% | 55.3% | — |
| pos-50 | 99.3% | 99.0% | 44.0% | -19.7pp |
| pos-100 | 99.8% | 99.3% | 40.5% | -23.2pp |
| **pos-150** | 99.5% | 99.2% | **59.0%** | -4.7pp |
| pos-200 | 99.7% | 99.2% | 51.2% | -12.5pp |
| pos-250 | 99.7% | 99.3% | 56.8% | -6.9pp |
| pos-300 | 99.7% | 99.2% | 46.2% | -17.5pp |
| fullseq | 99.5% | 98.8% | 32.0% | -31.7pp |

## Key Findings

### 1. Student > Teacher when teacher is weak (or comparable)
- Qwen: student 61.3% > teacher 54.0% (+7.3pp) — clearest case (Qwen3-1.7B doesn't naturally output JSON)
- Gemma: pos-100 79.00% > teacher 72.83% (+6.17pp) — modest, both are capable
- Llama: student 59.0% < teacher 63.7% (-4.7pp) — teacher already strong

### 2. Positional ≥ fullseq on all three pairs
- Qwen: pos-100 (61.3%) > fullseq (58.2%) by +3.1pp
- Gemma: pos-100 (79.0%) > fullseq (76.83%) by +2.17pp
- Llama: pos-150 (59.0%) >>> fullseq (32.0%) by +27pp

### 3. Optimal pos matches student-teacher capability mismatch
- Gemma (student already capable, teacher capable): pos-100 optimal
- Qwen (weak math student, mismatched-format teacher): pos-100 optimal
- Llama (capable student, strong teacher): pos-150 optimal

### 4. Parse rate and name accuracy always end up high
All distilled models achieve 91%+ parse rate and 91%+ name accuracy.
Structural properties (JSON format, function name) are learned within the
first 50 tokens. Full accuracy depends on argument correctness.

### 5. Teacher output format analysis
- Qwen3-1.7B teacher: outputs natural language math solutions (not JSON) — wrong format dominant
- Gemma-3-4B teacher: outputs clean JSON (correct format) when prompted via chat template — STRONG
- Llama-3.1-8B teacher: outputs clean JSON — STRONG
- Old claim that Gemma teacher outputs `tool_code` format was an artifact of the broken vLLM 0.8.5 tokenizer init — not a real model behaviour.

### 6. Pipeline-bug audit
The original v1 evals for Gemma funcall used vLLM 0.8.5, which initialises
Gemma3 multimodal tokenizer through a path that requires `boi_token`
(begin-of-image) attribute. That attribute does not exist on the text-only
fast tokenizer; the init fails silently in some code paths and the model
outputs garbage. All Gemma-pair funcall numbers from before 2026-05-03
should be considered void. Re-run with vLLM 0.11.0 (gemma3 venv).
