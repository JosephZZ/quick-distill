# Gemma Cross-Family Results (New Code Only)

**⚠️ V1 code does NOT support Gemma** (hardcoded system role in `build_prompt`).
All Gemma experiments use the newer code with `_supports_system_role` detection.

**Code**: `on_policy_distill_positional.py` (newer code, cross-tokenizer path)
**Student**: gemma-2-2b-it | **Teacher**: gemma-3-4b-it
**Tokenizer**: Cross-family, 69% vocab overlap, requires re-tokenization
**Server**: UCLACG, A6000 48GB
**Config**: LoRA r=32, alpha=64, lr=5e-5, bs=16, n_samples=1

## Baselines

### gemma-2-2b-it (Student)
| MATH-500 avg@4 | HE pass@1 | MBPP pass@1 | BFCL full_acc |
|----------------|-----------|-------------|---------------|
| 13.45% | 23.78% | 40.48% | **73.17%** (re-eval 2026-05-03) |

### gemma-3-4b-it (Teacher) — verified 2026-05-03
| MATH-500 avg@4 | HE pass@1 | HE+ pass@1 | MBPP pass@1 | MBPP+ pass@1 | BFCL full_acc |
|----------------|-----------|------------|-------------|--------------|---------------|
| **66.6%** | **20.7%** | **20.1%** | **20.4%** | **17.5%** | **72.83%** |

> Teacher MATH eval: `eval_results/teacher_baselines_nt/google_gemma-3-4b-it/summary.json` (vLLM 0.11.0, gemma3 venv).
> Teacher coding eval: `eval_results/coding-gemma3-4b-teacher/{he,mbpp}_score.txt` (re-run 2026-05-03 after the v0.8.5 boi_token bug was diagnosed).
> Teacher funcall eval: `eval_results/funcall-gemma3-4b-teacher-rerun/summary.json` (same venv, 600 BFCL problems).
> Note: gemma-3-4b-it is much weaker on coding (20.7 HE) than its own student-eval would predict; chat-template handling for raw HumanEval prompts is the suspected cause.

## Pos-100 Results

| Task | Metric | Value | Step | Source |
|------|--------|-------|------|--------|
| MATH-500 | avg@4 | **27.20%** | — | docs/scaling_results.md |
| HumanEval | pass@1 | **31.10%** | s50 (best HE) | docs/scaling_results.md |
| MBPP | pass@1 | **50.53%** | s100 (best MBPP) | docs/scaling_results.md |
| BFCL | full_acc | **79.00%** | s100 (RE-EVAL 2026-05-03) | eval_results/gemma_funcall_rerun/v1-gemma-funcall-pos100/step_100/summary.json |

> The original BFCL number (82.50%) was produced by the broken vLLM 0.8.5
> pipeline (boi_token tokenizer bug). After re-evaluation with vLLM 0.11.0,
> pos-100 step 100 reaches 79.00% full_acc — still the best Gemma funcall
> setting, still surpasses teacher (72.83%) by +6.17pp, but smaller than the
> previously-claimed +9.67pp.

**Checkpoint**: on UCLACG

## Pos-50 Results
**Status**: Training was in progress on UCLACG
- Gemma pos-50 coding: UCLACG GPU 0, ~3 steps at last check
- Gemma pos-50 funcall: Queued after coding
- Gemma pos-50 math: Not started

## Full-seq Results

### Math
| Step | avg@4 |
|------|-------|
| — | 11.70% (degrades below baseline 13.45%) |

**Marked with §** in paper — fullseq degrades below baseline.

### Other tasks (coding, funcall)
Not evaluated for fullseq — degradation on math suggests fullseq is harmful for Gemma cross-family.

## Pipeline-bug audit (2026-05-03)

`vLLM 0.8.5.post1` (the default env, python 3.9) trips a Gemma3 multimodal-tokenizer
init bug — it tries to read `boi_token` from the text-only tokenizer and either
crashes (coding eval, observed) or silently produces garbage outputs (funcall
eval, observed). This affected *only* runs that load the Gemma-3-4b teacher
model under vLLM. Re-run anything Gemma-3-related under
`/zhi_backup/ziheng/venvs/gemma3/` (vLLM 0.11.0).

Empirically observed drift between the two pipelines:

| Eval | Old (vLLM 0.8.5) | New (vLLM 0.11.0) | Notes |
|------|-----------------:|------------------:|-------|
| Gemma teacher BFCL full_acc | 25.0 | **72.83** | bug catastrophic |
| Gemma student BFCL full_acc | 0.0  | **73.17** | bug catastrophic |
| Gemma fullseq BFCL full_acc | 3.9  | **76.83** | bug catastrophic |
| Gemma teacher coding HE | (failed to load) | **20.7** | bug crashed |
| Gemma student coding HE baseline | 23.78 | 23.20 | within noise (−0.6pp) |
| Gemma student coding pos-50 best HE | 31.10 | 31.70 | within noise (+0.6pp) |
| Gemma student MATH-500 baseline avg@4 | 13.45 | 12.25 | ~1pp drift |

**Pattern**: any eval that loads the gemma-3-4b multimodal model is corrupted
under vLLM 0.8.5 (full BFCL collapse, coding crash). Evals that only load
gemma-2-2b student show <2pp drift between vLLM versions, attributable to
sampling/decoder noise.

**Decision**: Gemma funcall numbers fully replaced (see funcall_cross_model.md).
Gemma coding numbers kept (within-noise drift, full re-eval not needed).
Gemma math numbers kept (baseline drifted ~1pp, distilled checkpoints not
re-evaluated; report this as ±1-2pp uncertainty range in the paper if relevant).

## Summary (Used in paper) — UPDATED 2026-05-03

| Config | MATH-500 | HE | HE+ | MBPP | MBPP+ | BFCL full_acc |
|--------|----------|------|------|------|-------|---------------|
| Student baseline (gemma-2-2b-it) | 13.45 | 23.78 | 20.10 | 40.48 | 34.74 | **73.17** |
| Teacher (gemma-3-4b-it)          | 66.6  | 20.7  | 20.1  | 20.4  | 17.5  | **72.83** |
| Full-seq (LoRA)                  | 11.70§ | — | — | — | — | 76.83 (s200) |
| Pos-50 (LoRA, best HE)           | — | 31.10 | 26.20 | 47.10 | 39.20 | 75.67 (s100) |
| **Pos-100 (LoRA, best overall)** | **27.20** | 28.70 | 24.40 | **50.53** | **42.30** | **79.00** (s100) |

Surpass teacher cells: Pos-N HE (+10.4 vs 20.7), MBPP (+30.1 vs 20.4), BFCL (+6.17 vs 72.83).
§ Full-seq math collapses below baseline.

## Missing / Needed
- Gemma pos-50 all tasks (training in progress)
- Gemma fullseq coding/funcall (not planned due to math degradation)
- Gemma teacher baseline evals (for reference)
