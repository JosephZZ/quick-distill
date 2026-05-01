# Project: DFT-Distill (On-Policy Knowledge Distillation)

## Critical Rules
- **NEVER kill GPU processes without explicitly confirming with the user first.** This is the most important rule.
- Use only 1 GPU for eval unless user says otherwise.

## Design Principles

### Backward Compatibility
New experiments add features — they do NOT modify existing behavior. Old experiments must remain reproducible with their original settings. When adding a new training mode or loss variant, add it as a new flag/option rather than changing default behavior.

### Efficiency First
- **Inference**: Use vLLM for long-sequence generation (full-seq). Use HF generate for positional experiments (short sequences, no vLLM overhead needed).
- **Generation length**: Never generate more tokens than needed. `max_new_tokens` should match `position_limit` for positional experiments (auto-clamped in code).
- **Don't waste compute**: If a position limit is 50, only generate 50 tokens. Don't generate 3584 tokens and throw away 3534.

## Project Overview
On-policy distillation: student (Qwen2.5-Math-1.5B) generates responses, teacher (Qwen3-1.7B) scores them, student trains on reverse KL loss with LoRA.

**Pipeline per step**: Student generates on-policy → Teacher scores (forward pass) → Compute reverse KL loss on first N positions → LoRA gradient update.

## Repository Structure

> **First read for any new collaborator: `README.md` (static map) and `STATUS.md` (live state).**
> The deck `deck/slides.md` is the canonical narrative.

### Core Files (root)
- `on_policy_distill_positional.py` — Main training script (HF generate, positional loss, progressive mode, auto-clamp)
- `eval_math500.py` — MATH-500 evaluation (pass@k, avg@k, maj@k via vLLM)
- `eval_funcall.py` — BFCL function-calling evaluation
- `vllm_generate.py` — vLLM generation subprocess (used by eval, and optionally by training with `--use_vllm`)
- `model_registry.json` — registered model paths

### Scripts (`scripts/`, post-2026-04-28 reorg)
- `scripts/training/` — active training launchers (UCLACG)
- `scripts/eval/` — math/format-mask eval drivers, batch merge, parallel eval
- `scripts/monitoring/` — watchdogs, status reporters, watch_*.sh
- `scripts/analysis/` — offline analyses of generated logprobs / KL profiles
  - `kl_analysis_v3.py` — Per-token KL position analysis (vLLM gen + HF scoring)
  - `kl_x_entropy_buckets.py` — Position × entropy quartile cross-tab (Sale #2 evidence)
  - `position_x_bucket.py` — Position-vs-bucket distribution
  - `format_mask_threshold.py` — format/style token thresholding
  - `strategy_kl_coverage.py` — cumulative KL coverage by selection strategy
  - `kl_after200_analysis{,_v2}.py` — KL comparison by position range
  - `fullseq_degradation_analysis.py` — Full-seq `\boxed{}` repetition pattern
  - `analyze_generation_behavior.py` — Generation behavior patterns
  - `token_classification_analysis.py` — Token content classification
- `scripts/eval_humaneval.py` — HumanEval/MBPP code evaluation
- `scripts/compute_metrics.py`, `scripts/prepare_funcall_data.py`, `scripts/hf_models_env.sh`

### Results & Docs (`docs/`)
- `main_results.md` — **Canonical n1bs16 results** (math + coding), pos vs fullseq, LoRA vs FullFT
- `kl_position_analysis.md` — KL position distribution and cumulative analysis (n1bs16 numbers)
- `token_classification_analysis.md` — Token classification at high-KL positions
- `generation_behavior_analysis.md` — Cascade effect analysis
- `archive/n16bs16_legacy_results.md` — **Legacy n16bs16/n1bs1 numbers, do NOT quote in paper**
- `archive/fullseq_degradation_n16bs16_only.md` — Legacy fullseq `\boxed{}` repetition analysis (n16bs16 only — n1bs16 fullseq does not exhibit this)
- `prefix_continuation_analysis.md` — Test-time prefix swap (handoff effects)
- `funcall_results.md`, `format_mask_results.md` — task-specific results
- `paper_outline_v6.md` — outline matching `paper/main_v2.tex`
- `conceptual_framework_review.md` — review of the position-as-causal-proxy framing
- `suffix_experiment_plan.md` — pending suffix experiments
- `kl_position_analysis_v3/`, `signal_analysis/` — supporting raw data
- `archive/` — superseded outlines, missing-experiments lists, v2 KL analysis

### Archive (`archive/`)
- `paper_v1/` — previous `main.tex` + build artifacts
- `figures_legacy/` — exploration figures not used in `main_v2.tex`
- `scripts_legacy/` — old runners (config A/C, sglang, fullseq queues, top-token harness)
- `docs_legacy/` — placeholder

### Paper (`paper/`)
- `main_v2.tex`, `main_v2.pdf` — active submission
- `references.bib`, `references_extended.bib` — bibs
- `neurips_2026.sty` — style file
- `REVISION_LOG.md` — rubric scoring rounds
- `figures/` — only the 5 figures used in `main_v2.tex` (legacy moved to `archive/figures_legacy/`)

### Deck (`deck/`)
- `slides.md` — active 16-slide v2 deck (evidence-first ordering)
- `slides_v1_backup.md` — previous ordering, kept for reference
- `pages/*.md` — per-slide fragments
- `public/images/` — deck-only image copies of the 5 paper figures

### Directories
- `checkpoints/` — Training checkpoints (`<experiment>/step_{50,100,...,400}/`)
- `logs/` — Training and eval logs, organized by `task_student_teacher_setting/`
  - **Math LoRA (canonical)**: `math_qwen2.5-1.5B_qwen3-1.7B_pos_lora_n1bs16/` ← USE THIS
  - **Math LoRA fullseq (canonical)**: see `checkpoints/fullft-fullseq-n1/` and n1bs16 fullseq LoRA logs
  - **Math LoRA legacy** (n16bs16, n1bs1) — kept on disk only; numbers archived in `docs/archive/n16bs16_legacy_results.md`. **Do NOT quote in paper without explicitly tagging the config.**
  - **Math FullFT**: `math_qwen2.5-1.5B_qwen3-1.7B_{pos_fullft,fullseq_fullft}/`
  - **Math early**: `math_qwen2.5-1.5B_qwen3-1.7B_reverse-kl-dft/`
  - **Math analysis**: `math_qwen2.5-1.5B_qwen3-1.7B_{kl-analysis,forgetting-eval}/`
  - **Coding LoRA**: `coding_qwen2.5-1.5B_qwen3-1.7B_{pos-lora,fullseq_lora}/`
  - **Coding FullFT**: `coding_qwen2.5-1.5B_qwen3-1.7B_{pos-fullft,fullseq_fullft}/`
- `archive/` — Legacy scripts, shell files, and old eval outputs

## Key Files Detail

### on_policy_distill_positional.py
Main training script. Key args:
- `--position_limit N`: Only compute loss on first N response tokens (0 = full sequence)
- `--max_new_tokens N`: Max generation length (auto-clamped to position_limit if set)
- `--use_vllm`: Use vLLM for generation (for full-seq); omit for positional (uses HF generate)
- `--progressive_positions`: Linearly increase position_limit from 1 to max over training
- `--n_samples`: Trajectories per problem
- `--chunk_size`: Problems per batch
- `--loss_type`: `reverse_kl` (default)

### eval_math500.py
Eval script. Takes `--model` (path to merged model or HF name). Uses vLLM internally.
- Does NOT accept `--lora_path` — must merge LoRA first, then pass merged path.
- Answer extraction uses `last_boxed_only_string()` — known issue with full-seq models that repeat \boxed{}.

## Important Bugs & Gotchas
- **unmerge_adapter**: After `merge_adapter()` + save, MUST call `unmerge_adapter()` or LoRA training breaks (all subsequent checkpoints become identical).
- **Full-seq answer extraction (n16bs16 ONLY)**: In the legacy n16bs16 fullseq math run, models repeat `\boxed{}` 58-88 times after step 50. **Does NOT happen in n1bs16 math fullseq** (stable 61-62% across all steps). Coding/funcall fullseq do degrade in n1bs16, but via different failure modes. See `docs/archive/fullseq_degradation_n16bs16_only.md` for the legacy analysis.
- **Eval merge on CPU**: When merging LoRA for eval, do it on CPU (`CUDA_VISIBLE_DEVICES=""`) to leave GPU memory free for vLLM.
- **vLLM GPU memory**: After offloading models to CPU, PyTorch still reserves ~22GB. Use `gpu_memory_utilization=0.50` when models are loaded, `0.85` when GPU is free.

## Experiment Structure
- Checkpoints: `checkpoints/<experiment-name>/step_{50,100,150,200}/`
- Evals: `checkpoints/<experiment-name>/eval_step_{50,100,150,200}/summary.json`
- Merged models for eval: `checkpoints/<experiment-name>/_eval_merged_step_N/`

## Training Configurations

> **Canonical recipe (n1bs16) — use this unless explicitly noted.**
> Older n16bs16 results exist on disk and are archived in
> `docs/archive/n16bs16_legacy_results.md`. **Never quote n16bs16 numbers in
> the paper, deck, or new analysis.** They use a different recipe (200
> problems × 16 trajectories vs 3200 problems × 1 trajectory) and the
> numbers are not directly comparable.

### LoRA (default = n1bs16)
- LoRA: r=32, alpha=64, targets: q/k/v/o/gate/up/down_proj
- lr=5e-5, temperature=0.7, save every 50 steps, 200 steps total
- **3200 problems, n_samples=1, batch_size=16 (n1bs16)**

### Full Finetune
- No LoRA (`--full_finetune`), lr=5e-6
- Same structure otherwise

## Eval Configuration

### Math (MATH-500)
- n_samples=4, temperature=0.7
- vLLM: max_model_len=4096, gpu_memory_utilization=0.70

### Coding (HumanEval/MBPP)
- n_samples=1, temperature=0.0, max_tokens=512
- Uses evalplus for HumanEval+/MBPP+ scoring
- LoRA checkpoints must be merged before eval (merge on CPU)

## Completed Experiments

### Math (MATH-500) — n1bs16 LoRA (canonical, 3200 problems)
See `docs/main_results.md` for full per-step tables. **All numbers below are n1bs16.**

| Method | Best Step | avg@4 | maj@4 | pass@4 |
|--------|-----------|-------|-------|--------|
| No distill baseline | — | 50.95% | 61.20% | 72.80% |
| Top-KL-100 | 50 | 58.60% | 65.80% | 74.40% |
| Top-Entropy-Student-100 | 100 | 61.35% | 67.20% | 73.20% |
| format_mask (drop ~6% format toks) | 50 | 62.05% | 67.20% | 76.20% |
| Top-Entropy-Teacher-100 | 150 | 62.20% | 67.80% | 75.80% |
| **Full-seq (n1bs16)** | **150** | **62.35%** | **69.40%** | **74.60%** |
| Random-100 | 150 | 63.05% | 69.20% | 76.80% |
| Pos-100 | 200 | 65.85% | 70.80% | 79.80% |
| Pos-200tok | 50 | 66.05% | 71.20% | 81.00% |
| **Pos-50** | **150** | **66.65%** | **71.00%** | **81.00%** |
| Pos-150 | 100 | 66.65% | 67.00% | 81.00% |

(Pos-5/10/20 and pos-200tok = 66.75% are **n16bs16-only** — see
`docs/archive/n16bs16_legacy_results.md`. Do not quote without tagging.)

### Coding (HumanEval/MBPP)
See `docs/coding_distillation_results.md` for full results.

| Experiment | Best Step | HE | HE+ | MBPP | MBPP+ |
|------------|-----------|------|------|------|-------|
| LoRA pos-50 | 350 | **42.1** | **36.6** | 46.6 | 41.5 |
| LoRA pos-100 | 150 | **42.1** | 36.0 | 49.2 | 44.4 |
| LoRA pos-150 | 250 | 41.5 | **37.8** | 49.5 | 44.7 |
| LoRA fullseq | 50 | 40.2 | 35.4 | 52.6 | 46.3 |
| FullFT pos-250 | 350 | 37.8 | 31.7 | 54.2 | 47.4 |
| FullFT pos-150 | 300 | 36.6 | 31.1 | **54.2** | **46.8** |
| FullFT fullseq | 150 | 36.6 | 31.7 | 53.2 | 46.0 |
