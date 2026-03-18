# Project: DFT-Distill (On-Policy Knowledge Distillation)

## Critical Rules
- **NEVER kill GPU processes without explicitly confirming with the user first.** This is the most important rule.
- **GPU 1 only** for all experiments. Never touch GPU 0.
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

### Core Files (root)
- `on_policy_distill_positional.py` — Main training script (HF generate, positional loss, progressive mode, auto-clamp)
- `eval_math500.py` — MATH-500 evaluation (pass@k, avg@k, maj@k via vLLM)
- `vllm_generate.py` — vLLM generation subprocess (used by eval, and optionally by training with `--use_vllm`)

### Eval Scripts (`scripts/`)
- `eval_humaneval.py` — HumanEval/MBPP code evaluation (vLLM generation + evalplus scoring)

### Analysis Scripts (`scripts/`)
- `kl_analysis_v3.py` — Per-token KL position analysis (vLLM gen + HF scoring)
- `kl_after200_analysis.py` — KL comparison by position range (HF generation)
- `kl_after200_analysis_v2.py` — Same but with vLLM generation
- `fullseq_degradation_analysis.py` — Full-seq \boxed{} repetition analysis
- `analyze_generation_behavior.py` — Generation behavior patterns
- `token_classification_analysis.py` — Token content classification

### Results & Docs (`docs/`)
- `main_results.md` — Main results: Math (n1bs16) + Coding (n1), pos vs fullseq, LoRA vs FullFT
- `batch_config_comparison.md` — Batch config comparison: n1bs1 vs n1bs16 vs n16bs16
- `kl_position_analysis.md` — KL position distribution and cumulative analysis
- `token_classification_analysis.md` — Token classification at high-KL positions
- `fullseq_degradation_analysis.md` — Full-seq boxed repetition analysis
- `generation_behavior_analysis.md` — Cascade effect analysis
- `kl_position_analysis_v3/` — KL analysis plots and data
- `archive/` — Superseded reports (positional_distillation_results.md, coding_distillation_results.md, etc.)

### Directories
- `checkpoints/` — Training checkpoints (`<experiment>/step_{50,100,...,400}/`)
- `logs/` — Training and eval logs, organized by `task_student_teacher_setting/`
  - **Math LoRA**: `math_qwen2.5-1.5B_qwen3-1.7B_pos_lora_{n16bs16,n1bs1,n1bs16}/`
  - **Math LoRA fullseq**: `math_qwen2.5-1.5B_qwen3-1.7B_fullseq_lora_n16bs16/`
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
- **Full-seq answer extraction**: Full-seq models repeat `\boxed{}` 58-88 times after step 50, corrupting last-boxed extraction. Use first-boxed extraction for accurate scoring. True accuracy is stable at ~65%, not the apparent 43%.
- **Eval merge on CPU**: When merging LoRA for eval, do it on CPU (`CUDA_VISIBLE_DEVICES=""`) to leave GPU memory free for vLLM.
- **vLLM GPU memory**: After offloading models to CPU, PyTorch still reserves ~22GB. Use `gpu_memory_utilization=0.50` when models are loaded, `0.85` when GPU is free.

## Experiment Structure
- Checkpoints: `checkpoints/<experiment-name>/step_{50,100,150,200}/`
- Evals: `checkpoints/<experiment-name>/eval_step_{50,100,150,200}/summary.json`
- Merged models for eval: `checkpoints/<experiment-name>/_eval_merged_step_N/`

## Training Configurations

### LoRA (default)
- LoRA: r=32, alpha=64, targets: q/k/v/o/gate/up/down_proj
- lr=5e-5, temperature=0.7, save every 50 steps, 200 steps total
- 200 problems, n_samples=16 (or 1 for large-scale)

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

### Math (MATH-500)
See `docs/positional_distillation_results.md` for full results.

| Method | Best Step | avg@4 | maj@4 | pass@4 |
|--------|-----------|-------|-------|--------|
| No distill baseline | — | 50.95% | 61.2% | 72.8% |
| Pos-5 | 150 | 56.50% | 64.8% | 76.0% |
| Pos-10 | 50 | 59.50% | 66.4% | 75.8% |
| Pos-20 | 50 | 60.20% | 67.6% | 77.0% |
| Pos-50 | 50 | 62.45% | 68.8% | 77.0% |
| Progressive 1→200 | 200 | 62.30% | 69.0% | 78.8% |
| Pos-100 | 150 | 64.25% | 69.6% | 80.2% |
| Pos-150 | 100 | 65.70% | 71.4% | 77.4% |
| Full-seq (first-boxed) | 150 | 65.55% | — | — |
| **Pos-200tok** | **100** | **66.75%** | **72.0%** | **80.2%** |

#### n1bs16 LoRA (3200 problems)

| Method | Best Step | avg@4 | maj@4 | pass@4 |
|--------|-----------|-------|-------|--------|
| Pos-50 | 150 | 66.65% | 71.0% | 81.0% |
| Pos-100 | 200 | 65.85% | 70.8% | 79.8% |
| Pos-150 | 100 | 66.65% | 67.0% | 81.0% |
| Pos-200tok | 50 | 66.05% | 71.2% | 81.0% |

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
