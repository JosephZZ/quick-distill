# V1 Verified Experiment Results

All experiments in this folder were run with **V1 original code** (`on_policy_distill_positional_v1.py`, commit `bd81eb6`).

## Code Version
- **Script**: `on_policy_distill_positional_v1.py` (864 lines, per-trajectory KL loss)
- **Loss**: `F.kl_div(t_log_probs, s_log_probs_resp, log_target=True, reduction="batchmean")` — correct reverse KL, per trajectory
- **Commit**: `bd81eb6` (original code before batch path was added)
- **Patches applied**:
  - `.jsonl` dataset loading: `load_dataset("json", data_files=...)` for funcall data
  - No other modifications

## Why V1
The newer code introduced two bugs in the batch loss path:
1. **KL direction bug**: `exp(t_lp) * (t_lp - s_lp)` = forward KL instead of reverse KL (fixed in commit `0a0b91e`)
2. **Indentation bug**: `mb_loss=0` initialization inside `if vocab_mapping` block, causing same-tokenizer path to have zero loss (fixed in commit `92ad82f`)

All experiments were re-run with V1 code to ensure correctness.

## Common Config
- **LoRA**: r=32, alpha=64, targets: q/k/v/o/gate/up/down_proj
- **Training**: lr=5e-5, temperature=0.7, n_samples=1, bs=16, mini_bs=1, 200 steps
- **Num problems**: 3200 (chunk_size=16 with 200 problems, or num_problems=3200 with n_samples=1)

## Files
- `1.7b_math.md` — 1.7B teacher math (MATH-500) results + position sweep
- `1.7b_coding.md` — 1.7B teacher coding (HumanEval/MBPP) results
- `1.7b_funcall.md` — 1.7B teacher function calling (BFCL) results
- `4b_math.md` — 4B teacher math results
- `4b_coding.md` — 4B teacher coding results (from scaling_results.md, potentially non-V1)
- `4b_funcall.md` — 4B teacher funcall results (from scaling_results.md, potentially non-V1)
- `gemma.md` — Gemma cross-family results (uses new code, not V1)
- `training_status.md` — Live training status tracker
