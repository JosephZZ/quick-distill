# Where Teachers Teach — On-Policy Distillation: Positional Structure

> Research repository for the NeurIPS 2026 submission on positional knowledge distillation.
> A new collaborator + AI agent should be able to read this file end-to-end and know
> (a) what the paper claims, (b) where every artifact lives, (c) what's running right now.

**Status snapshot lives in [`STATUS.md`](STATUS.md).** Read that next. This file is the static map.

---

## 1. The paper in one minute

We train a small student LLM by reverse-KL distillation against an on-policy teacher.
Standard practice computes the loss on **all** generated tokens. We show:

1. **Planning > execution.** Loss on the first ~100 response tokens matches or beats
   full-sequence loss on math, function calling, and code, across Qwen / Gemma / Llama
   families.
2. **Position > entropy.** Despite covering only ~45% of cumulative KL and missing 60%
   of the high-surprise mass, the prefix wins against top-KL, top-entropy, and random
   selection of the same budget.
3. **10× efficiency.** One-line change in the loss; LoRA + prefix-100 trains in roughly
   1/10 the wall-clock of full-sequence with strictly better stability.

Bonus result on function calling: the small student exceeds the teacher (a *reverse-KL
mode-seeking* effect: the student commits to a JSON mode that exists in the teacher's
distribution but isn't its argmax).

The narrative as currently sold lives in `deck/slides.md` (16 slides) and
`paper/main_v2.tex`. Treat the deck as the source of truth for *story*; the paper
follows.

---

## 2. Repository map

```
quick-distillation/
├── README.md                    ← you are here
├── STATUS.md                    ← what's running, what's done, what's next
├── CLAUDE.md                    ← project rules for AI agents (GPU 1 only, etc.)
│
├── on_policy_distill_positional.py   ← MAIN training script (HF generate, positional loss)
├── eval_math500.py                   ← MATH-500 eval (vLLM, pass@k / avg@k / maj@k)
├── eval_funcall.py                   ← BFCL function-calling eval
├── vllm_generate.py                  ← vLLM subprocess wrapper
├── model_registry.json               ← registered model paths
├── data/                             ← training datasets (gitignored)
│
├── paper/                       ← NeurIPS submission
│   ├── main_v2.tex              ← active paper
│   ├── main_v2.pdf              ← rendered
│   ├── references.bib           ← active bibliography
│   ├── references_extended.bib  ← superset (cross-checked candidates)
│   ├── neurips_2026.sty         ← style file
│   ├── REVISION_LOG.md          ← rubric scores per round, fixes applied
│   └── figures/                 ← only the 5 figures used in main_v2.tex
│       ├── fig1_teaser.{pdf,png}
│       ├── fig1_kl_decay.{pdf,png}
│       ├── fig2_token_paradox.{pdf,png}
│       ├── fig3_cross_family.{pdf,png}
│       ├── fig4_funcall.{pdf,png}
│       ├── fig5_cascade.{pdf,png}
│       └── generate_teaser.py
│
├── deck/                        ← Slidev presentation (the source of truth for story)
│   ├── slides.md                ← active 16-slide v2 deck
│   ├── slides_v1_backup.md      ← previous ordering (paradox-first), kept for reference
│   ├── slides_v2.md             ← identical to slides.md (kept as label)
│   ├── pages/                   ← per-slide .md fragments
│   │   ├── 01_problem.md        ← OPD setup + failure modes
│   │   ├── 06_math_results.md   ← headline (Sale #1)
│   │   ├── 08_stability.md      ← seed stability (Sale #1 support)
│   │   ├── 05_method.md         ← efficiency (Sale #3)
│   │   ├── 12_cascade.md        ← KL leakage (mechanism)
│   │   ├── 16_continuation.md   ← test-time prefix swap (mechanism)
│   │   ├── 03_three_evidence.md ← selection comparison (Sale #2)
│   │   ├── 09_paradox.md        ← top-KL paradox (Sale #2 support)
│   │   ├── 10_resolution.md     ← position vs entropy (Sale #2 resolution)
│   │   ├── 04_cross_model.md    ← 45% KL coverage rule
│   │   ├── 14_cross_scale.md    ← cross-scale validation
│   │   ├── 07_funcall_results.md ← funcall + mode-seeking (bonus)
│   │   ├── 00_concurrent.md     ← concurrent work positioning
│   │   ├── 15_limitations.md    ← honest limits
│   │   └── 13_summary.md        ← summary
│   └── public/images/           ← deck-only image copies (mirrors paper/figures)
│
├── scripts/
│   ├── training/                ← active training launchers (UCLACG)
│   │   ├── run_hi_kl_hi_surp_uclacg.sh
│   │   ├── run_hi_kl_hi_surp_topk_uclacg.sh
│   │   ├── run_hi_kl_hi_ent_half_uclacg.sh
│   │   ├── run_hi_kl_hi_ent_topk_gpu0.sh
│   │   ├── run_topent_student_k200_uclacg.sh
│   │   ├── run_format_mask_uclacg.sh
│   │   ├── run_gemma_all.sh
│   │   └── run_gemma_fullseq_uclacg.sh
│   ├── eval/                    ← eval drivers (math, format-mask, batch merge)
│   ├── monitoring/              ← watchdogs, status reporters, watch_*.sh
│   ├── analysis/                ← offline analysis of generated logprobs / KL profiles
│   │   ├── kl_analysis_v3.py    ← per-position KL (vLLM gen + HF score)
│   │   ├── kl_x_entropy_buckets.py  ← position × entropy quartile cross-tab
│   │   ├── position_x_bucket.py     ← position-vs-bucket distribution
│   │   ├── format_mask_threshold.py
│   │   ├── strategy_kl_coverage.py  ← cumulative KL coverage by strategy
│   │   └── ...
│   ├── eval_humaneval.py        ← code eval (evalplus)
│   ├── compute_metrics.py
│   ├── prepare_funcall_data.py
│   ├── hf_models_env.sh         ← HF cache env
│   └── crontab_scaling_fullseq.txt
│
├── docs/
│   ├── main_results.md          ← CANONICAL n1bs16 results table (math + coding)
│   ├── conceptual_framework_review.md
│   ├── format_mask_results.md
│   ├── funcall_results.md
│   ├── generation_behavior_analysis.md
│   ├── kl_position_analysis.md  ← n1bs16 numbers
│   ├── prefix_continuation_analysis.md  ← test-time handoff results
│   ├── suffix_experiment_plan.md
│   ├── token_classification_analysis.md
│   ├── paper_outline_v6.md      ← latest outline (matches main_v2)
│   ├── kl_position_analysis_v3/ ← supporting raw data + plot
│   ├── signal_analysis/         ← supporting raw data
│   └── archive/                 ← superseded outlines AND legacy n16bs16/n1bs1 numbers
│       ├── n16bs16_legacy_results.md           ← n16bs16 batch-config doc (DO NOT QUOTE)
│       └── fullseq_degradation_n16bs16_only.md ← n16bs16-specific \boxed{} repetition
│
└── archive/                     ← everything kept for provenance, not for active work
    ├── paper_v1/                ← previous main.tex + build artifacts
    ├── figures_legacy/          ← exploration figures not used in main_v2
    ├── scripts_legacy/          ← old runners (config A/C, sglang, fullseq queues, etc.)
    └── docs_legacy/             ← (placeholder)
```

`checkpoints/`, `logs/`, `*.jsonl`, and LaTeX build artifacts are gitignored — see
[`.gitignore`](.gitignore).

---

## 3. Key files to read first (in order)

For a new collaborator:

1. **`STATUS.md`** — current state.
2. **`deck/slides.md`** + render the 16 slides (`cd deck && npm i && npm run dev`).
   This is the most efficient story upload.
3. **`paper/main_v2.tex`** — full claims with evidence and citations.
4. **`docs/main_results.md`** — the headline numbers.
5. **`on_policy_distill_positional.py`** — the training entry point. Understand
   `--token_select_mode`, `--position_limit`, `--top_k_frac`, `--use_vllm`.

---

## 4. Reproducing the headline (Math, Qwen 1.5B → 1.7B)

```bash
# Train: pos-100 LoRA on MATH (n_samples=1, batch=16, 200 steps, GPU 1 only)
python on_policy_distill_positional.py \
  --student_model Qwen/Qwen2.5-Math-1.5B \
  --teacher_model Qwen/Qwen3-1.7B \
  --token_select_mode prefix --position_limit 100 \
  --max_new_tokens 100 \
  --n_samples 1 --chunk_size 16 \
  --num_problems 3200 --steps 200 \
  --lr 5e-5 \
  --output_dir checkpoints/qwen_pos100_n1bs16
# Eval (merge LoRA on CPU first, then vLLM on GPU 1)
CUDA_VISIBLE_DEVICES="" python -c "..."     # see scripts/eval/batch_merge_eval_math.sh
CUDA_VISIBLE_DEVICES=1 python eval_math500.py --model checkpoints/qwen_pos100_n1bs16/_eval_merged_step_200
```

Expected: avg@4 ≈ 65.85% at step 200 (baseline = 50.95%). See `docs/main_results.md`.

---

## 5. Conventions

- **GPU 1 only.** Never touch GPU 0 on the local machine. (See `CLAUDE.md`.)
- **Backward compatibility.** New experiments add flags; never change defaults.
- **Incremental writes.** All eval scripts must write per-result, not buffered.
- **Merge LoRA on CPU before eval.** Then call `unmerge_adapter()` if continuing
  training.
- **Servers in use:** `scai5`, `infowave-develop` (storage only; GPU workers under
  `/sg-pvc/`), `UCLACG` (via `lion` ProxyJump). See user memory for details.

---

## 6. Glossary of `--token_select_mode` values

| Mode                     | What it does                                                               |
|--------------------------|----------------------------------------------------------------------------|
| `prefix`                 | First N tokens (`position_limit=N`). The headline method.                  |
| `top_kl`                 | Top-K tokens by per-position KL value (largest K out of trajectory length).|
| `top_entropy_student`    | Top-K by student-side entropy.                                             |
| `top_entropy_teacher`    | Top-K by teacher-side entropy.                                             |
| `random`                 | Uniform random selection of K tokens.                                      |
| `hi_kl_hi_surp`          | Tokens with both high KL and high student surprise (intersection).         |
| `hi_kl_hi_surp_topk`     | Same, then top-K by `KL × surprise` (per-trajectory K via `top_k_frac`).   |
| `hi_kl_hi_ent_topk`      | Same idea but full-vocab entropy in place of surprise.                     |
| `format_mask`            | Mask out format/style tokens above a learned threshold (R4 follow-up).     |

`--top_k_frac` sets K = ⌊trajectory_length × frac⌋ per trajectory. With
`--position_limit 0` plus a topk mode, K is independent of position.

---

## 7. What's been ruled out / is exploration

These directions have been tried and either superseded or set aside. Detailed
records are in `archive/scripts_legacy/` and `docs/archive/`:

- Old config A / config C runners (early naming).
- sglang inference for training (replaced by vLLM + HF generate hybrid).
- Old `top_token` shell harness (`run_top_token_experiments.sh`, root-level).
- Soft-weighting design (see `docs/archive/soft_weighting_design.md`) — abandoned in
  favor of hard token selection.
- Suffix-only experiments — planned in `docs/suffix_experiment_plan.md`, not yet run.
- Selective-token analysis — preliminary, in `docs/archive/selective_token_analysis.md`.

For an audit trail of what was tried and dropped, browse `archive/` — every move
preserved the original file.
