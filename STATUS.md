# Project Status — 2026-04-30 (post-V5-7, autonomous overnight loop concluded)

> **Active:** v5 narrative restructure complete. See `docs/v5_overnight_plan.md` for the per-round log
> and `paper/REVISION_LOG.md` for V5-1 through V5-7 details.
> Three-part structure: (1) what is a good signal indicator? (2) robustness of the claim, (3) why position works.
>
> **Paper file:** `paper/main_v5.tex` (12pp PDF: 10 main + 1 ref + 1 appendix; 586 KB).
> **Deck file:** `deck/slides.md` (16 slides; pages in `deck/pages/`).
> **Paper and deck are now in sync** as of V5-7 (multi-seed stability, prefix-swap recovery, Gemma row).
>
> ### Seven review rounds (autonomous, overnight):
> - **V5-1** Area-chair persona — 8-dim rubric. Abstract softened (BFCL 5–7pp → 2/3 pairs by 5.9–7.3pp;
>   FullFT-math loss surfaced); single-seed daggers in tables 3 & 5; mode-seeking softened to interpretation;
>   early-stop counterargument in §4.3; Part 2 retitled "Robustness of the Part 1 Claim"; bridge paragraphs
>   §3→§4, §4→§5; conclusion aphorism dropped.
> - **V5-2** Hostile entropy-KD advocate — exposed the "we test selection at fixed K=100" caveat;
>   §6 Concurrent Work fully rewritten with per-method actual losses (SelectKD, AdaKD, TSDKD, 80/20, preplan);
>   "Scope of comparison" paragraph added; §6 cuts re-arranged.
> - **V5-3** Numerical-fact verification (most damaging). Stability table inherited stale deck-era numbers;
>   replaced `57.18 ± 5.40` / `65.40 ± 0.45` with canonical `56.78 ± 6.0` / `62.42 ± 3.0`; std ratio reframed
>   from "12×" to "2.0×"; Table 3 Gemma "Family (math)" row corrected from funcall (30.9 / 3.9) to math
>   (27.20 / 11.70); prefix-swap recovery `~93%` → `~88%` with explicit arithmetic.
> - **V5-4** Theory-rigor / flow lens — 8-dim rubric on logic flow, falsifiability, mechanism, notation.
>   Notation pass at §2 ($K=N$ for prefix; "Pos-200tok" defined); §5.2 mode-seeking caveat promoted to opening
>   paragraph as "Disclaimer up front"; new `\textbf{The cleanest falsifier.}` paragraph at §7 + §8 conclusion
>   tying back to format-mask ablation.
> - **V5-5** V5-4 deferred items — cut Fig 3 cross_family (never `\ref`'d, pure visual duplicate of Table 3);
>   added §3.2 footnote forward-pointer to §6 scope-of-comparison; fixed `\ref{sec:related}` → `\ref{sec:concurrent}`.
> - **V5-6** §4.2 explicit conjecture for the two non-winning cells (FullFT-math: smaller LR tolerates
>   tail noise; Llama BFCL: raw student already close to teacher, signal lives in tail-format detail);
>   §7 falsifier folded into main Limitations paragraph for vertical compression.
> - **V5-7** Deck/paper sync — caught by 30-min cron self-check. The paper had been corrected in V5-3 but
>   the deck still carried `57.18 ± 5.40`, `65.40 ± 0.45`, `~93% prefix recovery`. Root cause: a 1.2-pp typo
>   on seed-42 full-seq (`63.55` instead of canonical `62.35`) propagating through deck stability tables.
>   Fixed in 7 deck files: `08_stability.md`, `08b_degradations.md`, `v5_08_generality.md`, `v5_01_setup.md`,
>   `01_problem.md`, `13_summary.md`, `v5_11_cascade.md`.
>
> ### Carry-over for human / pre-submission:
> - **9pp NeurIPS final compression.** Currently 10 pages main; conclusion lands on page 10. Candidates
>   to cut: §3.1 example-token list, §4.4 N-elbow rule-of-thumb, §2 n1bs16 hyperparameter line → appendix.
> - **Wall-clock / memory timing log.** "9.5h / 1.0h, 38GB / 9GB" claims are deck-only sources; needs a
>   timing log dropped into `docs/` before submission.
> - **Entropy-threshold sweep on UCLACG GPU 0.** Placeholder still PENDING in deck/paper. Requires user to
>   authorize remote check or pull results.
> - **Suffix-only ablation.** Cited as the "cleanest falsifier" in §7 but not yet run.



> Living snapshot. Update freely. The static repo map lives in `README.md`.

---

## Paper deadline

**NeurIPS 2026.** Submission within days. The active paper is `paper/main_v5.tex`
(rendered: `paper/main_v5.pdf`). Revision audit log: `paper/REVISION_LOG.md` (rounds V5-1 through V5-7).

The deck `deck/slides.md` is the canonical story; the paper follows it. As of V5-7 (2026-04-30) the deck
and paper agree on all multi-seed stability numbers and the prefix-swap recovery percentage.

---

## Current paper scope (as sold by the v2 deck — 16 slides)

Three sales:

1. **Sale #1 — Planning > execution.** Loss on the first ~100 response tokens matches
   or beats full-sequence loss. Stable across seeds; full-sequence collapses on
   2/3 seeds and across families.
2. **Sale #2 — Position > entropy.** Prefix-100 (45% KL coverage) beats top-KL-100
   (95% KL coverage) and beats top-entropy-100 in every selection mode tested.
   Position is a *causal* proxy, not an entropy proxy.
3. **Sale #3 — 10× efficiency.** One-line loss change. LoRA + prefix-100 trains in
   ~1/10 the wall-clock with strictly better stability than full-sequence.

Bonus: **mode-seeking on function calling.** Reverse-KL is mode-seeking, so the
student can commit to a JSON mode that the teacher distribution supports but
doesn't peak on. Student exceeds teacher by up to 9.7 pp on BFCL.

Cross-validation: Qwen 1.5B→1.7B (math, code, funcall), Gemma 2B→4B (math, funcall),
Llama 1B→8B (math, funcall). Four scales. Two training methods (LoRA, full FT).

---

## Headline numbers (locked in)

**MATH-500, Qwen2.5-Math-1.5B → Qwen3-1.7B, n1bs16 LoRA, 200 steps:**

| Method        | Best avg@4 | Best step |
|---------------|-----------:|----------:|
| Baseline      | 50.95%     | —         |
| Pos-50        | 66.65%     | 150       |
| **Pos-100**   | **65.85%** | 200       |
| Pos-150       | 66.65%     | 100       |
| Pos-200tok    | 66.05%     | 50        |
| Full-seq      | 62.35%     | 150       |

Top-K selections at K=100, same training budget:
top-KL = 58.60%, top-entropy-student = 61.35%, random = 63.05%, **prefix = 65.85%**.

Multi-seed stability (3 seeds, n1bs16): pos-100 = 62.42 ± 3.0; full-seq = 56.78 ± 6.0
(std ratio 2.0×; full-seq worst seed 50.45 falls below baseline 50.95).

**Cross-family math (best positional vs full-seq):**

| Pair           | Pos best        | Full-seq |
|----------------|----------------:|---------:|
| Qwen 1.5→1.7B  | 65.85% (N=100)  | 62.35% (mean across seeds 56.78; worst seed 50.45 < baseline) |
| Gemma 2→4B     | 27.20% (N=100)  | 11.70% (degradation) |
| Llama 1→8B     | 59.0% (N=150)   | 32.0%  (degradation) |

**BFCL function calling (full_acc):**

| Pair           | Teacher | Student | **Pos best**  | Full-seq |
|----------------|--------:|--------:|--------------:|---------:|
| Qwen 1.5→1.7B  | 54.0%   | 2.7%    | **61.3%** (N=100) | 58.2% |
| Gemma 2→4B     | 25.0%   | 0%      | **30.9%** (N=50)  | 3.9%  |
| Llama 1→8B     | 63.7%   | 55.3%   | 59.0% (N=150) | 32.0% |

Position-vs-entropy quartile evidence (head-N tokens, breakdown over surprise quartiles):

| Head N | Q1 share (top-25%) | Q4 share (bottom-25%) | Q1 enrichment |
|-------:|-------------------:|----------------------:|--------------:|
| 25     | 70.0%              | 7.4%                  | 2.8×          |
| 50     | 61.0%              | 13.1%                 | 2.4×          |
| 100    | 53.2%              | 17.3%                 | 2.1×          |
| 200    | 42.6%              | 23.1%                 | 1.7×          |
| 500    | 31.8%              | 27.1%                 | 1.3×          |

Head-100 covers 38.4% of the >p95 surprise mass. To cover 95% you need first 846
positions. Position and entropy are correlated, not identical.

---

## In-flight experiments (UCLACG, GPU 0 + GPU 1)

> Watchdogs poll every 5 min and restart on crash. They exit when `step_200/`
> appears in the checkpoint dir.

| GPU | Experiment                                  | Status      | Notes                              |
|----:|---------------------------------------------|-------------|------------------------------------|
| —   | `hi_kl_hi_surp_topk` K=100                  | **DONE**    | eval_final avg@4 = **56.75%** (maj 65.4, pass 73.6). Adds support for §3 "high-KL gating fails" — even surprise-conditioned top-K underperforms full-seq 62.35 / pos-100 65.85. Not yet slotted into paper. |
| 1   | `hi_kl_hi_ent_topk` half-length (`top_k_frac=0.5`) | **DONE step_200** | _merged_latest exists; **eval pending** (no eval_* dir yet). Watchdog exited 2026-04-29 05:09. |
| —   | `hi_kl_hi_ent_topk` K=100                   | **DONE step_200** | Finished 2026-04-29 17:32. _merged_latest exists; **eval pending**. Last log step=190 loss 1.88 kl 1.19 lr 3.80e-7 — loss DID NOT decrease through training (~1.88 throughout), suggesting joint top-K-by-(KL × entropy) may be a worse selector than full-seq even before eval. |
| 0   | `topent_student` K=200                      | running     | NEW — auto-launched by `queue_topent_k200.sh` when topk100 step_200 landed. PID 3597518, GPU 0 23 GB. |
| 1   | `hi_ent` H≥0.01 prefix-100 (entropy-thr sweep, ~top-56% mass) | running | step_100 saved (loss 0.69 ↓ from 0.77). **Path bug**: `output_dir` is at `/zhi_backup/ziheng/quick-distillation/quick-distillation/checkpoints/...` (nested) — eval will need the nested path. |
| 0   | `top_entropy_student` K=200 (queued)        | queued      | Will launch when `hi_kl_hi_ent_topk K=100` hits step_200. |
| —   | `hi_ent` H≥0.664 (top-20%)                  | **NOT QUEUED** | Entropy-threshold sweep variant 2 of 3 — paper §4 placeholder still PENDING. |
| —   | `hi_ent` H≥0.0004 (top-80%)                 | **NOT QUEUED** | Entropy-threshold sweep variant 3 of 3 — paper §4 placeholder still PENDING. |

Watchdog scripts (on UCLACG, see user memory `experiment_servers.md`):
- `/tmp/watchdog_ent_half.sh` — GPU 1 ent_half restart loop (exited DONE)
- `/tmp/queue_ent_topk100.sh` — GPU 0 chain (still ticking until step_200 of `hi_kl_hi_ent_topk`)
- `/tmp/queue_topent_k200.sh` — GPU 0 final link (waits on the chain)

**Action items surfaced by 2026-04-30 self-check (V5-8 polling):**
1. Eval `hi_kl_hi_ent_half`'s `_merged_latest` — datapoint sitting unused since 2026-04-29.
2. After GPU 1 frees (when `hi_ent` H≥0.01 finishes), launch the two missing entropy-threshold variants (H≥0.664 = top-20%, H≥0.0004 = top-80%) so the §4 paper placeholder can be filled.
3. Decide whether to slot the `hi_kl_hi_surp_topk` 56.75% number into paper §3 or §6 as additional negative evidence (currently paper says only "in-flight").

---

## Open work items (pre-submission)

- [ ] Add `--position_lower` flag for suffix-only experiments
- [ ] Run suffix-100+ and suffix-100+_hiKL_hiE
- [ ] Pull final numbers from in-flight UCLACG experiments
- [ ] Last paper polish round (rubric: see `paper/REVISION_LOG.md`)
- [ ] Verify deck v2 builds (`cd deck && npm i && npm run dev`)
- [ ] Decide whether to merge `slides_v2.md` into `slides.md` permanently
      (currently identical files; `slides_v1_backup.md` holds the previous ordering)

---

## Recently shipped

- Restructured deck to evidence-first ordering (16 pages, mode-seeking is bonus)
- Position-vs-entropy quartile analysis (paper appendix + slide 9)
- Test-time prefix swap evidence (slide 6: `pages/16_continuation.md`)
- `hi_kl_hi_ent_topk` training mode + `--top_k_frac` arg
- Repo cleanup (this audit) — see `archive/` for everything moved

---

## Server cheatsheet

- `scai5` — selective token experiments
- `infowave-develop` — no GPU, storage only; GPU workers write to `/sg-pvc/`
- `UCLACG` — current main GPU server (ProxyJump via `lion`)
- `/mnt/ziheng` on UCLACG = NFS (slow HDD); `/zhi_backup/ziheng` = SSD (use this)

See user-memory for SSH alias details.

---

## Hand-off checklist for the next person

1. Read `README.md` then this file.
2. Render `deck/slides.md` and walk through the 16 slides.
3. Skim `paper/main_v2.tex` (sections 1–5 are tightest).
4. Inspect any one in-flight experiment via the watchdog logs on UCLACG.
5. The training code is `on_policy_distill_positional.py`; eval is `eval_math500.py`.
6. **Do not touch GPU 0 on the local machine.** (See `CLAUDE.md`.)
