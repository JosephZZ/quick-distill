# V5 Overnight Plan — 2026-04-30

> Durable artifact for the autonomous 12-hour rewrite.
> If context truncates, START HERE.

## V5 Three-Part Structure

### Part 1 — "What is a good signal indicator? KL, entropy, or position?"
- §1 Intro: framing, OPD pipeline, the question of *which* tokens carry the signal
- §2 Setup: student/teacher pairs, MATH-500, n1bs16 LoRA, baselines
- §3 KL paradox: top-KL-100 = **58.60%** ≪ full-seq **62.35%**. High-KL tokens are mostly format/style
- §4 Entropy: top-ent-student = 61.35%, top-ent-teacher = 62.20%, format-mask = 62.05% — all ≈ fullseq.
  Entropy-threshold sweep (top-20% / 56% / 80%): **PENDING** — UCLACG GPU 0
- §5 Position uniquely **exceeds** fullseq (65.85% / 66.65%). Necessary-vs-sufficient framing:
  entropy is necessary but insufficient. Contiguous-prefix shape is the missing piece.

### Part 2 — "Position is a particularly great strategy for token selection"
- §6 Pos-K beats fullseq across configs (Qwen math n1bs16 + n16bs16 confirmation)
- §7 Generality (好/快/稳):
  * Cross-model: Qwen, Gemma, Llama
  * Cross-task: math, coding, BFCL
  * Cross-method: LoRA & FullFT
  * Stability: full-seq collapses 2/3 seeds; pos-K stable across 3/3
  * Speed: ~10× wall-clock; ~4× memory
- §8 N-selection: elbow at ~50–100 tokens for K2; sweep table

### Part 3 — "Further discussion: why is position so good?"
- §9 Cascading-error theory:
  * Test-time prefix swap (continuation evidence)
  * Auto-tail-KL drop without tail loss
  * Conditional-drift bound intuition
- §10 Student surpasses teacher (mode-seeking reverse-KL):
  * BFCL +9.7pp Llama
  * NEW token-level evidence: rank of correct branch in teacher distribution at planning tokens, top-3 mass, planning-vs-execution distinction
- §11 Limitations
- §12 Concurrent work
- §13 Summary

## Canonical numbers (n1bs16 LoRA, MATH-500, avg@4)

| Method                     | avg@4   |
|---------------------------|--------:|
| No-distill baseline       | 50.95   |
| Top-KL-100                | 58.60   |
| Top-ent-student-100       | 61.35   |
| Format-mask               | 62.05   |
| Top-ent-teacher-100       | 62.20   |
| **Full-seq**              | **62.35** |
| Random-100                | 63.05   |
| Pos-100                   | 65.85   |
| Pos-200tok                | 66.05   |
| **Pos-50 / Pos-150**      | **66.65** |
| Middle-100                | 47.80   |
| Last-100                  | 50.35   |

## Figure list (target)

- [ ] **fig1_paradox**: bar chart, x=method, y=avg@4, dashed fullseq line. Highlight Top-KL well below fullseq, position uniquely above. Use Part-1 colors.
- [ ] **fig2_kl_decay**: per-position KL curve (existing data, can reuse or regen).
- [ ] **fig3_signal_indicators**: panel of 3 — (a) KL paradox + (b) Entropy ≈ fullseq + (c) Position > fullseq.
- [ ] **fig4_generality**: 3×N grid of cross-model / cross-scale / cross-task results.
- [ ] **fig5_stability**: per-seed avg@4 curves over training steps for fullseq vs pos.
- [ ] **fig6_n_elbow**: pos-N sweep curve with elbow at 50–150.
- [ ] **fig7_cascade**: prefix-swap evidence — pre-distill prefix + distilled tail vs distilled prefix + pre-distill tail.
- [ ] **fig8_surpass**: rank-of-correct-branch CDF for teacher at planning tokens.

## Progress checkboxes

- [x] Cron 30-min self-check scheduled (id: 0e5ddf8e)
- [x] Master plan written (this file)
- [x] Deck `slides.md` + new `pages/v5_*.md` reordered to v5 (16 pages)
- [ ] Deck builds: `cd deck && npm run dev` (skip if Slidev not installed; check fragments compile)
- [x] New v5 figures: `fig3_signal_indicators`, `fig6_n_elbow`, `fig8_surpass_teacher`
- [x] Token-level evidence: range_summary.json numbers slotted into v5_12 (rank-CDF still pending)
- [x] Sub-agent: paper draft (`paper/main_v5.tex`) — DONE, 10 main + 2 appendix pages, 556 KB.
- [x] Sub-agent: research-review-loop reviewer — DONE (codex/gemini fallback hit quota, used own sub-agent).
- [x] Iterate: rubric → fix top-3 → re-score — V5-1 (area chair), V5-2 (hostile entropy-KD), V5-3 (data-integrity sweep) all logged in `paper/REVISION_LOG.md`.
- [x] Round V5-4: theory-rigor + flow persona; 3 fixes applied (notation, mode-seeking caveat upfront, falsifier paragraph). 4/8 dims promoted to 9.
- [x] Round V5-5: cut Fig 3 (never `\ref`'d); added §3.2 footnote forward-pointer to §6 scope-of-comparison. PDF still 12pp (10 main + 1 ref + 1 appendix); 584 KB.
- [x] Round V5-6: §4.2 conjecture for FullFT-math + Llama-BFCL added; §7 falsifier folded into main paragraph. Still 10 main + 1 ref + 1 appendix.
- [ ] Pre-submission only (requires editorial judgement, not overnight): cut ~10 lines from §1-§6 to fit 9pp NeurIPS final. Candidates: §3.1 example-token list, §2 n1bs16 hyperparam → appendix, §4.4 elbow rule-of-thumb.

## Plateau note (2026-04-30, after V5-6)

Six review rounds completed (V5-1 area chair → V5-2 hostile entropy-KD → V5-3 data integrity → V5-4 theory-rigor/flow → V5-5 deferred → V5-6 deferred). Each round caught smaller issues. Remaining open items require remote experiments (entropy threshold sweep, suffix ablation), human editorial judgement (which §1-§6 clause to cut for 9pp), or new measurements (timing log). Autonomous fix-rescore loop stopped.

## V5-8 (2026-04-30, post-V5-7 grep-the-others sweep) — STATUS.md sync

Routine post-V5-7 grep across the repo for `65.55`, `57.18`, `65.40 ± 0.45`, `~93%` surfaced stale numbers in `STATUS.md` (handoff doc). Headline math table had Full-seq `65.55% / 1st-boxed` (legacy n16bs16); top-K paragraph had `random = 58.20%` (canonical 63.05%); cross-family Qwen cell still framed as "1/3 seeds; 2/3 collapsed"; Gemma math row had funcall numbers `30.9 / 3.9` instead of math `27.20 / 11.70`. Fixed all four cells, added inline multi-seed stability summary so STATUS.md headline matches paper §4.3. Canonical-number set (62.35 full-seq, 65.85 pos-100, 56.78 ± 6.0 vs 62.42 ± 3.0, 88.3% prefix-swap, 27.20/11.70 Gemma math) is now consistent across `paper/main_v5.tex`, `deck/`, `STATUS.md`, this plan, and CLAUDE.md. Logged as `V5-8` in `paper/REVISION_LOG.md`.

## V5-7 (2026-04-30, after second 30-min cron) — deck/paper sync

V5-3 fixed the paper but the deck still carried `57.18 ± 5.40`, `65.40 ± 0.45`, and `~93% prefix recovery`. Source: a 1.2-pp typo on seed-42 full-seq (`63.55` instead of canonical `62.35`) propagated through every deck stability table. Fixed in 7 deck files: `08_stability.md`, `08b_degradations.md`, `v5_08_generality.md`, `v5_01_setup.md`, `01_problem.md`, `13_summary.md`, `v5_11_cascade.md`. Deck and paper now agree on multi-seed numbers, std ratio (2.0×), and prefix-swap recovery (88%).

## Pending experiments (placeholders in deck/paper)

- Entropy-threshold sweep: top-20% (H>0.664), top-56% (H>0.01), top-80% (H>0.0004) — UCLACG GPU 0
- §10 token-level rank-of-correct-branch — script needs to run on existing logprob jsonl

## Hard constraints

- ≤10 pages (final NeurIPS 9pp)
- GPU 1 only (don't touch GPU 0 locally)
- Backwards-compatible code changes
- Use n1bs16 numbers ONLY in main body (n16bs16 only in archive)
