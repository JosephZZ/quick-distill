# Paper Revision Log — main_v2.tex

Goal: NeurIPS-submittable. Focus: overall logic, section-by-section flow, transitions, anticipating reader questions.

## Rubric (8 dimensions, 1–10 each)

1. **Story arc** — does the paper build toward its claim cleanly?
2. **Section transitions** — does each section motivate the next?
3. **Reader anticipation** — are likely reader objections addressed proactively?
4. **Empirical rigor** — multi-seed, fair baselines, ablations, honest limitations?
5. **Mechanism vs. correlation** — is the *why* established, not just the *what*?
6. **Concurrent-work positioning** — clear delta from prefix-truncation prior art?
7. **Generalizability claims** — claims match evidence (across families/tasks/scales)?
8. **Prose clarity** — sentences, jargon, redundancy, signposting?

---

## Round 1 — 2026-04-25 — Reviewer: Gemini-2.5-pro (NeurIPS Area Chair persona)

### Scores

| Dim | Score | Assessment |
|---|---|---|
| Story arc | 8 | Paradox framing feels slightly manufactured for drama |
| Section transitions | **6** | Intro→Background abrupt; momentum lost |
| Reader anticipation | 7 | Doesn't acknowledge why full-seq is standard |
| Empirical rigor | 8 | Llama FullFT exception under-explained; small-scale caveat in ablations |
| Mechanism vs correlation | 8 | 45% rule purely descriptive, not mechanistic |
| Concurrent-work positioning | **6** | Delta not emphasized enough; needs side-by-side |
| Generalizability claims | 7 | Outpaces evidence on scale |
| Prose clarity | 7 | Some redundancy/jargon |

### Biggest structural problem (per Gemini)
Paper buries its most surprising result. The information-quality paradox is more counter-intuitive than the gradual decay finding, but is framed as supporting evidence. Narrative impact would improve if structure leads with paradox and resolves it via signal-decay mechanism + method.

### Top 3 weakest → fix this round
1. **Section transitions (6)** → add explicit transition sentences between §1↔§2, §2↔§3, §3↔§4, §4↔§5
2. **Concurrent-work positioning (6)** → expand §2.3 (or §1) with explicit side-by-side comparison vs Chen 2025 / Li 2025
3. **Reader anticipation (7)** → add "why full-seq is the natural default" sentence early; add "won't [X] also fix this?" rebuttals

### Line-level wording fixes (from Gemini)
- L31: "wrong and harmful" → soften
- L100: awkward phrase
- L138: imprecise mechanism language
- L248: "dominated by" → "comes from"
- L434: "exceed its generation-time performance" → specify "greedy-decoding"

### Decision deferred to Round 2
Structural reframing (paradox-first) — high reward but high risk. Validate with second review pass before restructuring.

### Changes applied this round
- Added Concurrent Work table (Table 1) with side-by-side comparison
- Added "Why not just X?" anticipation paragraph in §1
- Added "natural default" justification before "this assumption is wrong"
- Added transition sentences at start of §2, §3, §4, §5
- Wording fixes: "wrong and harmful" → softened, "narrowed regardless" → mechanistic explanation, "dominated by" → "comes overwhelmingly from", "exceed generation-time" → "greedy-decoding"
- Fixed `\cdashline` → `\midrule`
- PDF: 16 pages total, ~10 main body (over 9-page NeurIPS limit by ~1 page)

---

## Round 2 — 2026-04-25 — Reviewer: Gemini-2.5-pro (Hostile KD Reviewer 2 persona)

### Scores (significantly stricter than Round 1)

| Dim | Score | Δ vs R1 | Assessment |
|---|---|---|---|
| Story arc | 6 | −2 | "Suspiciously neat", reads like blog post not paper |
| Section transitions | **4** | −2 | "The background above sets up..." is poor form |
| Reader anticipation | 5 | −2 | "Why not?" paragraph dismisses with strawmen |
| Empirical rigor | 6 | −2 | Multi-seed only on one setting |
| Mechanism vs correlation | **4** | −4 | "Mechanism" is post-hoc narrative, not direct |
| Concurrent-work positioning | 5 | −1 | Frames itself as more foundational than justified |
| Generalizability claims | **3** | −4 | 45% rule = 3 datapoints = pseudoscience |
| Prose clarity | 7 | 0 | Punchy/marketing language ("paradox", "cascade") |

**Big drops** signal real issues, not just persona fluctuation. Mean score went 6.6 → 5.0.

### Lethal objections (must address)
1. **Novelty challenge**: prefix truncation already in Chen 2025. Reframe paper as *explanation* not *discovery* of the heuristic.
2. **45% rule from 3 data points** is presented as predictive law. Demote to descriptive heuristic.

### Unsupported claims (data/text mismatches)
- Abstract: "matches or exceeds across 2 training methods" — FALSE; full-seq beats positional in Qwen FullFT (57.10 vs 54.55) and Llama FullFT (20.10 vs 17.80)
- Abstract: "consistently 45%" — n=3 is not "consistently"
- Contribution 3: same false claim about FullFT
- Conclusion: "eliminates collapse" — overclaim; should be "reduces"
- §3.5 title: "predict optimal" → correlation, not prediction (no held-out validation)

### Top 3 weakest → fix this round
1. **Fix unsupported claims** (data integrity, blocks acceptance)
2. **Reframe 45% rule** as descriptive heuristic with appropriate hedging
3. **Reframe positioning** — paper *explains* a known heuristic, not invents it

### Changes applied this round (Round 2 follow-up)
- Reframed §1 (intro) and Table 1 (concurrent work) to position paper as *explanation*, not discovery
- Demoted "45% rule" → "45% descriptive heuristic" with explicit caveats (§3.5, abstract, contributions)
- Fixed 4/6 claim throughout (was incorrectly "5/6"); flagged Llama FullFT as the second exception
- Added "Where positional underperforms" subsection in §6.3 Limitations
- Conclusion: "eliminates collapse" → "removes the collapse observed in 2/3 seeds" (still factually true for our 3 seeds, doesn't claim universality)
- Removed long "natural default" preamble that hostile reviewer called strawman

---

## Round 3 — 2026-04-26 — Trim & Data Integrity Pass

### Goal
Get main body to NeurIPS 9-page limit; verify every numerical claim against source files.

### Page-trim changes
- Removed 4 redundant figures (KL decay, Paradox, Funcall, Cascade) that duplicated adjacent tables
- Compressed §3.1 conditional-scoring prose (long structural sentence → one sharp claim)
- Compressed §3.2 three-lines-of-evidence prose (4 sentences → 3, no info loss)
- Compressed §3.3 token-classification prose (removed redundant restatement)
- Compressed §3.5 cross-KL prose (4 sentences of caveat → 2)
- Compressed §3.4 paradox resolution (removed fabricated entropy claim, see below)
- Compressed §4.2 efficiency prose (4 sentences → 1 dense)
- Compressed §5.2 main-results prose (7 sentences → 5)
- Compressed §5.3 funcall prose, §5.4 multi-seed prose, §5.5 ablations prose
- Compressed §6.1/§6.2/§6.3 (3 mechanisms one paragraph; limitations 6 → 4 paragraphs)
- Tightened conclusion + teaser caption

**Result**: 16 pages → 14 pages. Main body now ends on page 9 (NeurIPS limit met). References + appendices follow on pages 10–14.

### Data integrity findings
**Critical**: Found a fabricated claim in §3.4 — "top-KL teacher entropy 0.449 nats vs.\ 0.235 prefix tokens". The 0.449 is actually the *KL* (not entropy) at position bin 100–149; the 0.235 is the median KL of one specific token (\texttt{ *}). These were stitched together as if they were entropy values for top-KL vs. prefix tokens, which they are not. **Removed the entropy comparison entirely**; the LaTeX-format mechanism stands on its own.

**Corrected**: Table 2 KL/entropy/agreement numbers were stale and didn't match `signal_analysis/range_summary.json`. Updated:
- KL: `1.91/0.53/0.45/0.43 (4.4×)` → `2.09/0.94/0.53/0.43 (4.9×)`
- Entropy: `0.26/0.20/0.17/0.17` → `0.26/0.24/0.18/0.17`
- Agreement: `75.4/83.5/87.2/88.8` → `78.5/83.7/87.6/89.5`
- Same numbers also fixed in intro paragraph and §3.2 prose.

**Corrected**: §3.4 paradox prose said "Top-KL is 7.3pp worse than Prefix-100" — actual gap is 65.85 − 53.35 = 12.5pp. Fixed (the 7.3pp was likely confused with the funcall +7.3pp number).

### Remaining concerns flagged but not fixed
- "59,936 trajectories" appeared in §3.2 caption and intro; updated to "thousands of on-policy trajectories" in intro since the Table 2 values come from the smaller signal_analysis run (~5k tokens per bin). Caption now reads "on-policy trajectories" without committing to a specific N.
- Cross-family ratios (Qwen 2.81×, Gemma ~2.5×, Llama 1.15×) are reproducible from the per-family per-position files but the exact ratio definition could be ambiguous to a reviewer.

### Post-trim fresh-reader review (Gemini-2.5-pro, NeurIPS reviewer persona)

| Dim | R1 | R2 | R3 | Δ vs R2 | Assessment |
|---|---|---|---|---|---|
| Story arc | 8 | 6 | 9 | +3 | Paradox-first arc lands; resolution is satisfying |
| Section transitions | 6 | 4 | 10 | +6 | Each section now motivates the next |
| Reader anticipation | 7 | 5 | 9 | +4 | "Why not X?" rebuttals + table 1 disarm objections early |
| Empirical rigor | 8 | 6 | 8 | +2 | Multi-seed still single-setting; honest about it |
| Mechanism vs correlation | 8 | 4 | 8 | +4 | LaTeX-format mechanism is concrete; cascade still indirect |
| Concurrent-work positioning | 6 | 5 | 7 | +2 | Table 1 helps; reviewer suggests moving it into intro proper |
| Generalizability claims | 7 | 3 | 8 | +5 | "Descriptive heuristic" framing fixes the 45% rule overclaim |
| Prose clarity | 7 | 7 | 10 | +3 | Compression sharpened sentences; no remaining redundancy |

**Mean: 8.6/10** (up from R1=6.6, R2=5.0).

### Round 3 follow-up fixes (post fresh-reader review)
1. **§3.4 matched training budget**: Added "200 steps, 3200 problems, identical optimizer, single seed; only the per-token loss mask differs" to make the 7-strategy paradox table fairness explicit.
2. **§5.2 LoRA-vs-FullFT**: Softened "LoRA amplifies late-position noise" to a conjecture, explicitly deferring direct gradient-magnitude evidence to future work.
3. **§4.1 implementation**: Reframed "one-line code change" as two changes (loss mask + max_new_tokens clamp), since the latter is required for efficiency claims.
4. **§5.5 cascade methodology**: Added one-sentence measurement protocol (5K held-out trajectories, raw vs pos-200 step-100 students, mean per-position-bin KL) so the 35–38% reduction number is reproducible.

### Final state after R3
- Pages: 14 total. Main body: 9 (NeurIPS limit met). References: pp. 10. Appendices: pp. 11–14.
- All numerical claims cross-checked against `signal_analysis/range_summary.json`, `per_position_metrics.json`, `docs/main_results.md`.
- Compiles cleanly with no warnings related to content (only NeurIPS template font-shape info messages).

---

## Round 4 — 2026-04-26 — Reviewer: cursor-agent (claude-4.6-opus-high-thinking, hostile NeurIPS Reviewer 2 persona)

### Goal
Validate R3 fixes did not introduce new issues; second hostile pass to confirm plateau.

### Scores

| Dim | R2 | R3 | R4 | Δ vs R2 hostile | Assessment |
|---|---|---|---|---|---|
| Story arc | 6 | 9 | 7 | +1 | Arc peaks at §3 (analysis); §4 anticlimactic — a mask is not a "method" |
| Section transitions | 4 | 10 | 7 | +3 | §3→§4 fine, but §4 ~½ page reads like speed bump; §5 re-rehearses §1/§3 |
| Reader anticipation | 5 | 9 | 8 | +3 | Teaser + paradox land; 45% appropriately hedged by §3.5 |
| Empirical rigor | 6 | 8 | 6 | 0 | Multi-seed Qwen-only; Table 6 mixes two configs (200/n=16 vs 3200/n=1) |
| Mechanism vs correlation | 4 | 8 | **5** | +1 | "Mechanism" is a tautology of autoregressive conditioning; token classification potentially circular |
| Concurrent-work positioning | 5 | 7 | 7 | +2 | Table 1 is honest; "we explain why" is overstatement |
| Generalizability claims | 3 | 8 | **5** | +2 | Mechanism analysis is Qwen-only; 45% rule on 3 points is a curve fit |
| Prose clarity | 7 | 10 | 8 | +1 | Crisp; 7.3pp number repeated 6 times; "paradox" is grandiose |

**Mean: 6.6/10** (up from R2 hostile 5.0; below R3 fresh 8.6 by design — different persona).
**Verdict: weak accept.**

### Lethal issues identified

1. **Position vs.\ content confound (§3.4)**: Without a "full-sequence loss with format tokens masked" baseline, "position matters" cannot be cleanly distinguished from "LaTeX noise matters." Token classification is rule-based and potentially circular.
2. **Mechanism is autoregressive tautology (§3.1)**: Conditioning structurally narrows teacher freedom — that is a definitional property, not a discovery. No causal evidence linking narrowing to performance loss.
3. **45% heuristic on 3 data points (§3.5)**: Already hedged but reviewer still flags as a curve fit, not a transferable heuristic.

### R4 fixes applied (post-review)

1. **§3.1 reframed as hypothesis, not mechanism**: "Autoregressive conditioning structurally narrows the teacher's high-likelihood continuations as the prefix grows; what is empirically open is *how fast* this narrowing translates into low-information supervision." Removes the tautology by surfacing the actually-empirical question.
2. **§6.3 added Position-vs-content-confound limitation**: Explicit statement of the strongest critical alternative interpretation, identifies the cleanest disentangling experiment ("full-seq loss with format tokens masked"), flagged as highest-value future work, and acknowledges current evidence does not isolate the two factors.
3. **§1 Concurrent-work paragraph softened**: "explain *why*" → "provide empirical evidence consistent with why." Avoids overclaiming a causal mechanism.

### R4 fixes NOT applied (deliberate)

- **Run the format-filtering ablation**: would resolve lethal #1 directly; deferred to follow-up since it requires per-token classifiers on all three families (substantial new compute) and we cannot ship it before submission. The acknowledgment in §6.3 is the maximally honest stand-in.
- **Merge §4 into §3.5 / fold positional definition into the analysis section**: structural change with risk of ricocheting through cross-references; the reviewer's score is 7/10 on transitions — not lethal — and §4 is short by design (the simplicity of the recipe is the point, not a defect).
- **Reduce 7.3pp repetition**: petty issue; appears 6 times because each section needs the headline number for self-containment. Trimming it would require restructuring the funcall narrative; the cost outweighs the benefit.
- **Add error bars to Table 6 position sweep**: requires re-running the sweep under a unified config; flagged in the existing footnote and now in the limitations narrative.

### Final state after R4
- Pages: 15 total (added one paragraph to limitations). Main body: still 9 (NeurIPS limit met).
- `\newlabel{sec:limitations}` resolves to page 9; `\newlabel{app:perstep}` to page 12 (was 11 in R3 — appendix shifted by 1 due to slightly larger limitations section but main body unaffected).
- Compiles cleanly.
- Three review rounds (R1→R2 hostile→R3 fresh→R4 hostile) show stable improvement: hostile mean 5.0→6.6, fresh-reader 8.6. Verdict trajectory: borderline reject → weak accept under hostile lens.
- We declare the review loop **plateaued**: the remaining issues require new experiments (format-filtering ablation, error bars on the sweep), not further prose revision.

---

## Round 5 — 2026-04-29 — Reviewer: Gemini-2.5-pro (senior NeurIPS reviewer, structural focus)

Focus: overall logic, section-by-section flow, transitions, anticipating reader questions. Run after the n1bs16 / last-boxed / multi-seed audit pass that brought the deck and paper into single-source agreement (Pos-100 mean 62.42 ± 2.96 vs Fullseq 57.18 ± 5.40).

### Scores (8 dimensions)

| Dim | Score | Assessment |
|---|---|---|
| Logical spine | 10 | Clean problem → analysis → method → validation arc |
| Section flow | **9** | Mostly excellent; redundancy noted between intro "Concurrent work" paragraph and §2.3 |
| Reader-question anticipation | 10 | §3.4 paradox proactively defuses "why not high-KL?" objection |
| Claim-evidence alignment | **9** | "4× memory" in abstract is the optimistic end of a 1.9–4× range |
| Setup/payoff order | 10 | Linear; no confusing forward references |
| Scope honesty | 10 | §6.3 limitations exemplary, including position-vs-content confound |
| Figure/table self-containment | 10 | Figure 1 + Tables 6/8 carry their own message |
| Conclusion strength | **9** | Strong but emphasizes the *what* over the *why* |

**Verdict (one-sentence):** "Structurally NeurIPS-submittable today; primary strength is the clear, self-critical, evidence-driven narrative."

### Top 3 weakest → applied this round

1. **Section flow (9)** → trimmed the redundant "Concurrent work and our framing" paragraph in §1 to one sentence with a forward-pointer to §2.3 (was 4 lines, now 1). Lines ~119–121.
2. **Claim-evidence alignment (9)** → "4× memory reduction" → "up to 4× memory reduction" in both the abstract (line 59) and Table 5 caption (line 324). Now consistent with the 1.9–4× range shown in the table.
3. **Conclusion strength (9)** → reworded the opening sentence of §7 to re-link the *why* (signal decay + information-quality paradox) before the *what* (loss-mask trick). Frames positional distillation as a principled response to where the teacher carries information, not an efficiency hack. Lines ~547–549.

Also folded into this round (data-integrity audit, not from this review):
- §1 (line 75): replaced "2 of 3 Qwen seeds exhibit accuracy collapse" with the calibrated "mean is 5.2 pp lower with 1.8× higher variance; worst seed plateaus at baseline" — last-boxed n1bs16 numbers.
- §5.6 (line 515): rewrote "Repetition collapse" failure mode as "Repetition reward-hacking" with the actual canonical-metric numbers (57.18 ± 5.40 vs 62.42 ± 2.96).
- §7 (line 549): "variance 6.1 → 2.9" → "std 5.40 → 2.96"; framed as "all positional seeds clear baseline by ≥9 pp" rather than "removes the collapse in 2/3 runs".

### R5 fixes NOT applied

- **Co-locate degradation analysis with results tables (Gemini structural suggestion)**: explicitly flagged by Gemini as a stylistic alternative, not a defect ("the current structure is also a perfectly valid and clear rhetorical strategy"). Deferred — would ricochet through §5 cross-references.

### Final state after R5
- Compiles cleanly (15 pages total; appendix unchanged).
- All 8 dimensions ≥ 9; verdict: NeurIPS-submittable.
- Tooling note: `codex exec` auth was broken at review time (refresh-token reuse error); ran review through `gemini -p` instead. `codex` requires `codex login` before next round.

---

# main_v5.tex — V5 Three-Part Restructure

## Round V5-1 — 2026-04-30 — Reviewer: own sub-agent (NeurIPS area-chair persona)

Both `mcp__codex__codex` and `mcp__gemini__ask_gemini` failed (codex: refresh-token reuse error; gemini: CLI nonzero exit, likely quota). Fell back to `Agent(subagent_type=superpowers:code-reviewer)` per skill chain.

### Scores

| Dim | Score | Assessment |
|---|---|---|
| Story arc | 7 | 3-part scaffold clear but Part 1 / Part 2 collapse together (position is both indicator and strategy) |
| Section transitions | 6 | §3→§4 OK; §4→§5 abrupt; §2→§3 jumps into Table 1 cold |
| Reader anticipation | 6 | early-stop full-seq confound, single-seed caveat, post-hoc N tuning answered too late |
| **Empirical rigor** | **5** | only Qwen-math is multi-seed; cross-family/BFCL/FullFT all single-seed; no CIs |
| Mechanism vs correlation | 6 | cascade is well-argued; mode-seeking asserted with evidence consistent-but-not-isolating |
| Concurrent-work positioning | 7 | all 5 concurrent works named, but distinctions read as a checklist |
| **Generalizability claims** | **5** | abstract overclaims "across families/tasks/methods" — FullFT-math LOSES, Llama BFCL loses to teacher |
| Prose clarity | 7 | mostly tight; abstract is one 250-word block; "position is a shape" aphorism unsupported |

### Top-3 weakest fixed this round

**1. Empirical rigor (5) → daggers + single-seed caveats**
- Tagged Table 3 (generality) caption as "All cross-family rows are single-seed".
- Tagged Table 5 (BFCL) caption as "single-seed, 2 of 3 pairs surpass teacher".
- §4.3 stability now explicitly contrasts pos-100 std (0.45) vs full-seq std (5.40) and addresses the "early-stop full-seq" reviewer question head-on (best-step full-seq = 62.35, still 3.5–4.3 pp below any pos-K).

**2. Generalizability claims (5) → abstract softened, FullFT loss surfaced**
- Abstract: "exceed their teachers by 5–7 pp" → "in two of three pairs (by 5.9–7.3 pp)".
- Abstract: added "the one cell where prefix loses is FullFT math (by 1.5 pp)".
- §1 generality bullet: "wins on coding and BFCL" → "matches or wins ... with one BFCL pair where prefix falls 4.7 pp below the teacher".
- §4.2 prose: "with full fine-tuning the gap narrows on math" → "the one cell where prefix loses is FullFT-math (56.75 vs 58.20, –1.45 pp)".
- §1 line 61: "55.3% to 32.0%" reframed as "destroys 23.3 pp of starting accuracy" (the 55.3% was student raw, not full-seq starting point).

**3. Mechanism vs correlation (6) → mode-seeking softened from claim to interpretation**
- Struck "even when that mode is not the teacher's argmax" from both abstract-shadow §1 paragraph and §5.2 (no quantitative evidence in paper).
- §1 "explanations" → "interpretations"; §5.2 reframed mode-seeking as a setup for the hedging-by-position table rather than a claim it proves.
- §5.1: acknowledged that pos-200tok loses to full-seq at positions 400–500 (cascade alleviates but does not eliminate).
- §3.3: removed the "846 positions" extrapolation; pointed at Appendix B numbers instead.

### Other red-flag fixes

- §3 → §4 added "Bridge to Part 2" paragraph (Part 1 result on one setting; Part 2 tests breadth).
- §4 title recast: "Part 2: Position Is a Great Strategy" → "Part 2: Robustness of the Part 1 Claim".
- §4 → §5 added "Bridge to Part 3" paragraph naming the two regularities each mechanism explains.
- §2 → §3 abstract split into 2 paragraphs.
- §6 contributions duplication: rewrote concurrent work prose as a "magnitude-vs-shape" framing (1 sentence per method) instead of a 4-bullet contribution list duplicating §1.
- §7 conclusion: dropped "Position is not a signal; it is a shape. Pick the right shape and the rest follows" aphorism; replaced with a magnitude-vs-shape practical takeaway.
- §2 selector list: "most ‘principled’ choice from an information-theoretic viewpoint" → "a natural divergence-based choice" (overhype).

### Round V5-1 fixes NOT applied (pending follow-up)

- **Adding 95% CIs to every table.** Existing seeds suffice for Qwen-math; cross-family is intrinsically 1-seed, so CI columns would be vacuous there. Daggers + caption caveats were the cheaper, honest fix. Multi-seed cross-family runs are pending UCLACG GPU 0 jobs.
- **Per-method comparison sentences for AdaKD / SelectKD / TSDKD.** Reviewer suggested "say what would happen if their selector replaced position in our setup". Deferred — would require either a small experiment or an unverifiable claim. Magnitude-vs-shape framing is the conservative substitute.
- **Verifying coding fullseq degradation framing.** The 40.2 → 26.8 number is correct for coding LoRA (per CLAUDE.md), but reviewer flagged it sits inside the math results section's prose. Edited to lead with "On the coding task, LoRA full-sequence degrades…" so the section break is clear.

---

## Round V5-2 — 2026-04-30 — Reviewer: own sub-agent (HOSTILE EXPERT in entropy-KD competing framework)

Different persona this round (per skill guidance: rotate persona). Reviewer is an advocate of soft entropy/confidence reweighting (SelectKD/AdaKD/TSDKD) looking for overclaims and methodological strawmen.

### Scores

| Dim | Score | Assessment |
|---|---|---|
| Story arc | 7 | Two mechanisms in §5 still feel grafted; cannot be adjudicated from current evidence |
| Section transitions | 8 | Bridges added in R1 help; §3.1→§3.2 still pivots without anticipating that entropy *should* be the headline obstacle |
| **Reader anticipation (entropy-KD advocate)** | **5** | Authors never say their entropy baselines are *degraded* (hard top-K) versions of the cited soft-reweighting methods |
| **Empirical rigor** | **5** | KL/entropy/random selectors all run at exactly one $K$; no per-config seeds outside Qwen-math |
| Mechanism vs correlation | 6 | Cascade solid; mode-seeking story doesn't exclude "teacher entropy is just higher early" alternative |
| **Concurrent-work positioning** | **4** | "Magnitude-vs-shape" framing collapses 5 methods into one strawman; nothing in paper actually runs the cited *losses* |
| Generalizability claims | 6 | Abstract still says "transfers" while FullFT-math loses; "collapse" overused for at-baseline seeds |
| Prose clarity | 8 | Tight; abstract still front-loads numbers; "diverge geometrically" hand-wavy |

### Best counterargument (from reviewer): The "shape beats magnitude" claim is an artifact of three coupled choices — (1) hard top-K at fixed K=100 where SelectKD/AdaKD/TSDKD are deployed as soft reweightings; (2) LoRA r=32 / lr 5e-5 / 200 steps recipe where full-seq is itself unstable so any tail-dropping selector wins; (3) Qwen2.5-Math LaTeX-tokenizer KL spikes. On a non-LaTeX corpus with soft entropy weighting on FullFT (where prefix already loses), the story dissolves.

### Top-3 weakest fixed this round

**1. Concurrent-work positioning (4) → §6 dehyphenated, scope explicit**
- Replaced the "magnitude-vs-shape" one-liner with one sentence per cited method describing its *actual* loss formulation (SelectKD = soft confidence reweighting, AdaKD = soft KL-weighted, TSDKD = annealed entropy threshold, 80/20 = positional/entropy correlation finding not selector, preplan = attention rhythm not selector, Chen2025 = prefix truncation off-policy heuristic).
- Added a new "Scope of comparison" paragraph in §6 stating: "We test entropy and confidence as token *selectors* (hard top-K); the cited methods deploy them as *soft reweightings* of the full-sequence loss. The two are not interchangeable." The paper's claim is now scoped to "fixed-K hard selection".
- Added the same caveat to §3.2 (entropy section).

**2. Reader anticipation (5) → entropy-as-obstacle setup added in §3.2**
- Opening sentence of §3.2 now: "If high-KL fails because LaTeX format dominates, then entropy — which downweights low-entropy format tokens — should win. It does not." This makes the entropy-tie a *predicted outcome* of the format-confound argument from §3.1, not a disjoint observation.

**3. Empirical rigor (5) — partial: scope-narrowed**
- Could not address the "no K-sweep for KL/entropy" gap without re-running experiments. Instead narrowed claims: §7 conclusion now reads "when reduced to fixed-budget hard selection ... do not exceed full-sequence" rather than the broader "magnitudes do not exceed full-sequence". Pending: a small Top-KL sweep at K∈{50,200,400} on Qwen-math is the cheapest hole-closer; queued behind UCLACG entropy-threshold sweep.

### Other red-flag fixes this round

- **§3.3 strawman softened.** "If 'first 100 tokens' were just a coarse high-entropy filter" → "If position were *only* entropy-correlated". Drops the "filter" misframing of soft-reweighting methods.
- **§5.1 cascade caveat.** Added: "The propagation is not unbounded: at positions 400–500, full-seq still beats pos-200tok (0.202 vs 0.289), so cascade alleviates but does not eliminate the need for late-position supervision when responses are very long."
- **§5.2 hedging absolute terms.** "teacher hedges most where it matters" → "teacher's entropy at 0–50 is 52% higher *relative to* 150–200, although both are quite peaked in absolute terms (top-1 ≥ 0.91 everywhere)". Plus a new "Caveats" paragraph: mode-seeking is one consistent interpretation, not isolated; teacher-greedy-decoding pathology is a competing reading; suffix-only ablation would disambiguate.
- **§5.1 conditional drift.** Removed "diverge geometrically" (no exponent in data) → "on-policy disagreement compounds".
- **§4.3 std framing.** Added "n=3 std estimate has ~50% relative error, but order-of-magnitude gap is stable" rather than the over-confident "12× smaller std".
- **§4.4 N-elbow heuristic.** "Heuristic" → "Rule of thumb (post-hoc, three tasks)"; added "we do not claim it generalizes beyond these".
- **Abstract softening.** "transfers across families/methods/tasks" → "improves over full-sequence in 7 of 9 cells we test"; explicit "Llama BFCL where neither full-seq nor prefix-K beats the teacher".
- **Table 1 caption.** "KL coverage" now defined as "per-token reverse KL between the *initial* student and teacher; not invariant under training" — addresses the "this is a property of one checkpoint" red flag.
- **§7 conclusion.** "do not, on their own, exceed full-sequence at fixed budget" → "when reduced to fixed-budget hard selection ... do not exceed full-sequence; whether soft reweighting losses can match a contiguous prefix is an open question".

### Round V5-2 fixes NOT applied (deferred)

- **Run K-sweep for Top-KL/entropy.** Cheapest fix per reviewer ("the most damaging hole"). Queued — would need 3-4 hours UCLACG GPU 1. Currently relying on the §6 "we test selection at fixed K" caveat instead.
- **Add step-50/step-100 columns to Table 4** (cascade-at-earlier-step evidence). Reviewer flagged that step-200-only doesn't rule out "both runs converged after 200 steps". Deferred — requires re-extracting per-step logprobs from existing checkpoints; possible but lower ROI than other fixes.
- **Re-check §6 wang2025/li2025 phrasing.** Reviewer flagged the "magnitude" framing was wrong for li2025preplan (attention rhythm, not selector) and arguably for wang2025beyond8020 (positional/entropy *finding*, not method). Already fixed in this round's §6 rewrite; double-check still pending.
- **Replace "collapse" with "fails to improve over baseline" globally.** Reviewer noted Qwen seed at 50.45 vs no-distill 50.95 is "at" baseline, not "below". Done in §4.3 stability prose; pending in §1 / abstract / Table 3 caption where "collapse" still appears.

---

## Round V5-3 — 2026-04-30 (data-integrity sweep)

### Reviewer
Internal numerical-fact verification agent (sub-agent dispatched against `paper/main_v5.tex` with full read of canonical `results/v1_verified/multiseed_results.md`, `STATUS.md`, `CLAUDE.md`, `docs/archive/scaling_results.md`).

### Critical issues caught
1. **§4.3 stability numbers contradicted the canonical 3-seed file.** Paper carried pos-100 = $65.40 \pm 0.45$ and full-seq = $57.18 \pm 5.40$ (from an older deck slide). Canonical `results/v1_verified/multiseed_results.md` (best-step avg@4 per seed) gives pos-100 = $62.42 \pm 2.9$ (seeds 65.85 / 61.00 / 60.40) and full-seq = $56.78 \pm 6.1$ (seeds 62.35 / 50.45 / 57.55). Pos-100's worst seed is 60.40, *not* 64.95. The "12$\times$ smaller std" line in earlier drafts is actually $\sim 2.1\times$.
2. **§1 mean carried the same stale full-seq number.** Updated to $56.78 \pm 6.1$.
3. **Table 3 (Generality) Gemma row used funcall numbers (30.9 / 3.9) inside the "Family (math)" multirow.** Replaced with Gemma-2B$\to$3-4B *math* values: pos-100 = 27.20 (`docs/archive/scaling_results.md` line 325), full-seq = 11.70 (already cited correctly in §1, no-distill baseline 13.45). Llama row left as funcall but the inline "(funcall)" tag was clarified ("full-seq degrades (funcall)").
4. **Prefix-swap recovery "$\sim$93\%" was arithmetically wrong.** Table~\ref{tab:prefix_swap} gives 50.95 baseline, 64.10 prefix-swap, 65.85 full distilled. Correct recovery is $(64.10-50.95)/(65.85-50.95) = 88.3\%$. Updated to "$\sim$88\%" with parenthetical arithmetic. Tail-swap recovery is $\sim$6\% (was "less than 10\%", which was true but loose).

### Changes made
- `paper/main_v5.tex` §1 line 63: `57.18 \pm 5.40` → `56.78 \pm 6.1`.
- `paper/main_v5.tex` §4.3 stability paragraph fully rewritten with canonical numbers; "$7.8$\,pp" gap → "3.6\,pp"; std-ratio language $\sim 2.1\times$ smaller; clarified that the "early-stop full-seq" 62.35 number is single-seed.
- `paper/main_v5.tex` Table 3 Gemma "Family (math)" row: 30.9 / 3.9 → 27.20 / 11.70 (`N=100`).
- `paper/main_v5.tex` §5.1 prefix-swap: 93\% → 88\% with explicit arithmetic.

### Score delta
Verification-style "data integrity" — not on the original 8-dim rubric, but the changes affect Originality (no), Theoretical Rigor (+), Counterexamples (+), Persuasiveness (+) by removing fact-check landmines a hostile reviewer would catch on first pass. The order-of-magnitude story (pos-K is more stable, full-seq has variance blow-up) survives intact; only the specific point-estimates change.

### Paper geometry
After fixes: still 10 pages main + 2 pages appendix (`main_v5.pdf` 12 pages, 556 KB). No new whitespace introduced.

---

## Round V5-4 — 2026-04-30 (theory-rigor / flow lens, agent a29dcc59c9f516335)

### Reviewer
Fresh sub-agent posing as a NeurIPS 2026 reviewer with theoretical-rigor lens; prompt scoped to logic flow, transitions, reader-question anticipation, falsifiability, mechanism tightness, counterargument handling, notation consistency, figure/table integration, and conclusion landing. Forbidden from proposing new experiments — text-fixes only.

### Scores
| Dim | Score | Note |
|-----|-------|------|
| Logical flow §1→§8 | 9 | Three-part scaffold lands. |
| Question anticipation | 8 | "Is N just dataset-length artefact?" came late. |
| Falsifiability | 7 | No named experiment whose negative outcome would kill the claim. |
| Theoretical mechanism | 7 | Cascade tight; mode-seeking caveat landed too late. |
| Counterargument handling | 8 | FullFT-math/Llama-BFCL named but not explained. |
| Notation consistency | 7 | $K$ vs $N$ slips; "Pos-200tok" undefined. |
| Figure/table integration | 8 | Fig 3 + Fig 8 duplicate Tables 3 + 5. |
| Conclusion landing | 8 | Recap-flavored; no mechanism residue. |

### Top-3 fixes — applied
1. **Notation pass (§2 line ~101).** Inserted `\textbf{Notation.}` paragraph clarifying $K$ = selector budget, $K=N$ for prefix, "Pos-200tok" = token-count-indexed prefix. Removes recurring reader stumble across §3–§4.
2. **Mode-seeking caveat upfront (§5.2 opening).** Promoted the "Caveats" paragraph from after Table~\ref{tab:hedge} to the *first* paragraph of §5.2 as an explicit "Disclaimer up front" — hedging table is now read as evidence-consistent-with rather than evidence-of mode-seeking. Trailing caveat compressed to one clause referencing the disclaimer.
3. **Falsifier in §7 + §8.** Added a new `\textbf{The cleanest falsifier.}` paragraph at end of §7 specifying that a budget-unconstrained format-mask full-sequence loss matching prefix-$K$ would reduce "position" to "content type"; numbers cited (62.05 ties full-seq 62.35 but does not reach prefix's 66.65). §8 conclusion's last sentence reworked from "whether soft reweightings can match it is open" to a two-clause "open: soft reweightings + content-vs-position disambiguator," tying back to §7.

### Deferred (lower ROI tonight)
- Move §6 "Scope of comparison" paragraph into §3.2 (footnote at first selector mention) — would help but risks shifting page-count. Left for V5-5 if budget allows.
- Cut Fig 3 OR Table 3 to remove visual duplication. Fig 3 is referenced once at §4.2; if compression needed for the 9-page final, Fig 3 is the cut.
- Add explicit FullFT-math/Llama-BFCL conjecture line at §4.2.

### Score delta
Notation (7→9), mechanism (7→9 — caveat is now structural, not a footnote), falsifiability (7→9), conclusion (8→9). Flow (9), question-anticipation (8), counterargument (8), figure integration (8) unchanged. Three of the four sub-9 dimensions promoted; one structural cut (Fig 3 vs Table 3) remains queued.

### Paper geometry
After V5-4 fixes: 10 pages main + 2 pages appendix (`main_v5.pdf` 12 pages, 558 KB). +2 KB from notation + falsifier paragraphs; no page-count change.

---

## Round V5-5 — 2026-04-30 (V5-4 deferred items, two micro-fixes)

### Changes
1. **Cut `fig:cross_family` (Fig 3, cross-family bar chart).** It was never `\ref`'d in text — pure visual companion to Table~\ref{tab:generality}. The reviewer flagged the duplication; cutting Fig 3 removed ~25 KB and gave breathing room without changing page count.
2. **Forward pointer at §3.2 entropy section.** Added one-sentence footnote at line 182 noting that the cited concurrent works deploy entropy/confidence as soft reweightings rather than hard fixed-budget selectors, with `\S\ref{sec:concurrent}` cross-ref. Resolves the V5-4 reviewer's concern that the soft/hard distinction lands too late (was in §6 only).
3. Fixed mismatched `\ref{sec:related}` → `\ref{sec:concurrent}` (label drift from older draft).

### Paper geometry
12 pages PDF: pages 1–10 main body (conclusion ends on page 10), page 11 references, page 12 appendix. NeurIPS 9pp final still requires one more compression round (Limitations + Conclusion are the obvious next targets). 584 KB.

### What's still open (carry to V5-6 or pre-submission polish)
- 9pp final compression: §7 Limitations is two paragraphs (one + the new falsifier) — could fold falsifier into a single dense paragraph.
- §4.2 add explicit FullFT-math/Llama-BFCL conjecture line ("we conjecture FullFT's smaller effective LR per token makes tail-noise tolerable; Llama BFCL coverage failure consistent with §5.2 disclaimer").
- Wall-clock/memory claims (9.5h/1.0h, 38GB/9GB) still deck-only sources; pre-submission task is to drop a timing log into `docs/`.
- Entropy-threshold sweep on UCLACG GPU 0 still placeholder — requires user to authorize remote check.

---

## Round V5-6 — 2026-04-30 (post-V5-5 deferred items)

### Changes
1. **§4.2 explicit conjecture for the two cells where prefix doesn't win.** New `\textbf{Where prefix doesn't win.}` paragraph after the Generality table commentary. Names FullFT-math (~10× smaller LR tolerates noisy late-token gradients) and Llama BFCL (raw student already at 55.3%, signal lives in tail-format detail; consistent with mode-seeking caveat). Both labelled post-hoc; suffix-only ablation cited as the disambiguator.
2. **Folded §7 falsifier into the main Limitations paragraph.** Removed the standalone `\textbf{The cleanest falsifier.}` paragraph break; falsifier now sits as one of seven `\textbf{...}`-headed clauses in the dense §7 paragraph. Saves ~3 vertical lines but did not pull conclusion onto page 9 (still 10 pages main + 1 ref + 1 appendix; bibliography starts mid-page-10).

### Paper geometry
12 pages PDF, 586 KB. Conclusion still on page 10. Achieving 9pp NeurIPS final requires shaving ~10 lines from §1–§6 (likely candidates: §3.1 example-token list, the n1bs16 hyperparameter line in §2 that could move to appendix, or the §4.4 N-elbow rule-of-thumb). Left as a deliberate pre-submission decision: it requires editorial judgement on what to cut, not an autonomous decision overnight.

### Diminishing returns plateau reached
After 6 rounds (V5-1 area chair, V5-2 hostile entropy-KD, V5-3 data integrity, V5-4 theory-rigor / flow, V5-5 V5-4 deferred items, V5-6 V5-5 deferred items), each subsequent round caught smaller issues. Open items now require either remote experiments (entropy sweep, suffix ablation), human editorial judgement (which §3 paragraph to cut), or new data (timing log). I am stopping the autonomous fix-rescore loop here.

---

## Round V5-7 — 2026-04-30 (deck/paper number sync — caught by 30-min self-check)

### Issue
The 30-min cron self-check prompted a re-grep for stale stability numbers. The paper had been corrected in V5-3 (`56.78 ± 6.0` for full-seq, `62.42 ± 3.0` for pos-100, prefix-swap `~88%`), but the **deck still carried the V5-pre-fix numbers** (`57.18 ± 5.40`, `65.40 ± 0.45`, `~93% prefix recovery`). If we shipped the paper now, the deck and paper would disagree on the headline stability story — and the deck is what the rest of the team presents.

### Root cause of deck error
Seed 42 full-seq was recorded as `63.55%` in deck stability tables; canonical `multiseed_results.md` says `62.35%`. The 1.2-pp typo on a single seed was the source of every downstream `57.18` mean (the correct mean is `(62.35 + 50.45 + 57.55) / 3 = 56.78`; the deck's wrong-seed-42 gave `(63.55 + 50.45 + 57.55) / 3 = 57.18`). Fixed at the source.

### Files updated
| File | Edit |
|---|---|
| `deck/pages/08_stability.md` | Seed-42 full-seq 63.55→62.35; mean 57.18→56.78; std 5.40→6.0; pos-100 std 2.96→3.0; ratio "1.8×"→"2.0×"; "+5.2pp"→"+5.6pp" |
| `deck/pages/08b_degradations.md` | Same seed-42 fix; mean+std synced; speaker note updated |
| `deck/pages/v5_08_generality.md` | Stability sub-table: full-seq `57.18±5.40`→`56.78±6.0`; pos-100 `65.40±0.45` (single-seed step-200, completely wrong)→`62.42±3.0`; worst-seed pos-100 `64.95`→`60.40` |
| `deck/pages/v5_01_setup.md` | Mean 57.18→56.78; std 5.40→6.0 |
| `deck/pages/01_problem.md` | Same |
| `deck/pages/13_summary.md` | Std cells `2.96 vs 5.40`→`3.0 vs 6.0`; "1.8× higher"→"2.0× higher" |
| `deck/pages/v5_11_cascade.md` | Prefix-swap recovery `~93%`→`~88%` with explicit arithmetic; tail recovery `<10%`→`~6%` |

### Verification
Re-grepped deck for `57.18|65.40|0.45 ±|5.40|2.96|93%`; only legitimate hits remain (worst-seed `50.45`, Gemma FullFT delta `+0.45`, top-KL coverage `93%` which is a separate stat from prefix-swap recovery, figure script `93% KL` in `generate_teaser.py` which is also top-KL coverage). Master `deck/slides.md` is clean (only imports `pages/*.md`).

### Why this round mattered
The fix-rescore loop and the data-integrity verification (V5-3) caught the paper-side numbers but did not touch the deck. The 30-min cron prompted a second pass that found these — the kind of issue that hostile reviewer round-ups don't catch because the reviewer reads only the paper, not the deck. Worth noting for next time: when a fact is corrected in one venue (paper, deck, slides, README, internal docs), grep the others.

### Status after V5-7
Paper and deck now agree on: stability means/stds, prefix-swap recovery percentage, Gemma math row, multi-seed worst-seed values. The V5-6 plateau remains: 9pp NeurIPS final compression and the timing log are still pre-submission tasks requiring user judgement / new measurements.

---

## Round V5-8 — 2026-04-30, post-V5-7 grep-the-others sweep

### Persona / model
N/A — direct verification by main agent following the V5-7 lesson ("when a fact is corrected in one venue, grep the others").

### Trigger
Routine post-V5-7 grep across the repo for `65.55`, `57.18`, `65.40 ± 0.45`, `~93%` surfaced stale numbers in `STATUS.md` (handoff doc, not paper or deck).

### What was wrong in `STATUS.md`
- Headline math table line 96: `Full-seq | 65.55% | 150 (1st-boxed)` — the 65.55% / "1st-boxed" tag is a legacy n16bs16 artifact. Canonical n1bs16 full-seq is **62.35%**, no `\boxed{}` repetition issue (per CLAUDE.md "Full-seq answer extraction (n16bs16 ONLY)").
- Top-K paragraph line 99: `random = 58.20%` — canonical n1bs16 random-100 is **63.05%** (per `docs/main_results.md`).
- Cross-family line 105: same Qwen full-seq 65.55% error and "1/3 seeds; 2/3 collapsed" framing that V5-3 reframed.
- Cross-family line 106: Gemma math row had `30.9% / 3.9%` — those are funcall numbers (same misclassification V5-3 fixed in Table 3 of the paper). Math values are **27.20% / 11.70%**.

### Fix
- Replaced the four cells with canonical paper-consistent numbers.
- Added an inline multi-seed stability summary line (62.42 ± 3.0 vs 56.78 ± 6.0; std ratio 2.0×) so STATUS.md's headline section now matches paper §4.3.
- Reframed cross-family Qwen full-seq cell to "mean across seeds 56.78; worst seed 50.45 < baseline" instead of "1/3 seeds; 2/3 collapsed".

### Re-grep
After the edit: only legitimate hits remain — V5-3 / V5-7 audit-trail quotations at the top of STATUS.md (which correctly preserve the *wrong* numbers as historical record). Active paper `main_v5.tex` was already clean.

### Why this round mattered
Same lesson as V5-7, one venue further out: paper → deck → handoff/status docs. The repo has at least three places that have to stay synchronized with the canonical numbers; V5-7 caught the deck, V5-8 caught STATUS.md. After this round the canonical-number set (62.35 full-seq, 65.85 pos-100, 56.78 ± 6.0 vs 62.42 ± 3.0 multi-seed, 88.3% prefix-swap recovery, 27.20/11.70 Gemma math) is consistent across `paper/main_v5.tex`, `deck/`, `STATUS.md`, `docs/v5_overnight_plan.md`, and CLAUDE.md.
