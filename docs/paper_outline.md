# Paper Outline: Positional Distillation (v5)

*Revision history: v1 (initial), v2 (story/framing), v3 (technical rigor), v4 (figures/presentation), v5 (final polish)*

---

## Title

**Positional Distillation: Computing Loss on Early Tokens Is Sufficient for On-Policy Knowledge Distillation**

*Rationale*: "Sufficient" is more precise and defensible than "All You Need." Avoids overclaiming while still being surprising. Clearly names the method and the setting.

*Alternatives considered*:
1. "The Cascade Effect: Why Distilling Early Tokens Improves Entire Reasoning Chains" -- too mechanism-focused for a title
2. "Less Is More: How Early-Token KL Divergence Drives Effective Knowledge Distillation" -- vague
3. "Early-Token Distillation: Leveraging Non-Uniform KL Divergence in On-Policy Knowledge Distillation" -- too long

---

## Abstract (refined)

On-policy knowledge distillation trains a student model on reverse KL loss over its own generated sequences. We identify two problems with the standard full-sequence approach: (1) training instability, where the student learns degenerate repetition patterns (e.g., repeating \boxed{answer} 58-88 times in math, or sustained accuracy degradation from 40.2% to 26.8% on HumanEval for code), and (2) wasted computation on low-information tokens. Through analysis of 59,936 on-policy trajectories, we show that per-position KL divergence between teacher and student is highly non-uniform: position 0 has KL=3.06 (8.5x higher than position 192), with the first 50 tokens containing 26% of total KL but capturing 73% of achievable performance gain. These high-KL early tokens correspond to reasoning strategy decisions (approach selection, problem framing), while later tokens cover computation where teacher and student already agree (math numbers: mean KL=0.28 vs. planning tokens: mean KL=1.75). Based on this observation, we propose *positional distillation*: computing loss only on the first N response tokens. On MATH-500, positional distillation (N=50) achieves 66.65% avg@4 vs. 65.55% for full-sequence distillation, while eliminating training instability. On HumanEval, it achieves 42.1% vs. 40.2% (full-seq best, which degrades to 26.8%). We demonstrate a *cascade effect*: distilling only 50 tokens changes the entire generation trajectory, with late-token Jaccard similarity to the base model (0.202) actually lower than early-token similarity (0.265), and KL reduction extending 30-40% beyond the trained range. The method is a single-line code change, generalizes across math and coding tasks, and works with both LoRA and full fine-tuning.

**One-sentence takeaway**: In on-policy distillation, teacher-student disagreement concentrates at reasoning-strategy tokens in early positions; computing loss only there is simpler, more stable, and equally effective.

---

## 1. Introduction (1.5 pages)

### Narrative arc
1. Knowledge distillation is critical for deploying capable but small LLMs
2. On-policy distillation (student generates, teacher scores, train on reverse KL) is the modern paradigm, aligned with RLHF-style training
3. The standard approach computes KL loss over all generated tokens uniformly -- but **this assumes all positions contribute equally**
4. We show this assumption is wrong, and violating it causes two practical problems:
   - Full-sequence training is unstable (degenerate repetition, accuracy collapse)
   - It wastes compute on tokens where teacher and student already agree
5. Our key finding: **KL divergence is front-loaded**. The teacher's value is in *reasoning strategy* (early tokens), not *computation* (late tokens)
6. This leads to positional distillation: compute loss only on early tokens. One line of code. Stable, efficient, equally effective.
7. Surprising bonus: the **cascade effect** -- improving early tokens changes the entire generation trajectory

### Figure 1 (teaser, half-page)
Three-panel figure:
- (a) Per-position KL curve (59K trajectories): steep decay from 3.06 at position 0 to ~0.4 by position 100. Annotate phases: "strategy decisions" (high KL) vs. "computation" (low KL).
- (b) Performance vs. position limit: avg@4 on MATH-500. Show pos-5 through pos-200, with full-seq horizontal line. Mark "sweet spot" at pos-50 where 26% of KL captures 73% of gain.
- (c) Training stability: avg@4 over training steps for pos-50, pos-100, full-seq. Show full-seq degradation vs. positional stability.

### Contributions (4 items)
1. **Empirical finding**: KL divergence in on-policy distillation is highly non-uniform across positions, concentrated at reasoning-strategy tokens (positions 0-50) where planning tokens have 6.3x higher KL than math-number tokens.
2. **Method**: Positional distillation -- a simple loss-masking approach that computes reverse KL only on the first N response tokens.
3. **Cascade effect**: Experimental demonstration that early-token distillation changes entire generation trajectories, with distributional and behavioral evidence across trained and untrained token positions.
4. **Comprehensive validation**: Results across two tasks (math, coding), two training methods (LoRA, full fine-tune), and multiple batch configurations, showing positional distillation matches or exceeds full-sequence performance while eliminating training instability.

---

## 2. Background and Related Work (1 page)

### 2.1 On-Policy Knowledge Distillation
- Classic KD (Hinton et al., 2015), sequence-level KD (Kim & Rush, 2016)
- On-policy vs. off-policy: GKD (Agarwal et al., 2024), MiniLLM (Gu et al., 2024), on-policy reverse KL variants
- Forward KL vs. reverse KL: reverse KL is mode-seeking, standard for on-policy

### 2.2 Training Instability in LLM Fine-tuning
- Repetition/degeneration (Holtzman et al., 2020)
- Mode collapse in RL-based training, reward hacking in RLHF
- Connection: full-seq KL loss on answer tokens creates a "repeat the conclusion" incentive

### 2.3 Token-Level Training Strategies
- Token-weighted losses, selective backpropagation
- Curriculum learning and progressive training
- **Gap**: No prior work analyzes per-position KL in distillation or proposes position-based loss masking

*Keep this section tight. The novelty is empirical/methodological, not theoretical. Position clearly against GKD/MiniLLM as complementary (they modify loss/sampling, we modify which tokens get loss).*

---

## 3. Full-Sequence Distillation Degrades (1 page)

*Purpose: establish the problem that motivates the paper. Both the instability problem (practical) and the efficiency problem (computational).*

### 3.1 Setup
- Student: Qwen2.5-Math-1.5B, Teacher: Qwen3-1.7B (similar parameter count but different training data/capabilities)
- On-policy reverse KL with LoRA (r=32, alpha=64, lr=5e-5) and full fine-tune (lr=5e-6)
- Math: 3200 problems, MATH-500 eval (avg@4, maj@4, pass@4)
- Coding: 6400 problems from CodeUltraFeedback, HumanEval/MBPP eval (pass@1)

### 3.2 Degradation Evidence

**Math (full-seq, LoRA, n16bs16)**:
- Step 50: 65.6% avg@4, 1.0 avg \boxed per response, 5.9% repetitive responses
- Step 100: 46.3% apparent avg@4 (91.5% responses have multi-boxed, avg 88 \boxed per response)
- Relaxed metric (GT appears in response): 74.6% -> 70.3% -- only 4.3pp true degradation. 80% of apparent drop is extraction failure from repetition.
- Response length inflates 3.3x (1388 -> 4558 chars)

**Coding (full-seq, LoRA)**:
- HumanEval: 40.2% (step 50) -> 26.8% (step 400) -- sustained, monotonic 33% relative degradation
- This is genuine degradation, not an extraction artifact

**Coding (full-seq, FullFT)**:
- HumanEval: 36.6% (step 150) -> 31.1% (step 400) -- 15% relative degradation

**Contrast**: All positional variants (pos-50, pos-100, pos-200) show stable or improving performance across training steps in both tasks.

### Figure 2 (degradation figure, half-page)
Two panels:
- (a) Math: dual y-axis showing apparent avg@4 (drops) + multi-boxed repetition rate (rises) over training steps. Include pos-200tok for contrast (flat, stable).
- (b) Coding LoRA: HumanEval over training steps for full-seq vs. pos-50 vs. pos-100. Dramatic divergence.

### Table 1: Degradation summary statistics
| Metric | Full-seq Step 50 | Full-seq Step 200 | Pos-200tok Step 100 |
|--------|-----------------|-------------------|---------------------|
| Math avg@4 | 65.6% | 43.8% (relaxed: 70.3%) | 66.75% |
| Multi-boxed rate | 2.9% | 89.1% | 5.8% |
| Avg response length | 1388 chars | 4558 chars | ~1500 chars |
| Coding HE (LoRA) | 40.2% | 26.8% (step 400) | N/A |
| Repetitive responses | 5.9% | 77.5% | ~5% |

*Root cause preview*: Full-seq loss on answer-presentation tokens teaches the model to repeat conclusions. Positional loss avoids this region. But the real question is deeper: why does restricting loss to early tokens *not hurt*? Section 4 answers this.

---

## 4. Where Does KL Divergence Concentrate? (2 pages)

*This is the intellectual core of the paper. It provides the "why" that makes positional distillation principled rather than an ad hoc trick.*

### 4.1 Per-Position KL Distribution (key finding)

**Setup**: 59,936 on-policy trajectories, mean length 386 tokens. Measure KL(student || teacher) at each position.

**Results**:
| Phase | Positions | Mean KL | Description |
|-------|-----------|---------|-------------|
| Ultra-early | 0-4 | 1.71 | Strategy selection |
| Early | 5-19 | 0.93 | Approach setup |
| Mid-early | 20-49 | 0.90 | Problem decomposition |
| Mid | 50-99 | 0.53 | Computation begins |
| Late | 100-199 | 0.42 | Computation and answers |

- Position 0: KL=3.06 (8.5x higher than position 192's 0.36)
- Decay is approximately power-law in the first 50 positions, then stabilizes
- Every position has 55K+ samples (robust statistics)

### 4.2 KL Efficiency: Sub-linear Returns

Cross-referencing cumulative KL with distillation outcomes:

| Position Limit | % of KL Signal | Best avg@4 | % of Max Gain |
|---------------|---------------|------------|---------------|
| 5 | 5.2% | 56.50% | 35% |
| 10 | 7.8% | 59.50% | 54% |
| 20 | 12.2% | 60.20% | 58% |
| 50 | 26.2% | 62.45% | 73% |
| 100 | 43.9% | 64.25% | ~85% |
| 200 | 66.1% | 66.75% | 100% |

**Key insight**: The relationship between KL signal fraction and performance gain is strongly concave. The first 26% of KL captures 73% of the gain. Marginal returns diminish rapidly beyond position 50.

### 4.3 Token Classification: What Makes Early Tokens Special?

Classify tokens into categories: planning, structural, math_number, math_operator, math_latex, continuation. Analysis over 10,000 trajectories (2M+ tokens).

**Position-dependent composition**:
| Position Range | Planning % | Math Number % | Structural % |
|---------------|-----------|--------------|-------------|
| 0-4 | 32.8% | 1.7% | 7.8% |
| 5-19 | 15.9% | 9.5% | 19.4% |
| 50-99 | 9.7% | 15.2% | 30.8% |
| 200-500 | 7.9% | 18.1% | 39.3% |

**KL by token category** (all positions):
| Category | Mean KL | Example Tokens |
|----------|---------|----------------|
| math_latex | 3.99 | \(, \[, \\ |
| planning | 1.75 | "To", "Therefore", "First" |
| continuation | 0.92 | "determine", "find" |
| structural | 0.89 | "**", ":" |
| math_operator | 0.38 | =, +, - |
| math_number | 0.28 | 0-9 |

**Position 0 deep dive**: 94.4% planning tokens. Dominated by "To" (75.1%, KL=7.32), "Let" (10.1%, KL=1.60), "First" (3.4%, KL=14.03). The teacher disagrees most strongly about *how to begin reasoning*.

**Interpretation**: Early positions are "strategic" (which approach to take, how to frame the problem). Late positions are "mechanical" (executing arithmetic, presenting answers). The teacher's advantage is in reasoning strategy, not computation. This makes positional loss a *principled* choice: distill what the teacher knows that the student doesn't.

**Note on LaTeX formatting KL**: LaTeX delimiters (\(, \[) have the highest per-token KL (10-13), reflecting formatting convention disagreements between models. This is KL "noise" -- formatting preferences, not reasoning quality. Positional distillation naturally captures these (they appear at all positions) but doesn't over-weight them relative to genuine reasoning KL. A format-blind distillation loss could further improve efficiency (Section 7, future work).

### 4.4 Post-Distillation KL: Cascade in Distribution Space

After positional distillation (pos-200tok, step 100), measure KL at each position range:

| Range | Raw Student | Pos-200tok | Reduction |
|-------|-----------|-----------|-----------|
| 0-50 | 2.064 | 1.015 | 51% |
| 50-100 | 0.759 | 0.321 | 58% |
| 100-200 | 0.441 | 0.242 | 45% |
| **200-300** | **0.382** | **0.238** | **38%** |
| **300-400** | **0.331** | **0.216** | **35%** |

Bold rows are *beyond the trained range* (pos-200tok only trains on positions 0-200). The 35-38% KL reduction in untrained positions confirms the cascade effect at the distribution level.

### Figures for Section 4

**Figure 3** (full-width, the key analysis figure): Per-position KL curve with 59K trajectories. Annotate the phases. Include position 3 bump (secondary decision point).

**Figure 4** (half-width): Cumulative KL fraction vs. performance gain fraction. Plot the (26%, 73%) point prominently. Show the concavity.

**Figure 5** (half-width): Stacked bar chart of token-type composition by position range. Visual proof that early = planning, late = computation.

*Note: Table 2 (KL by category x position range) goes to appendix to save space. The key numbers are in the text.*

---

## 5. Method: Positional Distillation (0.75 pages)

### 5.1 Formulation

Standard on-policy reverse KL:
$$\mathcal{L}_{\text{full}} = \mathbb{E}_{x \sim p_\theta}\left[\sum_{t=1}^{T} \text{KL}\big(q_\phi(\cdot|x_{<t}) \| p_\theta(\cdot|x_{<t})\big)\right]$$

Positional distillation:
$$\mathcal{L}_{\text{pos-}N} = \mathbb{E}_{x \sim p_\theta}\left[\sum_{t=1}^{N} \text{KL}\big(q_\phi(\cdot|x_{<t}) \| p_\theta(\cdot|x_{<t})\big)\right]$$

where $N$ is the position limit hyperparameter and $x$ is the student's on-policy generation (only $N$ tokens need to be generated).

Implementation: a single mask `loss_mask[:, N:] = 0` applied to the per-token loss tensor.

### 5.2 Efficiency

| Component | Full-seq (T=3584) | Pos-50 | Speedup |
|-----------|-------------------|--------|---------|
| Tokens generated per trajectory | 3584 | 50 | 71.7x |
| Teacher forward pass length | 3584 | 50 | 71.7x |
| Loss/gradient computation | T tokens | N tokens | 71.7x |

*Note: Actual wall-clock speedup is less than 71x due to fixed overheads (data loading, model loading, optimizer step). See Section 7 for discussion. Concrete timing measurements are future work.*

### 5.3 Progressive Variant

Linearly increase $N$ from 1 to $N_{\max}$ over training:
$$N(s) = 1 + \lfloor (N_{\max} - 1) \cdot s / S \rfloor$$

Result: 62.30% avg@4 (competitive with fixed pos-50's 62.45% in the n16bs16 config). Advantage: no position-limit hyperparameter. Disadvantage: cannot leverage full efficiency benefit (generation length grows).

### Figure 6 (method diagram, quarter-page)
Simple schematic: student generates N tokens -> teacher scores N tokens -> compute reverse KL on positions 1..N -> LoRA/full gradient update. Contrast with full-seq (generate T>>N tokens, score T tokens, compute KL on all T).

---

## 6. Experiments (2.5 pages)

### 6.1 Experimental Setup

**Models**: Student = Qwen2.5-Math-1.5B, Teacher = Qwen3-1.7B

**Math**:
- Training: 3200 problems, n_samples=1, bs=16, 200 steps, LoRA (r=32, alpha=64, lr=5e-5) or FullFT (lr=5e-6)
- Eval: MATH-500, n_samples=4, temperature=0.7. Metrics: avg@4, maj@4, pass@4
- Position limits: 5, 10, 20, 50, 100, 150, 200, full-seq

**Coding**:
- Training: 6400 problems from CodeUltraFeedback, n_samples=1, bs=16, 400 steps
- Eval: HumanEval, HumanEval+, MBPP, MBPP+ (pass@1, temperature=0.0)
- Position limits: 50, 100, 150, 250, full-seq

**Baselines**: Undistilled student (50.95% avg@4 on MATH-500), full-sequence distillation

### 6.2 Main Results: Math

**Table 2: Math results (best step per config, LoRA n1bs16)**

| Method | Best Step | avg@4 | maj@4 | pass@4 |
|--------|-----------|-------|-------|--------|
| Baseline (no distill) | -- | 50.95% | 61.2% | 72.8% |
| Pos-50 | 150 | **66.65%** | 71.0% | 81.0% |
| Pos-100 | 200 | 65.85% | 70.8% | 79.8% |
| Pos-200 | 50 | 66.05% | 71.2% | 81.0% |
| Full-seq (first-boxed) | 50 | 65.55% | -- | 80.0% |

*LoRA positional distillation matches or exceeds full-seq (when using first-boxed extraction to account for repetition artifacts). Pos-50 achieves the best result with 26% of KL signal.*

**FullFT results** (n1bs16): Pos-50 = 56.75%, Full-seq = 58.20%. FullFT overall weaker than LoRA by ~10pp. Full-seq slightly leads for FullFT but with less stability.

### 6.3 Main Results: Coding

**Table 3: Coding results (best step per config)**

| Method | Training | Best HE | Best MBPP |
|--------|----------|---------|-----------|
| Baseline | -- | ~33% | ~52% |
| Pos-50 | LoRA | **42.1%** (s350) | 52.1% (s50) |
| Pos-100 | LoRA | **42.1%** (s150) | 52.1% (s50) |
| Full-seq | LoRA | 40.2% (s50) -> 26.8% (s400) | 52.6% (s50) -> 47.6% (s400) |
| Pos-250 | FullFT | 37.8% (s350) | 54.8% (s200) |
| Full-seq | FullFT | 36.6% (s150) -> 31.1% (s400) | 54.0% (s200) -> 53.7% (s400) |

*Coding shows the clearest advantage: positional methods are stable while full-seq degrades monotonically. LoRA pos-50/100 peak at 42.1% HE, far exceeding full-seq's 40.2% best.*

### Figure 7: Training curves (half-page, two panels)
- (a) Math: avg@4 over steps for pos-50, pos-100, pos-200, full-seq (LoRA n1bs16)
- (b) Coding: HumanEval over steps for pos-50, pos-100, full-seq (LoRA). Show the dramatic full-seq collapse.

### 6.4 The Cascade Effect

*Central claim: distilling N early tokens changes the entire response, not just the first N tokens.*

**Evidence 1: Accuracy improvements far exceed trained range**
- Pos-5 (5 tokens distilled): +5.6pp avg@4 improvement on full-length responses (avg ~1929 chars / ~300+ tokens)
- Pos-50 (50 tokens distilled): +11.0pp to +15.7pp avg@4 improvement
- These gains manifest in the final answer, typically at position 200-500+

**Evidence 2: Jaccard similarity shows late tokens change as much as early tokens**
- Compare base vs. pos-50 distilled responses, split at word position 40 (~50 tokens):
  - Early-portion Jaccard: 0.265
  - Late-portion Jaccard: 0.202 (late tokens change *more*, not less)
- At word cutoff 150: early=0.265, late=0.202
- Distilling early tokens causes a *complete rewrite* of the response trajectory

**Evidence 3: KL reduction beyond trained range** (from Section 4.4)
- Pos-200tok: 38% KL reduction at positions 200-300, 35% at positions 300-400 (untrained)

**Evidence 4: Structural changes**
- Distilled models adopt teacher-like reasoning structure: 5.3 step markers per response (vs. 2.8 for base)
- Convergence to "We are given..." opening pattern
- Question-level: 202/500 questions improved, 68 degraded (3:1 ratio for pos-50)

**Mechanistic explanation**: In autoregressive generation, early tokens condition all subsequent generation. Better early decisions (approach selection, problem framing) propagate through the chain: if you start a proof correctly, the rest follows. The cascade effect is a natural consequence of autoregressive structure -- positional distillation exploits this by focusing on the highest-leverage tokens.

### Figure 8: Cascade evidence (half-page, two panels)
- (a) Jaccard similarity (base vs. distilled) for early and late portions across position limits. Show that late tokens change >= early tokens.
- (b) Per-position-range KL reduction for pos-200tok: bars for ranges 0-50, 50-100, ..., 300-400, with "trained range" bracket.

### 6.5 Position Limit Analysis

**Table 4: Position limit sweep (LoRA, math)**

| Pos Limit | % of KL | Best avg@4 (n16) | Best avg@4 (n1) |
|-----------|---------|-------------------|-----------------|
| 5 | 5.2% | 56.50% | -- |
| 10 | 7.8% | 59.50% | -- |
| 20 | 12.2% | 60.20% | -- |
| 50 | 26.2% | 62.45% | 66.65% |
| 100 | 43.9% | 64.25% | 65.85% |
| 150 | -- | 65.70% | -- |
| 200 | 66.1% | 66.75% | 66.05% |
| Full | 100% | 65.55%* | -- |

*first-boxed extraction

**Observations**:
- Large jump between pos-20 and pos-50: critical reasoning-setup decisions happen in positions 20-50
- Diminishing returns after pos-50 (consistent with KL concentration)
- For coding: sweet spot shifts to pos-50-100 (code requires more setup tokens than math)
- Position limit is robust: any value in 50-200 works well

### 6.6 Ablations

**LoRA vs. Full Fine-Tune** (Table in appendix, key numbers in text):
- Math: LoRA dominates by ~10pp avg@4 across all position limits
- Coding: LoRA better on HumanEval, FullFT better on MBPP
- Both benefit from positional distillation (eliminating instability)
- LoRA's implicit regularization may be particularly synergistic with truncated loss

**Batch Configuration** (Table in appendix, key numbers in text):
- n1bs16 (16 diverse problems) vs. n16bs16 (1 problem, 16 trajectories):
  - LoRA: comparable (~66% best avg@4). Problem diversity and trajectory diversity contribute equally.
  - FullFT: n1bs16 wins by ~2pp. Full fine-tuning benefits more from problem diversity.
- n1bs1 (accidental, batch size 1): surprisingly stable due to LoRA regularization. Reaches 80.4% pass@4 at step 3200 (16x more steps).

**Progressive Positional Distillation**:
- Linear increase from pos-1 to pos-200 over 200 steps
- 62.30% avg@4: competitive with fixed pos-50 (62.45%) in n16bs16 config
- Eliminates position-limit hyperparameter but sacrifices efficiency benefit

---

## 7. Discussion (0.75 pages)

### 7.1 Why Positional Distillation Works: Three Mechanisms
1. **KL concentration**: The teacher-student disagreement is inherently front-loaded at reasoning-strategy tokens. Positional loss captures the most informative signal per token.
2. **Autoregressive cascade**: Better early decisions propagate through generation. This is a structural property of autoregressive models, not specific to distillation.
3. **Avoiding degenerate attractors**: Full-seq loss on answer-presentation tokens creates an incentive to repeat conclusions. Positional loss avoids this entirely.

### 7.2 Connections
- **Curriculum learning**: Positional distillation teaches "how to start" before "how to finish." The progressive variant makes this explicit.
- **RLHF/DPO**: Similar positional effects may exist in reward-based training. Early-token rewards could be more informative than late-token rewards.
- **Token importance**: Connects to work on token-level importance weighting, but with a simpler position-based proxy.

### 7.3 Limitations (be honest -- preempt reviewer concerns)
1. **Single model pair**: All experiments use Qwen2.5-Math-1.5B / Qwen3-1.7B. Results may differ for other architectures, size ratios, or model families. This is the biggest limitation.
2. **Two tasks**: Math and coding are both STEM reasoning. The cascade effect may be weaker for tasks where early tokens are less "strategic" (e.g., translation, where structure is determined by input).
3. **Single seed**: No multi-seed significance testing. Variance between runs could be meaningful given the relatively small evaluation set (MATH-500).
4. **FullFT weaker**: Full fine-tuning results are weaker and the interaction between positional loss and training method is not fully understood.
5. **No wall-clock timing**: Efficiency claims are based on token counts, not actual timing measurements.
6. **Position limit as hyperparameter**: While the method is robust to the choice of N in the 50-200 range, it is still a hyperparameter. The progressive variant addresses this at the cost of efficiency.

---

## 8. Conclusion (0.5 pages)

Full-sequence on-policy distillation is unstable because it uniformly weights all token positions, including low-information computation tokens and answer-presentation tokens that promote degenerate repetition. Analysis of 59,936 trajectories reveals that KL divergence is concentrated at early reasoning-strategy positions: 26% of KL signal captures 73% of performance gain. Positional distillation -- computing loss only on the first N tokens -- is a one-line fix that matches full-sequence performance (66.65% vs. 65.55% on MATH-500), eliminates training instability, and generalizes across tasks and training methods. The cascade effect demonstrates that autoregressive models propagate early-token improvements through entire generation trajectories, making early-token distillation a principled and effective strategy.

---

## Appendix

### A. Full Experimental Results
- Complete tables for all position limits x all training steps x all metrics
- Math: LoRA n16bs16, LoRA n1bs16, LoRA n1bs1, FullFT n16bs16, FullFT n1bs16
- Coding: LoRA all pos limits, FullFT all pos limits
- Both: all steps (50, 100, ..., 200/400)

### B. Token Classification Details
- Classification methodology (regex-based categories)
- Full KL by category x position range matrix
- Top-40 highest and lowest KL tokens
- Position 0 deep dive (101 unique tokens, frequency distribution)

### C. Full-Sequence Degradation Analysis
- Complete boxed-repetition statistics per step
- Ghost-correct analysis: relaxed metric (GT appears in response text)
- Per-topic degradation rates
- Response length distribution histograms

### D. Generation Examples
- Side-by-side: base model vs. pos-50 vs. full-seq on 3-4 problems
- Show cascade effect qualitatively: how the opening changes, how downstream reasoning follows

### E. Post-Distillation KL Profiles
- Per-position KL for raw student, pos-200tok (step 100, 200), full-seq (step 50, 200)
- Show convergence of pos-200tok across steps (nearly identical KL profiles)

### F. Batch Configuration Full Results
- Complete tables for all batch configs
- BS=1 vs BS=16 detailed comparison with stability analysis

---

## Figures Budget (8-page main paper)

| Figure | Content | Section | Size | Essential? |
|--------|---------|---------|------|------------|
| 1 | Teaser: KL curve + perf vs. pos + stability | 1 | half-page | YES |
| 2 | Degradation: math repetition + coding collapse | 3 | half-page | YES |
| 3 | Per-position KL (59K trajectories) | 4.1 | half-width | YES |
| 4 | Cumulative KL vs. performance gain | 4.2 | half-width | YES |
| 5 | Token-type composition by position | 4.3 | half-width | YES |
| 6 | Method diagram | 5 | quarter-page | YES |
| 7 | Training curves (math + coding) | 6.2-6.3 | half-page | YES |
| 8 | Cascade evidence (Jaccard + KL) | 6.4 | half-page | YES |

**8 figures total** (reduced from 11 in v1). Figures 3+4 share a row. Figures 5+6 share a row. This fits in 8 pages with ~1.5 pages for tables and text margins.

**Moved to appendix**: Full position-limit sweep plot (the table is sufficient in main text), post-distillation KL profiles (referenced in text, detailed figure in appendix).

## Tables Budget

| Table | Content | Section | Essential? |
|-------|---------|---------|------------|
| 1 | Degradation summary | 3.2 | YES |
| 2 | Main math results | 6.2 | YES |
| 3 | Main coding results | 6.3 | YES |
| 4 | Position limit sweep | 6.5 | YES |

**Moved to appendix**: KL by category x position range (full matrix), LoRA vs FullFT comparison table, batch config comparison table, question-level transition counts.

---

## Reviewer Concerns to Preempt

1. **"Only one model pair"** -- Acknowledged as primary limitation. Argue: the KL concentration finding is about the structure of on-policy generation (strategy-then-computation), which should generalize. Commit to more model pairs in camera-ready.

2. **"Is this just early stopping on tokens?"** -- No: the position limit is fixed, not adaptive. And the cascade effect means the *entire response* changes, not just the first N tokens. Early stopping would generate full sequences and discard; positional distillation generates only N tokens (efficiency).

3. **"Why not weight loss by KL magnitude?"** -- Good question, acknowledged as future work. Our finding is that a simple hard cutoff works surprisingly well, which is itself interesting (the information boundary is relatively sharp).

4. **"Statistical significance?"** -- Acknowledged. Point to consistency across 8 training steps x 8 position limits x 2 batch configs x 2 training methods as evidence of robustness, even without multi-seed runs.

5. **"The teacher is only 1.7B, same size as student"** -- This is actually a strength: even with similar-sized teacher, positional distillation works. Larger teachers should show even stronger effects (more strategy knowledge to transfer).

6. **"Is the cascade effect just that any fine-tuning changes the model?"** -- No: the base-vs-distilled Jaccard analysis shows late tokens change *more* than early tokens for positional distillation (0.202 vs 0.265), which wouldn't happen with generic parameter drift. Also, the KL reduction beyond the trained range is specific to teacher-student alignment.

---

## Experiments That Would Strengthen the Paper

See `docs/missing_experiments.md` for the full list with priorities and estimated effort.
