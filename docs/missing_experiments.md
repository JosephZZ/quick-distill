# Missing Experiments for Paper Strengthening

Extracted from paper outline v5 analysis, cross-referenced with all available experiment data. Organized by priority for acceptance at a top venue (ICML/NeurIPS).

---

## Tier 1: Critical for Acceptance

### 1. Additional Student-Teacher Pairs
**Why**: Single model pair is the paper's biggest weakness. Reviewers will question generalizability.
**What to run**:
- Llama-3.2-1B (student) / Llama-3.1-8B (teacher) on MATH-500
  - Tests cross-family generality (non-Qwen)
  - Tests different size ratio (1B/8B vs 1.5B/1.7B)
- Qwen2.5-Math-1.5B (student) / Qwen2.5-Math-7B (teacher) on MATH-500
  - Same family but much larger teacher
  - Tests whether KL concentration pattern holds with larger capability gap
**Minimum**: Run pos-50, pos-200, full-seq with LoRA for each pair, 200 steps
**Estimated effort**: 2-3 days GPU time per pair
**Impact**: Would transform the paper from "interesting finding on one pair" to "general phenomenon"

### 2. Multi-Seed Runs for Key Comparisons
**Why**: All current results are single-seed. Reviewers at top venues expect error bars.
**What to run**:
- 3 seeds for: pos-50 LoRA, pos-200 LoRA, full-seq LoRA (math, n1bs16)
- 3 seeds for: pos-50 LoRA, full-seq LoRA (coding)
- Report mean +/- std for all metrics
**Estimated effort**: 1-2 days GPU time
**Impact**: Credibility of all claims. Even if variance is high, showing it honestly helps.

### 3. KL Position Analysis on Coding Tasks
**Why**: KL concentration analysis is math-only. The paper claims cross-task applicability but doesn't verify the underlying mechanism for coding.
**What to run**:
- Generate on-policy trajectories (10K+) from student on coding problems
- Score with teacher, compute per-position KL
- Token classification adapted for code (function defs, imports, logic, comments, syntax)
**Estimated effort**: 1 day generation + analysis
**Impact**: If KL is also front-loaded for code, strongly supports the "general phenomenon" claim. If not, still valuable (different tasks may have different optimal position limits -- which is consistent with pos-50-100 being the coding sweet spot vs. pos-50 for math).

### 4. Wall-Clock Timing Measurements
**Why**: Efficiency claims are based on token counts (71x fewer tokens). Actual speedup depends on fixed overheads.
**What to run**:
- Time one full training step for pos-50, pos-100, pos-200, full-seq
- Break down: generation time, teacher forward pass, loss computation, backward pass, optimizer step
- Report speedups for each component and overall
**Estimated effort**: 1 hour
**Impact**: Converts "up to 71x fewer tokens" into concrete "Nx wall-clock speedup per step"

---

## Tier 2: Strongly Recommended

### 5. Forward KL Comparison
**Why**: Tests whether KL concentration is specific to reverse KL or a general property of teacher-student divergence.
**What to run**:
- Pos-50 and full-seq with forward KL loss (instead of reverse KL)
- Math, LoRA, n1bs16, 200 steps
**Estimated effort**: 1 day
**Impact**: If forward KL shows same positional pattern, the finding is about teacher-student divergence structure, not the specific loss. More general claim.

### 6. Position-Weighted Loss (Soft Masking)
**Why**: The paper proposes a hard cutoff at position N. A reviewer will ask "why not weight by position or by KL magnitude?"
**What to run**:
- Exponential decay weighting: weight_t = exp(-t/tau) for tau = 50, 100
- KL-magnitude weighting: weight_t proportional to empirical per-position KL
- Compare against pos-50 and pos-100 hard cutoffs
**Estimated effort**: 1 day (minor code change)
**Impact**: If hard cutoff matches soft weighting, that's a clean result (simplicity wins). If soft weighting is better, that's a method improvement.

### 7. Token-Type-Weighted Loss
**Why**: Token classification shows planning tokens have 6.3x higher KL than math numbers. Why not weight by token type directly?
**What to run**:
- Upweight planning tokens, downweight math_number tokens (ratios from empirical KL)
- Compare against positional masking (pos-50, pos-100)
**Estimated effort**: 1-2 days (need real-time token classification in training loop)
**Impact**: Tests whether position is a good proxy for token importance, or whether content-based weighting is better. Either outcome is informative.

### 8. Longer Training Runs
**Why**: Current max is 200 steps (math) / 400 steps (coding). Reviewers may ask about long-term stability.
**What to run**:
- Pos-50 and pos-100 LoRA for 1000 steps (math)
- Pos-50 LoRA for 1000 steps (coding)
**Estimated effort**: 2-3 days
**Impact**: Confirms long-term stability of positional distillation. Current 200-step results already suggest stability, but 1000 steps is more convincing.

### 9. Off-Policy Comparison
**Why**: Paper focuses on on-policy distillation. Does positional masking also help when the teacher generates?
**What to run**:
- Generate trajectories from teacher, compute student KL on teacher sequences
- Analyze per-position KL distribution (is it also front-loaded?)
- Train pos-50 and full-seq in off-policy mode
**Estimated effort**: 1-2 days
**Impact**: If the pattern holds off-policy, broadens applicability significantly.

---

## Tier 3: Nice to Have

### 10. Non-STEM Task
**Why**: Math and coding are both STEM reasoning with clear "strategy then computation" structure. Does the cascade effect hold for other tasks?
**What to run**:
- Instruction following: Alpaca-style data, eval on AlpacaEval or MT-Bench
- Or summarization: CNN/DailyMail, eval on ROUGE
- Pos-50, pos-200, full-seq comparison
**Estimated effort**: 2-3 days
**Impact**: If it works, massive generality claim. If it doesn't, that's also interesting (the cascade effect is specific to structured reasoning tasks).

### 11. Larger-Scale Models
**Why**: 1.5B student is small by current standards.
**What to run**:
- Qwen2.5-7B (student) / Qwen2.5-72B (teacher), even just one experiment
- Pos-100 vs full-seq, LoRA, MATH-500
**Estimated effort**: 3-5 days (need multi-GPU for 72B teacher)
**Impact**: "This works at scale" is a strong statement for any methods paper.

### 12. GKD / MiniLLM Comparison
**Why**: Positions against concurrent/recent distillation methods.
**What to run**:
- Implement GKD (mixture of on-policy and teacher sampling) with positional masking
- Compare GKD vs GKD + positional masking
**Estimated effort**: 2-3 days (need to implement GKD)
**Impact**: Shows positional masking is complementary to existing methods, not competitive.

### 13. Attention Analysis for Cascade Mechanism
**Why**: Provides mechanistic explanation for the cascade effect.
**What to run**:
- Extract attention patterns from base and distilled models
- Measure attention influence of early tokens on late tokens
- Compare: do distilled models show stronger early->late attention?
**Estimated effort**: 1 day
**Impact**: Mechanistic understanding. Nice figure for the paper but not required.

### 14. Adaptive Position Limits by Problem Difficulty
**Why**: Harder problems may need more tokens of distillation.
**What to run**:
- Split MATH-500 by difficulty (easy/medium/hard based on base model accuracy)
- Analyze per-difficulty optimal position limit
- Test adaptive: pos-20 for easy, pos-100 for hard
**Estimated effort**: 1 day (analysis) + 1 day (training)
**Impact**: Practical improvement and deeper understanding. But the current fixed position limit already works well enough.

### 15. RLHF/DPO Connection
**Why**: If the positional insight applies to reward-based training, the paper has broader impact.
**What to run**:
- Standard DPO on math with positional reward (only early tokens contribute to preference)
- Compare against full-sequence DPO
**Estimated effort**: 3-5 days (need DPO infrastructure)
**Impact**: Opens a new direction. But this is really a separate paper.

---

## Priority Summary

| Priority | Experiment | Effort | Impact |
|----------|-----------|--------|--------|
| **MUST** | Additional model pairs (#1) | 4-6 days | Generality |
| **MUST** | Multi-seed runs (#2) | 1-2 days | Credibility |
| **MUST** | KL analysis on coding (#3) | 1 day | Mechanism validation |
| **MUST** | Wall-clock timing (#4) | 1 hour | Efficiency claim |
| SHOULD | Forward KL (#5) | 1 day | Loss generality |
| SHOULD | Position-weighted loss (#6) | 1 day | Method completeness |
| SHOULD | Token-type weighting (#7) | 1-2 days | Ablation |
| SHOULD | Longer training (#8) | 2-3 days | Stability |
| SHOULD | Off-policy (#9) | 1-2 days | Broader applicability |
| COULD | Non-STEM task (#10) | 2-3 days | Generality |
| COULD | Larger models (#11) | 3-5 days | Scale |
| COULD | GKD comparison (#12) | 2-3 days | Baselines |
| COULD | Attention analysis (#13) | 1 day | Mechanism |
| COULD | Adaptive limits (#14) | 2 days | Refinement |
| BONUS | RLHF/DPO (#15) | 3-5 days | New direction |

**Total estimated effort for MUST items: ~7-10 days**
**Total for MUST + SHOULD: ~12-17 days**
