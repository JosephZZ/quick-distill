# Hypothesis B: Top-KL Failure Analysis

## Hypothesis
High-KL tokens are "teacher confusion" (high entropy, unreliable gradient) rather than "useful disagreement" (low entropy, teacher knows better).

## Results (50 problems, Qwen2.5-Math-1.5B → Qwen3-1.7B)

| Metric | Top-KL tokens (k=100) | Prefix tokens (0-99) | Late tokens (100+) |
|--------|----------------------|----------------------|-------------------|
| Teacher entropy | **0.449** | 0.235 | 0.158 |
| Student entropy | 0.696 | 0.466 | — |
| Mean KL | 4.092 | 2.015 | 1.809 |
| Mean position | 127 | 0-49 (center) | 100+ |

## Key Finding

**Hypothesis B SUPPORTED.** Top-KL tokens have teacher entropy 1.9× higher than prefix tokens (0.449 vs 0.235).

This means:
1. **Top-KL tokens are where teacher is CONFUSED** — high entropy + high KL = teacher doesn't know the answer either, just disagrees with student randomly
2. **Prefix tokens are where teacher is CONFIDENT but student is WRONG** — moderate KL + low entropy = reliable correction signal
3. **Late tokens are where teacher AGREES** — low KL + very low entropy = rubber-stamping

The ideal distillation signal is: **low teacher entropy + high KL** (teacher knows better, student is wrong).
Top-KL selects the opposite: **high teacher entropy + high KL** (both confused, maximum noise).

## Implication for Paper
This provides the mechanistic explanation for why signal-strength-based selection fails:
- "The teacher's per-token KL divergence conflates two distinct components: confident correction (useful) and confused disagreement (harmful). Prefix positions are dominated by confident correction, while top-KL positions select for confused disagreement."
