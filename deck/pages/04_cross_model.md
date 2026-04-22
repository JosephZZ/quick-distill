# This Pattern Holds Across Model Families

Different families have different KL profiles — but the same principle applies.

<br>

| Model Pair | KL Profile | Ratio (first100/rest) | Optimal N | Coverage at optimum |
|------------|-----------|----------------------|-----------|-------------------|
| **Qwen** 1.5B → 1.7B | Front-loaded | **2.81×** | N = 100 | ~44% |
| **Llama** 1B → 8B | Flat | **1.15×** | N = 200 | ~45% |

<br>

### The rule: optimal N ≈ 40–50% cumulative KL coverage

- **Front-loaded** profiles (Qwen, Gemma): teacher signal concentrated early → N = 50–100
- **Flat** profiles (Llama): signal spread evenly → need more tokens, N = 150–200
- **Same coverage fraction** despite very different absolute KL values

<br>

**Practical heuristic**: Run ~1,000 trajectories, compute KL profile, pick N at 40–50% cumulative KL. No grid search needed.

<!--
[~1.5 min]
The observation generalizes. Different model pairs need different N,
but the underlying principle (40-50% coverage) is universal.
-->
