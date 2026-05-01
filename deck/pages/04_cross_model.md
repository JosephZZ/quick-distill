# Choosing $N$: The 45% KL Coverage Heuristic

### How long should the prefix be? Pick $N$ where cumulative KL hits ~45%.

<br>

| Model Pair | KL Profile | Ratio (first100/rest) | Optimal N | Coverage at optimum |
|------------|-----------|----------------------|-----------|-------------------|
| **Qwen** 1.5B → 1.7B | Front-loaded | **2.81×** | N = 100 | **44%** |
| **Gemma** 2B → 4B | Mid | ~**2.5×** | N = 50–100 | **40%** |
| **Llama** 1B → 8B | Flat | **1.15×** | N = 200 | **45%** |

<br>

### The 45% descriptive heuristic: optimal $N$ falls at the prefix capturing **40–45% cumulative KL**

- **Front-loaded** profiles (Qwen, Gemma): teacher signal concentrated early → N = 50–100
- **Flat** profiles (Llama): signal spread evenly → need more tokens, N = 150–200
- **Same coverage fraction** despite very different absolute KL values

<br>

**Practical recipe**: Run ~1,000 trajectories, compute cumulative KL profile, pick $N$ where coverage hits ~45%. No grid search needed.

⚠️ Calibrated on n=3 families. Presented as a *starting point*, not a validated law.

<!--
[~1.5 min]
The observation generalizes. Different model pairs need different N,
but the underlying principle (40-50% coverage) is universal.
-->
