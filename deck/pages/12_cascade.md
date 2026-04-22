# The Cascade Effect: Early Changes Propagate Everywhere

### Training on first 200 tokens changes the *entire* response

<br>

| Position Range | Raw Student KL | After pos-200 | Reduction |
|---------------|----------------|---------------|-----------|
| 0–50 (trained) | 2.064 | 1.015 | **51%** |
| 50–100 (trained) | 0.759 | 0.321 | **58%** |
| 100–200 (trained) | 0.441 | 0.242 | **45%** |
| **200–300 (untrained)** | **0.382** | **0.238** | **38%** |
| **300–400 (untrained)** | **0.331** | **0.216** | **35%** |

<br>

### KL reduces 35–38% even at positions the model was **never trained on**.

<br>

**Why?** Autoregressive generation means every token conditions all subsequent tokens. Better early decisions → better entire trajectory.

This is not an approximation — it's **leveraging the structure of autoregressive models**.

Positional distillation works because reasoning is a cascade: fix the strategy, and the execution follows.

<!--
[~1.5 min]
The cascade effect is our strongest mechanistic evidence.
Training on 200 tokens changes positions 300-400 by 35%.
This is WHY prefix isn't an approximation — it's the real thing.
-->
