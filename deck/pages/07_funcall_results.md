# Function Calling: Students Exceed Their Teachers

### BFCL full_acc — the most surprising result

<br>

| | Teacher | Student baseline | **Best positional** | Fullseq |
|--|---------|-----------------|-------|---------|
| **Qwen** 1.5B→1.7B | 54.0% | 2.7% | **61.3%** ↑ (N=100) | 58.2% |
| **Gemma** 2B→4B | 25.0% | 0% | **30.9%** ↑ (N=50) | 3.9% |
| **Llama** 1B→8B | 63.7% | 55.3% | **59.0%** (N=150) | 32.0% |

<br>

### Why can students exceed teachers?

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

**Teacher's OUTPUT is wrong:**
- Qwen3-1.7B outputs **natural language** math solutions, not JSON
- Gemma-3-4B outputs **tool_code** format, not JSON

</div>
<div style="width: 50%;">

**Teacher's DISTRIBUTION is right:**
- Early-position probabilities encode *which function to call*
- Student extracts this signal, outputs in its own (correct) JSON format
- **Distributional knowledge > generation ability**

</div>
</div>

<!--
[~2 min]
This is the "wow" result. Student beats teacher by 7pp.
The mechanism: teacher's distribution is better than its samples.
Positional distillation extracts the cleanest part of that distribution.
-->
