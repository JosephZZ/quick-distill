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

**Teacher's OUTPUT is wrong format:**
- Qwen3-1.7B outputs **natural language**, not JSON
- Gemma-3-4B outputs **tool_code** format

**But teacher KNOWS the right function:**
- 50% of responses **mention correct function name** in natural language
- 20% even use **correct call syntax** (just not JSON)

</div>
<div style="width: 50%;">

**How distillation extracts this:**
1. Student generates `[{"name": "` (its own format)
2. Teacher's distribution at this point has high P(correct function name) — because teacher knows the right function
3. Student learns **which function to call** from teacher's distribution
4. Student outputs in **its own correct JSON format**

**Result: distributional knowledge > generation ability**

</div>
</div>

<!--
[~2 min]
This is the "wow" result. Student beats teacher by 7pp.
The mechanism: teacher's distribution is better than its samples.
Positional distillation extracts the cleanest part of that distribution.
-->
