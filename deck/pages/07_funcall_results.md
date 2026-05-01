# Function Calling: When Students Beat Teachers

### BFCL full_acc — the "wow" result

<br>

| | Teacher | Student baseline | **Best positional** | Fullseq |
|--|---------|-----------------|-------|---------|
| **Qwen** 1.5B→1.7B | 54.0% | 2.7% | **61.3%** ↑ (N=100) | 58.2% |
| **Gemma** 2B→4B | 25.0% | 0% | **30.9%** ↑ (N=50) | 3.9% |
| **Llama** 1B→8B | 63.7% | 55.3% | **59.0%** (N=150) | 32.0% |

<br>

### A reverse-KL mechanism: small student can find a mode the teacher knows but doesn't peak on

<div style="display: flex; gap: 2rem; align-items: flex-start;">
<div style="width: 50%;">

**Teacher's *output* is wrong format**:
- Qwen3-1.7B emits **natural language**, not JSON
- Gemma-3-4B emits **`tool_code`**, not JSON

**But teacher's *distribution* knows the right call**:
- 50% of teacher rollouts mention the correct function name
- 20% even use correct call syntax in the wrong wrapper

The right mode lives in the teacher's distribution — it just isn't the **argmax** mode.

</div>
<div style="width: 50%;">

**Why on-policy reverse-KL extracts it:**

$D_{\text{KL}}(p_\theta \| p_T)$ is **mode-seeking**:
- penalizes student putting mass where teacher has none
- *does not* require student to cover all of teacher's mass

So the student is free to **commit to one mode**, as long as it's a mode the teacher also supports.
- Student samples on-policy → occasionally lands on JSON
- At those tokens, teacher distribution still gives high probability to the correct **content**
- Student locks in the JSON wrapper *and* the right function name

→ Student gets **both modes right** in a way the teacher itself doesn't.

</div>
</div>

<br>

### Why the same trick doesn't work on long math

In long-form reasoning, the *correct* answer is short and the *wrong* trajectories are long. After 200 tokens of execution, the student's distribution is dominated by the (long) modes that lead nowhere useful — and reverse-KL faithfully reinforces those.

→ The student-exceeds-teacher phenomenon is real but **scenario-specific**: short answers + multimodal teacher distribution. Generalizing this is open future work.

<!--
[~2 min]
Position 14 in the new ordering.
This is the "bonus" insight slide.
Mode-seeking explanation is the key new conceptual content here —
it gives a theoretical handle on when small models can exceed teachers,
and flags why it doesn't generalize to long-form reasoning.
-->
