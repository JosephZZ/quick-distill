# Honest limitations

### Where we don't have evidence

1. **Scale.** Largest student studied is 8B (Llama 1B → 8B). We do not yet know how the elbow shifts at 70B+ scales.
2. **Long-form generation.** Math/code/funcall responses are < 1k tokens. We have no tested guidance for multi-thousand-token CoT or agentic trajectories.
3. **Suffix-only experiments not yet run.** A clean test of "is the planner-tail asymmetry literally about the *position*, or about the *content type*?" requires suffix-only training, which is queued but not finished.
4. **Mode-seeking advantage assumes the teacher distribution covers the truth.** If the teacher's top-K does *not* include the correct branch, mode-seeking concentrates on the *wrong* mode. We saw this on Llama BFCL where the student fell 4.7pp below teacher.
5. **Single seed for some cross-family configs.** Stability table is for Qwen-math; cross-family runs are 1-seed.

<br>

### What we do *not* claim

- That position is a universal prior. It's an indicator that **wins under the OPD reverse-KL pipeline at small-to-mid scale**.
- That cascade theory is proven. It's consistent with all our evidence; we have not falsified plausible alternatives.

<!--
[~1 min] Clear, narrow limitations. Don't overclaim — the paper is much stronger if Part 2's
empirics are a load-bearing wall and Part 3 is acknowledged interpretation.
-->
