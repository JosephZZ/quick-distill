"""
Analysis of full-seq distillation degradation from step 50 to step 200.
Full-seq (position_limit=0) trains on ALL response tokens.
Peak at step 50, severe degradation by step 200.

This script analyzes eval results (no GPU needed).
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import statistics

# ============================================================
# Config
# ============================================================
BASE = Path("/CGLab/ziheng/projects/dft-distill/checkpoints")
FULLSEQ_DIR = BASE / "full-seq-3584tok"
POS200TOK_DIR = BASE / "pos-limit-200tok"

EVAL_FILES = {
    "fullseq_step50": FULLSEQ_DIR / "eval_step_50" / "results.jsonl",
    "fullseq_step100": FULLSEQ_DIR / "eval_step_100" / "results.jsonl",
    "fullseq_step150": FULLSEQ_DIR / "eval_step_150" / "results.jsonl",
    "fullseq_step200": FULLSEQ_DIR / "eval_step_200" / "results.jsonl",
    "pos200tok_step100": POS200TOK_DIR / "eval_step_100" / "results.jsonl",
}

OUTPUT_PATH = Path("/CGLab/ziheng/projects/dft-distill/docs/fullseq_degradation_analysis.md")

# ============================================================
# Helpers
# ============================================================
def load_results(path):
    """Load results.jsonl into a dict keyed by idx."""
    results = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            results[item["idx"]] = item
    return results

def classify_topic(question):
    """Rough classification of math topic from question text."""
    q = question.lower()
    # Order matters - check more specific patterns first
    if any(w in q for w in ["probability", "probabilit", "dice", "coin", "randomly", "random", "chance", "expected value", "expectation"]):
        return "Probability/Combinatorics"
    if any(w in q for w in ["how many", "combinat", "permut", "choose", "ways", "arrangements", "selections"]):
        return "Counting/Combinatorics"
    if any(w in q for w in ["triangle", "circle", "rectangle", "square", "polygon", "area", "perimeter",
                             "angle", "parallel", "perpendicular", "radius", "diameter", "circumference",
                             "cylinder", "sphere", "cone", "volume", "surface area", "quadrilateral",
                             "pentagon", "hexagon", "coordinate", "midpoint", "distance between"]):
        return "Geometry"
    if any(w in q for w in ["prime", "divisor", "factor", "gcd", "lcm", "modulo", "mod ", "remainder",
                             "divisible", "congruent", "digit", "base ", "binary", "integer solutions"]):
        return "Number Theory"
    if any(w in q for w in ["matrix", "matric", "determinant", "eigenvalue", "vector", "linear transformation",
                             "rank", "null space", "basis"]):
        return "Linear Algebra"
    if any(w in q for w in ["integral", "derivative", "limit", "continuous", "differentiat", "converge",
                             "series", "sum_{", "sum_\\", "\\sum", "taylor", "maclaurin", "sequence"]):
        return "Calculus/Analysis"
    if any(w in q for w in ["polynomial", "equation", "solve", "root", "zero", "quadratic", "cubic",
                             "inequality", "function", "f(x)", "log", "exponential", "absolute value"]):
        return "Algebra"
    if any(w in q for w in ["convert", "polar", "rectangular", "complex", "imaginary", "i =", "|z|"]):
        return "Precalculus"
    return "Other"

def response_length(resp_text):
    """Length of a response in characters."""
    return len(resp_text)

def response_word_count(resp_text):
    """Word count of a response."""
    return len(resp_text.split())

def avg_correct_count(item):
    """Number of correct responses out of 4."""
    return sum(1 for r in item["responses"] if r["is_correct"])

def detect_repetition(text, min_repeat_len=50):
    """Detect if a response contains repetitive patterns."""
    # Check for repeated substrings
    if len(text) < 200:
        return False, ""

    # Check for very long responses (>3x median would be suspicious)
    # Check for repeated phrases
    sentences = re.split(r'[.!?\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) > 5:
        seen = Counter(sentences)
        most_common = seen.most_common(1)
        if most_common and most_common[0][1] >= 3:
            return True, f"Repeated sentence ({most_common[0][1]}x): '{most_common[0][0][:80]}...'"

    # Check for repeated multi-line blocks
    lines = text.split('\n')
    for block_size in [3, 5, 8]:
        if len(lines) < block_size * 2:
            continue
        blocks = []
        for i in range(0, len(lines) - block_size + 1, block_size):
            block = '\n'.join(lines[i:i+block_size])
            blocks.append(block)
        block_counts = Counter(blocks)
        for block, count in block_counts.most_common(1):
            if count >= 2 and len(block) > min_repeat_len:
                return True, f"Repeated block ({count}x, {len(block)} chars)"

    # Check for character-level repetition: take last 500 chars and see if it repeats earlier
    if len(text) > 1000:
        suffix = text[-300:]
        earlier = text[:-300]
        if suffix in earlier:
            return True, "Last 300 chars appear earlier verbatim"

    return False, ""

def detect_format_issues(text):
    """Detect formatting issues like missing boxed answer, code blocks, etc."""
    issues = []
    if "\\boxed" not in text and "boxed" not in text.lower():
        issues.append("no_boxed_answer")
    if "```python" in text or "```code" in text:
        issues.append("contains_code_block")
    if text.count("```") >= 4:
        issues.append("many_code_blocks")
    if len(text) > 10000:
        issues.append("very_long_response")
    if len(text) < 50:
        issues.append("very_short_response")
    return issues

def pairwise_similarity(responses):
    """Rough measure of response diversity using character-level Jaccard on word sets."""
    word_sets = []
    for r in responses:
        words = set(r["response"].lower().split())
        word_sets.append(words)

    similarities = []
    for i in range(len(word_sets)):
        for j in range(i+1, len(word_sets)):
            if len(word_sets[i] | word_sets[j]) == 0:
                similarities.append(1.0)
            else:
                sim = len(word_sets[i] & word_sets[j]) / len(word_sets[i] | word_sets[j])
                similarities.append(sim)
    return statistics.mean(similarities) if similarities else 0.0

# ============================================================
# Main Analysis
# ============================================================
def main():
    report_lines = []
    _report_buffer = []
    def report(s="", end="\n"):
        if end == "\n":
            full_line = "".join(_report_buffer) + s
            _report_buffer.clear()
            report_lines.append(full_line)
            print(full_line)
        else:
            _report_buffer.append(s)
            print(s, end=end)

    # Load data
    data = {}
    for name, path in EVAL_FILES.items():
        if path.exists():
            data[name] = load_results(path)
            report(f"Loaded {name}: {len(data[name])} problems")
        else:
            report(f"MISSING: {path}")

    report()

    s50 = data.get("fullseq_step50", {})
    s100 = data.get("fullseq_step100", {})
    s150 = data.get("fullseq_step150", {})
    s200 = data.get("fullseq_step200", {})
    p200tok = data.get("pos200tok_step100", {})

    # ============================================================
    # 1. Overall accuracy progression
    # ============================================================
    report("=" * 80)
    report("# Full-Seq Distillation Degradation Analysis")
    report("=" * 80)
    report()
    report("## 1. Overall Accuracy Progression")
    report()

    for name, label in [("fullseq_step50", "Step 50"), ("fullseq_step100", "Step 100"),
                         ("fullseq_step150", "Step 150"), ("fullseq_step200", "Step 200"),
                         ("pos200tok_step100", "Pos-200tok Step 100")]:
        if name not in data:
            continue
        d = data[name]
        n = len(d)
        any_correct = sum(1 for item in d.values() if item.get("any_correct", False) or any(r["is_correct"] for r in item["responses"]))
        total_correct = sum(avg_correct_count(item) for item in d.values())
        avg_acc = total_correct / (n * 4) * 100
        pass_at_4 = any_correct / n * 100

        # Majority voting
        maj_correct = 0
        for item in d.values():
            answers = [r["extracted_answer"] for r in item["responses"]]
            answer_counts = Counter(answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            # Check if majority answer is correct
            for r in item["responses"]:
                if r["extracted_answer"] == majority_answer:
                    if r["is_correct"]:
                        maj_correct += 1
                    break
        maj_acc = maj_correct / n * 100

        report(f"  {label:25s}: avg@4={avg_acc:.2f}%  maj@4={maj_acc:.2f}%  pass@4={pass_at_4:.2f}%")

    report()

    # ============================================================
    # 2. Degraded problems: step 50 correct, step 200 wrong
    # ============================================================
    report("## 2. Degraded Problems (Step 50 correct → Step 200 wrong)")
    report()

    degraded_idxs = []
    improved_idxs = []

    for idx in s50:
        s50_any = any(r["is_correct"] for r in s50[idx]["responses"])
        s200_any = any(r["is_correct"] for r in s200[idx]["responses"])

        if s50_any and not s200_any:
            degraded_idxs.append(idx)
        elif not s50_any and s200_any:
            improved_idxs.append(idx)

    report(f"  Total problems: {len(s50)}")
    report(f"  Degraded (step50 ✓ → step200 ✗): {len(degraded_idxs)}")
    report(f"  Improved (step50 ✗ → step200 ✓): {len(improved_idxs)}")
    report(f"  Net degradation: {len(degraded_idxs) - len(improved_idxs)} problems")
    report()

    # 2a. Topic distribution of degraded problems
    report("### 2a. Topic Distribution of Degraded Problems")
    report()

    degraded_topics = Counter()
    all_topics = Counter()
    for idx in s50:
        topic = classify_topic(s50[idx]["question"])
        all_topics[topic] += 1
        if idx in degraded_idxs:
            degraded_topics[topic] += 1

    report(f"  {'Topic':<30s} {'Degraded':>10s} {'Total':>10s} {'Rate':>10s}")
    report(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    for topic in sorted(all_topics.keys(), key=lambda t: degraded_topics.get(t, 0), reverse=True):
        d = degraded_topics.get(topic, 0)
        t = all_topics[topic]
        rate = d / t * 100 if t > 0 else 0
        report(f"  {topic:<30s} {d:>10d} {t:>10d} {rate:>9.1f}%")
    report()

    # 2b. Response length comparison
    report("### 2b. Response Length Comparison (Degraded Problems)")
    report()

    s50_lens_degraded = []
    s200_lens_degraded = []
    for idx in degraded_idxs:
        for r in s50[idx]["responses"]:
            s50_lens_degraded.append(response_length(r["response"]))
        for r in s200[idx]["responses"]:
            s200_lens_degraded.append(response_length(r["response"]))

    if s50_lens_degraded:
        report(f"  Step 50 avg response length: {statistics.mean(s50_lens_degraded):.0f} chars (median: {statistics.median(s50_lens_degraded):.0f})")
        report(f"  Step 200 avg response length: {statistics.mean(s200_lens_degraded):.0f} chars (median: {statistics.median(s200_lens_degraded):.0f})")
        report(f"  Ratio (step200/step50): {statistics.mean(s200_lens_degraded)/statistics.mean(s50_lens_degraded):.2f}x")
    report()

    # 2c. Failure pattern analysis for step 200 wrong answers
    report("### 2c. Failure Patterns in Step 200 Responses (Degraded Problems)")
    report()

    repetition_count = 0
    format_issue_counts = Counter()
    total_responses_checked = 0
    repetition_examples = []

    for idx in degraded_idxs:
        for r in s200[idx]["responses"]:
            total_responses_checked += 1
            is_rep, rep_detail = detect_repetition(r["response"])
            if is_rep:
                repetition_count += 1
                if len(repetition_examples) < 3:
                    repetition_examples.append((idx, rep_detail))
            issues = detect_format_issues(r["response"])
            for issue in issues:
                format_issue_counts[issue] += 1

    report(f"  Total responses checked: {total_responses_checked}")
    report(f"  Responses with repetitive patterns: {repetition_count} ({repetition_count/total_responses_checked*100:.1f}%)")
    for issue, count in format_issue_counts.most_common():
        report(f"  {issue}: {count} ({count/total_responses_checked*100:.1f}%)")
    report()

    if repetition_examples:
        report("  Repetition examples:")
        for idx, detail in repetition_examples:
            report(f"    Problem {idx}: {detail}")
        report()

    # ============================================================
    # 3. Improved problems (step 200 right, step 50 wrong)
    # ============================================================
    report("## 3. Improved Problems (Step 50 wrong → Step 200 correct)")
    report()

    improved_topics = Counter()
    for idx in improved_idxs:
        topic = classify_topic(s50[idx]["question"])
        improved_topics[topic] += 1

    report(f"  {'Topic':<30s} {'Improved':>10s}")
    report(f"  {'-'*30} {'-'*10}")
    for topic in sorted(improved_topics.keys(), key=lambda t: improved_topics[t], reverse=True):
        report(f"  {topic:<30s} {improved_topics[topic]:>10d}")
    report()

    # ============================================================
    # 4. Response characteristics comparison (ALL problems)
    # ============================================================
    report("## 4. Response Characteristics: Step 50 vs Step 200 (All Problems)")
    report()

    for name, label, d in [("s50", "Step 50", s50), ("s100", "Step 100", s100),
                            ("s150", "Step 150", s150), ("s200", "Step 200", s200)]:
        if not d:
            continue
        all_lens = []
        all_word_counts = []
        all_similarities = []
        rep_count = 0
        total_resp = 0
        format_issues_all = Counter()

        for idx, item in d.items():
            sim = pairwise_similarity(item["responses"])
            all_similarities.append(sim)
            for r in item["responses"]:
                total_resp += 1
                l = response_length(r["response"])
                wc = response_word_count(r["response"])
                all_lens.append(l)
                all_word_counts.append(wc)
                is_rep, _ = detect_repetition(r["response"])
                if is_rep:
                    rep_count += 1
                for issue in detect_format_issues(r["response"]):
                    format_issues_all[issue] += 1

        report(f"  ### {label}")
        report(f"    Avg response length: {statistics.mean(all_lens):.0f} chars ({statistics.mean(all_word_counts):.0f} words)")
        report(f"    Median response length: {statistics.median(all_lens):.0f} chars")
        report(f"    Std dev response length: {statistics.stdev(all_lens):.0f} chars")
        report(f"    Max response length: {max(all_lens)} chars")
        report(f"    Avg pairwise similarity (Jaccard): {statistics.mean(all_similarities):.4f}")
        report(f"    Responses with repetition: {rep_count}/{total_resp} ({rep_count/total_resp*100:.1f}%)")
        for issue, count in format_issues_all.most_common(3):
            report(f"    {issue}: {count}/{total_resp} ({count/total_resp*100:.1f}%)")
        report()

    # ============================================================
    # 4b. Length distribution buckets
    # ============================================================
    report("### 4b. Response Length Distribution")
    report()

    buckets = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 10000), (10000, float('inf'))]

    report(f"  {'Bucket':<20s}", end="")
    for name, label, d in [("s50", "Step50", s50), ("s200", "Step200", s200)]:
        report(f" {label:>10s}", end="")
    report()

    for lo, hi in buckets:
        hi_label = f"{hi}" if hi != float('inf') else "∞"
        report(f"  {lo}-{hi_label:<16s}", end="")
        for name, label, d in [("s50", "Step50", s50), ("s200", "Step200", s200)]:
            count = 0
            total = 0
            for item in d.values():
                for r in item["responses"]:
                    total += 1
                    l = response_length(r["response"])
                    if lo <= l < hi:
                        count += 1
            report(f" {count:>10d}", end="")
        report()
    report()

    # ============================================================
    # 5. Comparison with pos-200tok
    # ============================================================
    if p200tok:
        report("## 5. Comparison: Pos-200tok Step 100 vs Full-Seq Step 200")
        report()

        p200tok_right_fullseq200_wrong = []
        fullseq200_right_p200tok_wrong = []

        for idx in p200tok:
            if idx not in s200:
                continue
            p_any = any(r["is_correct"] for r in p200tok[idx]["responses"])
            f_any = any(r["is_correct"] for r in s200[idx]["responses"])

            if p_any and not f_any:
                p200tok_right_fullseq200_wrong.append(idx)
            elif f_any and not p_any:
                fullseq200_right_p200tok_wrong.append(idx)

        report(f"  Pos-200tok correct, Full-seq step200 wrong: {len(p200tok_right_fullseq200_wrong)}")
        report(f"  Full-seq step200 correct, Pos-200tok wrong: {len(fullseq200_right_p200tok_wrong)}")
        report()

        # Topic distribution
        report("  Topics where pos-200tok succeeds but full-seq step200 fails:")
        topic_counts = Counter()
        for idx in p200tok_right_fullseq200_wrong:
            topic = classify_topic(s200[idx]["question"])
            topic_counts[topic] += 1
        for topic, count in topic_counts.most_common():
            report(f"    {topic}: {count}")
        report()

        # Length comparison on these problems
        if p200tok_right_fullseq200_wrong:
            p_lens = []
            f_lens = []
            for idx in p200tok_right_fullseq200_wrong:
                for r in p200tok[idx]["responses"]:
                    p_lens.append(response_length(r["response"]))
                for r in s200[idx]["responses"]:
                    f_lens.append(response_length(r["response"]))
            report(f"  On these problems:")
            report(f"    Pos-200tok avg response length: {statistics.mean(p_lens):.0f} chars")
            report(f"    Full-seq step200 avg response length: {statistics.mean(f_lens):.0f} chars")
            report(f"    Ratio: {statistics.mean(f_lens)/statistics.mean(p_lens):.2f}x")
            report()

    # ============================================================
    # 6. Step-by-step degradation tracking
    # ============================================================
    report("## 6. Step-by-Step Degradation Tracking")
    report()
    report("Track how individual problems change across steps 50→100→150→200:")
    report()

    # Count correct responses per problem at each step
    trajectories = defaultdict(list)  # idx -> [n_correct_50, n_correct_100, n_correct_150, n_correct_200]

    for idx in s50:
        traj = []
        for d in [s50, s100, s150, s200]:
            if idx in d:
                traj.append(avg_correct_count(d[idx]))
            else:
                traj.append(None)
        trajectories[idx] = traj

    # Classify trajectories
    monotonic_decrease = 0
    early_peak = 0  # peak at 50 or 100
    late_recovery = 0  # dips then recovers
    always_correct = 0
    always_wrong = 0

    for idx, traj in trajectories.items():
        if all(t is not None for t in traj):
            if all(t > 0 for t in traj):
                if traj[0] >= traj[1] >= traj[2] >= traj[3] and traj[0] > traj[3]:
                    monotonic_decrease += 1
            if all(t == 0 for t in traj):
                always_wrong += 1
            elif all(t == 4 for t in traj):
                always_correct += 1

    report(f"  Always all-correct (4/4 at every step): {always_correct}")
    report(f"  Always all-wrong (0/4 at every step): {always_wrong}")
    report(f"  Monotonically decreasing correct count: {monotonic_decrease}")
    report()

    # Average correct count at each step
    report("  Average correct count per problem:")
    for step_label, d in [("Step 50", s50), ("Step 100", s100), ("Step 150", s150), ("Step 200", s200)]:
        avg_c = statistics.mean(avg_correct_count(item) for item in d.values())
        report(f"    {step_label}: {avg_c:.3f} / 4")
    report()

    # ============================================================
    # 7. Deep dive: sample degraded problems
    # ============================================================
    report("## 7. Sample Degraded Problems (Step 50 → Step 200)")
    report()
    report("Showing up to 10 degraded problems with analysis:")
    report()

    # Sort degraded problems by how many step 50 got correct (most confident first)
    degraded_sorted = sorted(degraded_idxs, key=lambda idx: avg_correct_count(s50[idx]), reverse=True)

    for i, idx in enumerate(degraded_sorted[:10]):
        item50 = s50[idx]
        item200 = s200[idx]

        n50 = avg_correct_count(item50)
        n200 = avg_correct_count(item200)
        topic = classify_topic(item50["question"])

        s50_avg_len = statistics.mean(response_length(r["response"]) for r in item50["responses"])
        s200_avg_len = statistics.mean(response_length(r["response"]) for r in item200["responses"])

        report(f"  ### Problem {idx} [{topic}]")
        report(f"  Question: {item50['question'][:200]}...")
        report(f"  Ground truth: {item50['ground_truth']}")
        report(f"  Step 50: {n50}/4 correct, avg {s50_avg_len:.0f} chars")
        report(f"  Step 200: {n200}/4 correct, avg {s200_avg_len:.0f} chars (ratio: {s200_avg_len/s50_avg_len:.2f}x)")

        # Check step 200 answers
        s200_answers = [r["extracted_answer"] for r in item200["responses"]]
        report(f"  Step 200 extracted answers: {s200_answers}")

        # Check for repetition in step 200
        for j, r in enumerate(item200["responses"]):
            is_rep, detail = detect_repetition(r["response"])
            if is_rep:
                report(f"  Step 200 response {j}: REPETITIVE - {detail}")

        report()

    # ============================================================
    # 8. Answer extraction analysis
    # ============================================================
    report("## 8. Answer Extraction / Format Analysis")
    report()

    # Check if step 200 responses have boxed answers
    for step_label, d in [("Step 50", s50), ("Step 200", s200)]:
        no_boxed = 0
        total = 0
        none_answer = 0
        for item in d.values():
            for r in item["responses"]:
                total += 1
                if "\\boxed" not in r["response"]:
                    no_boxed += 1
                if r["extracted_answer"] is None or r["extracted_answer"] == "" or r["extracted_answer"] == "None":
                    none_answer += 1
        report(f"  {step_label}: {no_boxed}/{total} responses missing \\boxed ({no_boxed/total*100:.1f}%)")
        report(f"  {step_label}: {none_answer}/{total} responses with None/empty extracted answer ({none_answer/total*100:.1f}%)")
    report()

    # ============================================================
    # 9. Diversity analysis
    # ============================================================
    report("## 9. Response Diversity Analysis")
    report()

    for step_label, d in [("Step 50", s50), ("Step 200", s200)]:
        # Unique extracted answers per problem
        unique_answers = []
        for item in d.values():
            answers = set(str(r["extracted_answer"]) for r in item["responses"])
            unique_answers.append(len(answers))
        report(f"  {step_label}: avg unique answers per problem: {statistics.mean(unique_answers):.2f}")
        report(f"  {step_label}: problems with all same answer: {sum(1 for u in unique_answers if u == 1)}")
    report()

    # ============================================================
    # 10. Correct answer analysis on degraded problems
    # ============================================================
    report("## 10. Why Step 200 Gets Wrong Answers on Degraded Problems")
    report()

    # For degraded problems, compare: is the step 200 answer consistently wrong in the same way?
    consistent_wrong = 0
    diverse_wrong = 0

    for idx in degraded_idxs:
        answers = [str(r["extracted_answer"]) for r in s200[idx]["responses"]]
        unique = set(answers)
        if len(unique) == 1:
            consistent_wrong += 1
        else:
            diverse_wrong += 1

    report(f"  All 4 step-200 responses give same (wrong) answer: {consistent_wrong}/{len(degraded_idxs)} ({consistent_wrong/len(degraded_idxs)*100:.1f}%)")
    report(f"  Step-200 responses give diverse (all wrong) answers: {diverse_wrong}/{len(degraded_idxs)} ({diverse_wrong/len(degraded_idxs)*100:.1f}%)")
    report()

    # Check if the wrong answer is a common error type
    report("  Most common wrong answers on degraded problems (step 200):")
    wrong_answer_counter = Counter()
    for idx in degraded_idxs:
        for r in s200[idx]["responses"]:
            wrong_answer_counter[str(r["extracted_answer"])] += 1
    for ans, count in wrong_answer_counter.most_common(10):
        report(f"    '{ans}': {count} times")
    report()

    # ============================================================
    # 11. Summary / Hypothesis
    # ============================================================
    report("=" * 80)
    report("## Summary and Hypotheses")
    report("=" * 80)
    report()

    # Compute key stats for summary
    s50_avg_len = statistics.mean(response_length(r["response"]) for item in s50.values() for r in item["responses"])
    s200_avg_len = statistics.mean(response_length(r["response"]) for item in s200.values() for r in item["responses"])

    s50_rep = sum(1 for item in s50.values() for r in item["responses"] if detect_repetition(r["response"])[0])
    s200_rep = sum(1 for item in s200.values() for r in item["responses"] if detect_repetition(r["response"])[0])

    s50_sim = statistics.mean(pairwise_similarity(item["responses"]) for item in s50.values())
    s200_sim = statistics.mean(pairwise_similarity(item["responses"]) for item in s200.values())

    report(f"Key findings:")
    report(f"  1. {len(degraded_idxs)} problems degraded (step50→step200), only {len(improved_idxs)} improved")
    report(f"  2. Avg response length: step50={s50_avg_len:.0f} → step200={s200_avg_len:.0f} chars ({s200_avg_len/s50_avg_len:.2f}x)")
    report(f"  3. Repetitive responses: step50={s50_rep} → step200={s200_rep}")
    report(f"  4. Response similarity: step50={s50_sim:.4f} → step200={s200_sim:.4f}")
    report(f"  5. {consistent_wrong}/{len(degraded_idxs)} degraded problems have all 4 step-200 responses giving the SAME wrong answer")
    report()
    report("Hypotheses for degradation:")
    report("  H1: Full-seq KL loss on all tokens causes the model to over-fit to the teacher's")
    report("      reasoning style/verbosity, sacrificing mathematical accuracy for stylistic mimicry.")
    report("  H2: Training on later tokens (where the teacher is most 'opinionated') pushes the")
    report("      student away from its own valid reasoning patterns, causing mode collapse.")
    report("  H3: The loss on reasoning tokens overwhelms the loss on answer tokens, so the model")
    report("      learns to generate teacher-like reasoning but loses answer accuracy.")
    report("  H4: Pos-200tok works because early tokens (problem setup, approach selection) are")
    report("      where distillation is most beneficial; later tokens just add noise.")
    report()

    # Save report
    report_text = "\n".join(report_lines)
    OUTPUT_PATH.write_text(report_text)
    print(f"\nReport saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
