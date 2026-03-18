"""
Generation Behavior Analysis: Distilled Models vs Base Student Model

Analyzes:
1. Response opening similarity (prefix diversity)
2. Cascade effect (does early-token distillation improve later tokens?)
3. Response structure analysis (structural markers, reasoning patterns)
4. Correctness by response length
"""

import json
import re
import os
import sys
from collections import Counter, defaultdict
import statistics

# ============================================================
# Configuration: models to analyze
# ============================================================
BASE_DIR = "/CGLab/ziheng/projects/dft-distill"

MODELS = {
    "Base (Qwen2.5-Math-1.5B)": f"{BASE_DIR}/eval_results/raw-qwen2.5-math-1.5b/results.jsonl",
    "Pos-5 (step 200)": f"{BASE_DIR}/checkpoints/pos-limit-5tok/eval_step_200/results.jsonl",
    "Pos-10 (step 200)": f"{BASE_DIR}/checkpoints/pos-limit-10tok/eval_step_200/results.jsonl",
    "Pos-20 (step 200)": f"{BASE_DIR}/checkpoints/pos-limit-20tok/eval_step_200/results.jsonl",
    "Pos-50 (step 200)": f"{BASE_DIR}/checkpoints/positional-distill-50tok-v2/eval_step_200/results.jsonl",
    "Pos-200tok (step 100)": f"{BASE_DIR}/checkpoints/pos-limit-200tok/eval_step_100/results.jsonl",
    "Progressive 1->200 (step 200)": f"{BASE_DIR}/checkpoints/progressive-pos-1to200/eval_step_200/results.jsonl",
    "Full-seq (step 50)": f"{BASE_DIR}/checkpoints/full-seq-3584tok/eval_step_50/results.jsonl",
}

def load_results(path):
    """Load results.jsonl and return list of dicts."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_all_responses(results):
    """Flatten all responses from all questions."""
    all_resp = []
    for item in results:
        for r in item["responses"]:
            all_resp.append(r)
    return all_resp


def get_responses_with_meta(results):
    """Get responses paired with question metadata."""
    out = []
    for item in results:
        for r in item["responses"]:
            out.append({
                "idx": item["idx"],
                "question": item["question"],
                "response": r["response"],
                "is_correct": r["is_correct"],
            })
    return out


# ============================================================
# 1. Response Opening Similarity
# ============================================================
def analyze_prefix_diversity(all_data, prefix_lengths=[50, 100, 200, 500]):
    """
    For each model, look at the first N characters of responses for the same question.
    Measure how diverse the openings are (unique prefix ratio).
    """
    print("\n" + "="*80)
    print("1. RESPONSE OPENING SIMILARITY (PREFIX DIVERSITY)")
    print("="*80)

    output_lines = []
    output_lines.append("\n## 1. Response Opening Similarity (Prefix Diversity)")
    output_lines.append("")
    output_lines.append("For each question, we look at the first N characters of the 4 generated responses")
    output_lines.append("and count how many are unique. A ratio of 1.0 means all 4 responses start differently;")
    output_lines.append("lower values mean more convergent openings.")
    output_lines.append("")

    for plen in prefix_lengths:
        header = f"### Prefix length = {plen} chars"
        output_lines.append(header)
        output_lines.append("")
        output_lines.append(f"| Model | Avg Unique Prefixes (out of 4) | Unique Ratio |")
        output_lines.append(f"|-------|-------------------------------|-------------|")

        print(f"\n--- Prefix length = {plen} chars ---")
        for model_name, results in all_data.items():
            unique_counts = []
            for item in results:
                prefixes = set()
                for r in item["responses"]:
                    prefix = r["response"][:plen].strip()
                    prefixes.add(prefix)
                unique_counts.append(len(prefixes))
            avg_unique = statistics.mean(unique_counts)
            ratio = avg_unique / 4.0
            print(f"  {model_name:40s}: avg unique = {avg_unique:.2f}/4  ratio = {ratio:.3f}")
            output_lines.append(f"| {model_name} | {avg_unique:.2f} | {ratio:.3f} |")
        output_lines.append("")

    # Also: pairwise prefix overlap between models (compare first response of each question)
    output_lines.append("### Cross-model prefix similarity (first response per question, first 200 chars)")
    output_lines.append("")
    output_lines.append("Fraction of questions where Model A and Model B produce the same 200-char prefix:")
    output_lines.append("")

    model_names = list(all_data.keys())
    header = "| Model |"
    for mn in model_names:
        short = mn.split("(")[0].strip()
        header += f" {short} |"
    output_lines.append(header)
    output_lines.append("|" + "---|" * (len(model_names) + 1))

    print("\n--- Cross-model prefix overlap (200 chars, first response per question) ---")
    for i, mn_i in enumerate(model_names):
        row = f"| {mn_i.split('(')[0].strip()} |"
        for j, mn_j in enumerate(model_names):
            if i == j:
                row += " 1.000 |"
                continue
            matches = 0
            total = min(len(all_data[mn_i]), len(all_data[mn_j]))
            for k in range(total):
                p_i = all_data[mn_i][k]["responses"][0]["response"][:200].strip()
                p_j = all_data[mn_j][k]["responses"][0]["response"][:200].strip()
                if p_i == p_j:
                    matches += 1
            ratio = matches / total if total > 0 else 0
            row += f" {ratio:.3f} |"
        output_lines.append(row)
        print(f"  {mn_i}: overlaps = {[f'{m:.3f}' for m in [0]*len(model_names)]}")  # placeholder
    output_lines.append("")

    return "\n".join(output_lines)


# ============================================================
# 2. Cascade Effect Analysis
# ============================================================
def analyze_cascade_effect(all_data):
    """
    Compare base model vs distilled models:
    - Overall accuracy (avg@4, pass@4, maj@4)
    - Average response length
    - Key: for pos-50 model, the distillation only touched first 50 tokens,
      but does the ENTIRE response quality improve?
    """
    print("\n" + "="*80)
    print("2. CASCADE EFFECT ANALYSIS")
    print("="*80)

    output_lines = []
    output_lines.append("\n## 2. Cascade Effect: Does Early-Token Distillation Improve Later Tokens?")
    output_lines.append("")
    output_lines.append("The key question: Pos-50 only trains on the first 50 tokens of the response,")
    output_lines.append("but does the model produce better responses overall (including after position 50)?")
    output_lines.append("")

    # Compute metrics for each model
    output_lines.append("### Overall Metrics")
    output_lines.append("")
    output_lines.append("| Model | avg@4 | maj@4 | pass@4 | Avg Response Length (chars) | Median Resp Len |")
    output_lines.append("|-------|-------|-------|--------|---------------------------|----------------|")

    for model_name, results in all_data.items():
        correct_count = 0
        total_count = 0
        maj_correct = 0
        pass_correct = 0
        lengths = []

        for item in results:
            votes = []
            any_c = False
            for r in item["responses"]:
                total_count += 1
                if r["is_correct"]:
                    correct_count += 1
                    any_c = True
                votes.append(r["is_correct"])
                lengths.append(len(r["response"]))

            # Majority vote
            if sum(votes) > len(votes) / 2:
                maj_correct += 1
            # pass@4
            if any_c:
                pass_correct += 1

        n_questions = len(results)
        avg_acc = correct_count / total_count if total_count > 0 else 0
        maj_acc = maj_correct / n_questions if n_questions > 0 else 0
        pass_acc = pass_correct / n_questions if n_questions > 0 else 0
        avg_len = statistics.mean(lengths)
        med_len = statistics.median(lengths)

        line = f"| {model_name} | {avg_acc*100:.1f}% | {maj_acc*100:.1f}% | {pass_acc*100:.1f}% | {avg_len:.0f} | {med_len:.0f} |"
        output_lines.append(line)
        print(f"  {model_name:40s}: avg@4={avg_acc*100:.1f}%, maj@4={maj_acc*100:.1f}%, pass@4={pass_acc*100:.1f}%, avg_len={avg_len:.0f}, med_len={med_len:.0f}")

    output_lines.append("")

    # Per-question comparison: base vs pos-50
    output_lines.append("### Per-Question Accuracy Shift: Base -> Pos-50")
    output_lines.append("")

    base_results = all_data.get("Base (Qwen2.5-Math-1.5B)")
    pos50_results = all_data.get("Pos-50 (step 200)")

    if base_results and pos50_results:
        improved = 0
        degraded = 0
        unchanged = 0
        total = min(len(base_results), len(pos50_results))

        for i in range(total):
            base_correct = sum(1 for r in base_results[i]["responses"] if r["is_correct"])
            pos50_correct = sum(1 for r in pos50_results[i]["responses"] if r["is_correct"])
            if pos50_correct > base_correct:
                improved += 1
            elif pos50_correct < base_correct:
                degraded += 1
            else:
                unchanged += 1

        output_lines.append(f"- Questions improved (more correct samples): {improved}/{total} ({improved/total*100:.1f}%)")
        output_lines.append(f"- Questions degraded (fewer correct samples): {degraded}/{total} ({degraded/total*100:.1f}%)")
        output_lines.append(f"- Questions unchanged: {unchanged}/{total} ({unchanged/total*100:.1f}%)")
        output_lines.append(f"- Net improvement: {improved - degraded} questions")
        print(f"  Base->Pos50: improved={improved}, degraded={degraded}, unchanged={unchanged}")

    output_lines.append("")

    # Response length distribution comparison
    output_lines.append("### Response Length Distribution")
    output_lines.append("")
    output_lines.append("| Model | Mean | Std | P10 | P25 | P50 | P75 | P90 |")
    output_lines.append("|-------|------|-----|-----|-----|-----|-----|-----|")

    for model_name, results in all_data.items():
        lengths = []
        for item in results:
            for r in item["responses"]:
                lengths.append(len(r["response"]))
        lengths.sort()
        n = len(lengths)
        mean_l = statistics.mean(lengths)
        std_l = statistics.stdev(lengths) if n > 1 else 0
        p10 = lengths[int(n*0.1)]
        p25 = lengths[int(n*0.25)]
        p50 = lengths[int(n*0.5)]
        p75 = lengths[int(n*0.75)]
        p90 = lengths[int(n*0.9)]
        output_lines.append(f"| {model_name} | {mean_l:.0f} | {std_l:.0f} | {p10} | {p25} | {p50} | {p75} | {p90} |")

    output_lines.append("")
    return "\n".join(output_lines)


# ============================================================
# 3. Response Structure Analysis
# ============================================================
def analyze_response_structure(all_data):
    """
    Analyze structural markers in responses:
    - Step/numbered markers
    - Structural keywords
    - Newline frequency
    - Code block usage
    - boxed answer usage
    """
    print("\n" + "="*80)
    print("3. RESPONSE STRUCTURE ANALYSIS")
    print("="*80)

    output_lines = []
    output_lines.append("\n## 3. Response Structure Analysis")
    output_lines.append("")
    output_lines.append("Do distilled models produce more structured, teacher-like responses?")
    output_lines.append("")

    # Define structural markers
    markers = {
        "Step N:": r"(?i)step\s+\d+",
        "### heading": r"^#{1,4}\s+\w",
        "\\boxed{}": r"\\boxed\{",
        "```code```": r"```",
        "First,/Second,": r"(?i)\b(first|second|third|fourth|fifth|next|then|finally)\b[,:]",
        "Therefore/Thus": r"(?i)\b(therefore|thus|hence|so we have|we conclude)\b",
        "Let's/We": r"(?i)^(let'?s|we (can|will|need|know|have|are|use|see|get|note))",
        "**bold**": r"\*\*[^*]+\*\*",
        "Verification": r"(?i)(verif|check|confirm|let'?s verify)",
    }

    output_lines.append("### Structural Marker Frequency (avg per response)")
    output_lines.append("")
    header = "| Model |"
    for marker_name in markers:
        header += f" {marker_name} |"
    header += " Newlines | Avg Steps |"
    output_lines.append(header)
    output_lines.append("|" + "---|" * (len(markers) + 3))

    for model_name, results in all_data.items():
        all_responses = get_all_responses(results)
        n = len(all_responses)

        counts = {}
        for marker_name, pattern in markers.items():
            total = 0
            for r in all_responses:
                total += len(re.findall(pattern, r["response"], re.MULTILINE))
            counts[marker_name] = total / n

        # Newlines per response
        total_newlines = sum(r["response"].count("\n") for r in all_responses)
        avg_newlines = total_newlines / n

        # Count "steps" (numbered items like "Step 1", "1.", "1)", etc.)
        total_steps = 0
        for r in all_responses:
            steps = re.findall(r"(?i)(?:step\s+\d+|^\d+[\.\)]\s)", r["response"], re.MULTILINE)
            total_steps += len(steps)
        avg_steps = total_steps / n

        row = f"| {model_name} |"
        for marker_name in markers:
            row += f" {counts[marker_name]:.2f} |"
        row += f" {avg_newlines:.1f} | {avg_steps:.1f} |"
        output_lines.append(row)

        print(f"  {model_name:40s}: steps={avg_steps:.1f}, headings={counts['### heading']:.2f}, boxed={counts[chr(92)+'boxed{}']:.2f}, newlines={avg_newlines:.1f}")

    output_lines.append("")

    # Response opening patterns
    output_lines.append("### Most Common Response Openings (first 80 chars)")
    output_lines.append("")

    for model_name, results in all_data.items():
        all_responses = get_all_responses(results)
        openings = Counter()
        for r in all_responses:
            # Get first line
            first_line = r["response"].strip().split("\n")[0][:80]
            openings[first_line] += 1

        output_lines.append(f"**{model_name}** - Top 5 openings:")
        output_lines.append("")
        for opening, count in openings.most_common(5):
            pct = count / len(all_responses) * 100
            output_lines.append(f"- ({count}, {pct:.1f}%) `{opening}`")
        output_lines.append("")

    return "\n".join(output_lines)


# ============================================================
# 4. Correctness by Response Length
# ============================================================
def analyze_correctness_by_length(all_data):
    """
    Bin responses by length and compute accuracy per bin.
    """
    print("\n" + "="*80)
    print("4. CORRECTNESS BY RESPONSE LENGTH")
    print("="*80)

    output_lines = []
    output_lines.append("\n## 4. Correctness by Response Length")
    output_lines.append("")
    output_lines.append("Responses binned by character length. Accuracy computed per bin.")
    output_lines.append("")

    # Define bins
    bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000), (3000, 5000), (5000, 10000), (10000, float('inf'))]
    bin_labels = ["0-500", "500-1K", "1K-1.5K", "1.5K-2K", "2K-3K", "3K-5K", "5K-10K", "10K+"]

    header = "| Model |"
    for label in bin_labels:
        header += f" {label} |"
    output_lines.append(header)
    output_lines.append("|" + "---|" * (len(bin_labels) + 1))

    for model_name, results in all_data.items():
        all_responses = get_all_responses(results)

        bin_correct = defaultdict(int)
        bin_total = defaultdict(int)

        for r in all_responses:
            length = len(r["response"])
            for i, (lo, hi) in enumerate(bins):
                if lo <= length < hi:
                    bin_total[i] += 1
                    if r["is_correct"]:
                        bin_correct[i] += 1
                    break

        row = f"| {model_name} |"
        for i in range(len(bins)):
            if bin_total[i] > 0:
                acc = bin_correct[i] / bin_total[i]
                row += f" {acc*100:.0f}% ({bin_total[i]}) |"
            else:
                row += " - |"
        output_lines.append(row)

        print(f"  {model_name}")
        for i in range(len(bins)):
            if bin_total[i] > 0:
                acc = bin_correct[i] / bin_total[i]
                print(f"    {bin_labels[i]:10s}: {acc*100:.1f}% (n={bin_total[i]})")

    output_lines.append("")

    # Shorter = better? Correlation between length and correctness
    output_lines.append("### Correct vs Incorrect Response Length")
    output_lines.append("")
    output_lines.append("| Model | Avg Len (Correct) | Avg Len (Incorrect) | Ratio (Correct/Incorrect) |")
    output_lines.append("|-------|-------------------|--------------------|--------------------------| ")

    for model_name, results in all_data.items():
        all_responses = get_all_responses(results)
        correct_lens = [len(r["response"]) for r in all_responses if r["is_correct"]]
        incorrect_lens = [len(r["response"]) for r in all_responses if not r["is_correct"]]

        if correct_lens and incorrect_lens:
            avg_c = statistics.mean(correct_lens)
            avg_i = statistics.mean(incorrect_lens)
            ratio = avg_c / avg_i
            output_lines.append(f"| {model_name} | {avg_c:.0f} | {avg_i:.0f} | {ratio:.2f} |")
            print(f"  {model_name:40s}: correct_len={avg_c:.0f}, incorrect_len={avg_i:.0f}, ratio={ratio:.2f}")
        else:
            output_lines.append(f"| {model_name} | N/A | N/A | N/A |")

    output_lines.append("")
    return "\n".join(output_lines)


# ============================================================
# 5. Token-level cascade analysis (using word approximation)
# ============================================================
def analyze_token_cascade(all_data):
    """
    For pos-50 model: analyze whether tokens AFTER position ~50 (word approx) differ from base.
    Use word-level approximation since we don't have tokenizer.
    """
    print("\n" + "="*80)
    print("5. EARLY vs LATE RESPONSE QUALITY (CASCADE ANALYSIS)")
    print("="*80)

    output_lines = []
    output_lines.append("\n## 5. Early vs Late Response Quality (Token Cascade)")
    output_lines.append("")
    output_lines.append("We approximate token positions using words (roughly 1 word ~ 1.3 tokens for English/math).")
    output_lines.append("For Pos-50, distillation only affected the first ~50 tokens (~38 words).")
    output_lines.append("We analyze: does the LATER part of the response also change?")
    output_lines.append("")

    base_results = all_data.get("Base (Qwen2.5-Math-1.5B)")

    # Compare base vs each distilled model: similarity in first N words vs later words
    word_cutoffs = [40, 80, 150]  # approximate 50, 100, 200 tokens

    for cutoff in word_cutoffs:
        output_lines.append(f"### Word cutoff = {cutoff} (approx {int(cutoff*1.3)} tokens)")
        output_lines.append("")
        output_lines.append("| Model | Avg Jaccard (first {0} words) | Avg Jaccard (after {0} words) | Early Change > Late Change? |".format(cutoff))
        output_lines.append("|" + "---|" * 4)

        if base_results is None:
            output_lines.append("| (no base model data) | - | - | - |")
            continue

        for model_name, results in all_data.items():
            if model_name == "Base (Qwen2.5-Math-1.5B)":
                continue

            early_sims = []
            late_sims = []
            total = min(len(base_results), len(results))

            for i in range(total):
                # Compare first response only
                base_words = base_results[i]["responses"][0]["response"].split()
                dist_words = results[i]["responses"][0]["response"].split()

                base_early = set(base_words[:cutoff])
                dist_early = set(dist_words[:cutoff])
                base_late = set(base_words[cutoff:])
                dist_late = set(dist_words[cutoff:])

                if base_early and dist_early:
                    jaccard_early = len(base_early & dist_early) / len(base_early | dist_early)
                    early_sims.append(jaccard_early)

                if base_late and dist_late:
                    jaccard_late = len(base_late & dist_late) / len(base_late | dist_late)
                    late_sims.append(jaccard_late)

            avg_early = statistics.mean(early_sims) if early_sims else 0
            avg_late = statistics.mean(late_sims) if late_sims else 0
            cascade = "Yes" if avg_early < avg_late else "No (late also changed)"
            # Lower jaccard = more different

            output_lines.append(f"| {model_name} | {avg_early:.3f} | {avg_late:.3f} | {cascade} |")
            print(f"  {model_name:40s} cutoff={cutoff}: early_jaccard={avg_early:.3f}, late_jaccard={avg_late:.3f}")

        output_lines.append("")

    output_lines.append("Note: Lower Jaccard similarity = more different from base model.")
    output_lines.append("If 'late' Jaccard is also low, it means distillation cascaded to affect later tokens too.")
    output_lines.append("")

    return "\n".join(output_lines)


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data...")
    all_data = {}
    for model_name, path in MODELS.items():
        if os.path.exists(path):
            all_data[model_name] = load_results(path)
            print(f"  Loaded {model_name}: {len(all_data[model_name])} questions, {len(all_data[model_name][0]['responses'])} responses each")
        else:
            print(f"  WARNING: {path} not found, skipping {model_name}")

    # Run all analyses
    sections = []
    sections.append("# Generation Behavior Analysis: Distilled vs Base Student Model")
    sections.append("")
    sections.append("Date: 2026-03-11")
    sections.append("")
    sections.append("**Goal**: Analyze whether distilling only the first N tokens causes behavioral changes")
    sections.append("BEYOND position N (the 'cascade effect').")
    sections.append("")
    sections.append("**Models compared**:")
    for model_name in all_data:
        sections.append(f"- {model_name}")
    sections.append("")

    sections.append(analyze_prefix_diversity(all_data))
    sections.append(analyze_cascade_effect(all_data))
    sections.append(analyze_response_structure(all_data))
    sections.append(analyze_correctness_by_length(all_data))
    sections.append(analyze_token_cascade(all_data))

    # Write report
    report = "\n".join(sections)
    output_path = f"{BASE_DIR}/docs/generation_behavior_analysis.md"
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n{'='*80}")
    print(f"Report saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
