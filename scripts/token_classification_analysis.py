"""
Token-level classification analysis of KL divergence in on-policy distillation trajectories.
Analyzes WHAT tokens have high KL divergence, not just WHERE.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from transformers import AutoTokenizer
import re
import os

# --- Config ---
DATA_FILE = "/CGLab/ziheng/projects/dft-distill/output/qwen3-1.7B-logprobs.jsonl"
OUTPUT_FILE = "/CGLab/ziheng/projects/dft-distill/docs/token_classification_analysis.md"
MAX_TRAJECTORIES = 10000
TOKENIZER_NAME = "Qwen/Qwen2.5-Math-1.5B"

# --- Load data ---
print("Loading data...")
data = []
with open(DATA_FILE, "r") as f:
    for i, line in enumerate(f):
        if i >= MAX_TRAJECTORIES:
            break
        data.append(json.loads(line))
print(f"Loaded {len(data)} trajectories")

# --- Load tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# --- Token classification ---
def classify_token(token_str):
    """Classify a decoded token into categories."""
    stripped = token_str.strip()

    # Planning/reasoning starters
    planning_words = {
        "to", "let", "we", "first", "the", "since", "note", "recall",
        "now", "next", "then", "so", "thus", "hence", "therefore",
        "given", "consider", "suppose", "assume", "because", "if",
        "for", "by", "from", "using", "applying", "substituting",
        "simplifying", "solving", "calculating", "computing", "evaluating",
        "finally", "step", "answer", "solution",
    }
    if stripped.lower() in planning_words:
        return "planning"

    # Structural tokens
    if token_str in ("\n", "\r\n", "\r") or token_str.strip() == "":
        if "\n" in token_str or token_str.strip() == "":
            return "structural"
    if stripped in ("**", "##", "#", "---", "```", ":", ";", ",", ".", "!", "?", "(", ")", "[", "]", "{", "}"):
        return "structural"
    if stripped.startswith("**") or stripped.startswith("##"):
        return "structural"

    # Math/LaTeX tokens
    if stripped.startswith("\\") and len(stripped) > 1:
        return "math_latex"
    math_operators = {"+", "-", "*", "/", "=", "<", ">", "^", "_", "≤", "≥", "≠", "±", "×", "÷"}
    if stripped in math_operators:
        return "math_operator"
    if stripped.replace(".", "").replace(",", "").isdigit():
        return "math_number"
    # LaTeX environment markers
    if any(x in stripped for x in ["frac", "sqrt", "sum", "int", "lim", "boxed", "text", "cdot", "times"]):
        return "math_latex"

    # Check if it's a number embedded in text
    if re.match(r'^-?\d+\.?\d*$', stripped):
        return "math_number"

    # Punctuation that wasn't caught above
    if len(stripped) <= 2 and not stripped.isalnum():
        return "structural"

    # Everything else is "continuation" (common words continuing sentences)
    return "continuation"


# --- Collect per-position data ---
print("Analyzing tokens...")

# For each position, collect: token_ids, decoded tokens, KL values
max_pos = 500  # analyze up to position 500
position_data = defaultdict(lambda: {"tokens": [], "token_strs": [], "kl_values": [], "categories": []})
token_kl_map = defaultdict(list)  # token_str -> list of KL values
all_category_kl = defaultdict(list)  # category -> list of (position, kl)

for item in data:
    token_ids = item["response_token_ids"]
    student_lp = item["response_log_probs"]
    teacher_lp = item["teacher_log_probs"]

    for pos in range(min(len(token_ids), max_pos)):
        tid = token_ids[pos]
        s_lp = student_lp[pos]
        t_lp = teacher_lp[pos]

        # KL divergence = |student_logprob - teacher_logprob|
        kl = abs(s_lp - t_lp)

        # Decode token
        token_str = tokenizer.decode([tid])
        category = classify_token(token_str)

        position_data[pos]["tokens"].append(tid)
        position_data[pos]["token_strs"].append(token_str)
        position_data[pos]["kl_values"].append(kl)
        position_data[pos]["categories"].append(category)

        token_kl_map[token_str].append(kl)
        all_category_kl[category].append((pos, kl))

print(f"Analyzed positions 0-{max_pos-1}")

# --- Build report ---
report_lines = []
def report(s=""):
    report_lines.append(s)
    print(s)

report("# Token-Level KL Divergence Classification Analysis")
report()
report(f"Data: `{DATA_FILE}`")
report(f"Trajectories analyzed: {len(data)}")
report(f"Tokenizer: {TOKENIZER_NAME}")
report()

# ========== 1. Position 0 Deep Dive ==========
report("## 1. Position 0 Deep Dive")
report()
report("Position 0 has KL=8.2 which is ~10x higher than other positions. What tokens appear here?")
report()

pos0 = position_data[0]
if pos0["tokens"]:
    # Count tokens at position 0
    token_counter = Counter(pos0["token_strs"])
    token_kl_at_pos0 = defaultdict(list)
    for ts, kl in zip(pos0["token_strs"], pos0["kl_values"]):
        token_kl_at_pos0[ts].append(kl)

    report(f"Total trajectories with position 0: {len(pos0['tokens'])}")
    report(f"Mean KL at position 0: {np.mean(pos0['kl_values']):.4f}")
    report(f"Median KL at position 0: {np.median(pos0['kl_values']):.4f}")
    report(f"Unique tokens at position 0: {len(token_counter)}")
    report()

    report("### Top 30 tokens at position 0 (by frequency)")
    report()
    report("| Token | Count | % | Mean KL | Median KL | Category |")
    report("|-------|-------|---|---------|-----------|----------|")
    for token_str, count in token_counter.most_common(30):
        pct = count / len(pos0["tokens"]) * 100
        mean_kl = np.mean(token_kl_at_pos0[token_str])
        med_kl = np.median(token_kl_at_pos0[token_str])
        cat = classify_token(token_str)
        # Escape for markdown
        display = repr(token_str)
        report(f"| {display} | {count} | {pct:.1f}% | {mean_kl:.2f} | {med_kl:.2f} | {cat} |")

    report()

    # Category breakdown at position 0
    cat_counter_pos0 = Counter(pos0["categories"])
    report("### Category breakdown at position 0")
    report()
    report("| Category | Count | % | Mean KL |")
    report("|----------|-------|---|---------|")
    for cat, count in cat_counter_pos0.most_common():
        pct = count / len(pos0["categories"]) * 100
        cat_kls = [kl for c, kl in zip(pos0["categories"], pos0["kl_values"]) if c == cat]
        report(f"| {cat} | {count} | {pct:.1f}% | {np.mean(cat_kls):.2f} |")

report()

# ========== 2. Token Content at First 50 Positions ==========
report("## 2. Token Content at High-KL Positions (0-49)")
report()
report("What tokens appear at each of the first 20 positions?")
report()

for pos in range(20):
    pd = position_data[pos]
    if not pd["tokens"]:
        continue
    tc = Counter(pd["token_strs"])
    mean_kl = np.mean(pd["kl_values"])
    top5 = tc.most_common(5)
    top5_str = ", ".join([f"`{repr(t)}` ({c}/{len(pd['tokens'])})" for t, c in top5])
    report(f"**Position {pos}** (mean KL={mean_kl:.2f}, n={len(pd['tokens'])}): {top5_str}")

report()

# ========== 3. Token Type Classification by Position Range ==========
report("## 3. Token Type Distribution by Position Range")
report()

ranges = [(0, 5), (5, 20), (20, 50), (50, 100), (100, 200), (200, 500)]

report("| Position Range | n_tokens | planning % | structural % | math_number % | math_operator % | math_latex % | continuation % | Mean KL |")
report("|----------------|----------|------------|--------------|---------------|-----------------|--------------|----------------|---------|")

for start, end in ranges:
    cats = []
    kls = []
    for pos in range(start, end):
        pd = position_data[pos]
        cats.extend(pd["categories"])
        kls.extend(pd["kl_values"])

    if not cats:
        continue

    total = len(cats)
    cat_counts = Counter(cats)
    mean_kl = np.mean(kls)

    planning_pct = cat_counts.get("planning", 0) / total * 100
    structural_pct = cat_counts.get("structural", 0) / total * 100
    number_pct = cat_counts.get("math_number", 0) / total * 100
    operator_pct = cat_counts.get("math_operator", 0) / total * 100
    latex_pct = cat_counts.get("math_latex", 0) / total * 100
    cont_pct = cat_counts.get("continuation", 0) / total * 100

    report(f"| {start}-{end} | {total} | {planning_pct:.1f}% | {structural_pct:.1f}% | {number_pct:.1f}% | {operator_pct:.1f}% | {latex_pct:.1f}% | {cont_pct:.1f}% | {mean_kl:.2f} |")

report()

# Mean KL by category across all positions
report("### Mean KL by Category (all positions)")
report()
report("| Category | n_tokens | Mean KL | Median KL | Std KL |")
report("|----------|----------|---------|-----------|--------|")

for cat in ["planning", "structural", "math_number", "math_operator", "math_latex", "continuation"]:
    kls = [kl for _, kl in all_category_kl.get(cat, [])]
    if kls:
        report(f"| {cat} | {len(kls)} | {np.mean(kls):.3f} | {np.median(kls):.3f} | {np.std(kls):.3f} |")

report()

# Mean KL by category, broken down by position range
report("### Mean KL by Category and Position Range")
report()
report("| Category | 0-4 | 5-19 | 20-49 | 50-99 | 100-199 | 200-499 |")
report("|----------|-----|------|-------|-------|---------|---------|")

for cat in ["planning", "structural", "math_number", "math_operator", "math_latex", "continuation"]:
    row = f"| {cat}"
    for start, end in ranges:
        kls = []
        for pos in range(start, end):
            pd = position_data[pos]
            for c, kl in zip(pd["categories"], pd["kl_values"]):
                if c == cat:
                    kls.append(kl)
        if kls:
            row += f" | {np.mean(kls):.2f}"
        else:
            row += " | -"
    row += " |"
    report(row)

report()

# ========== 4. High-KL Token Analysis ==========
report("## 4. High-KL Tokens (Top Tokens by Mean KL)")
report()
report("Tokens appearing at least 50 times, ranked by mean KL divergence:")
report()

# Filter tokens with enough occurrences
token_stats = []
for token_str, kls in token_kl_map.items():
    if len(kls) >= 50:
        token_stats.append({
            "token": token_str,
            "count": len(kls),
            "mean_kl": np.mean(kls),
            "median_kl": np.median(kls),
            "category": classify_token(token_str),
        })

token_stats.sort(key=lambda x: x["mean_kl"], reverse=True)

report("### Top 40 highest-KL tokens")
report()
report("| Rank | Token | Category | Count | Mean KL | Median KL |")
report("|------|-------|----------|-------|---------|-----------|")
for i, ts in enumerate(token_stats[:40]):
    display = repr(ts["token"])
    report(f"| {i+1} | {display} | {ts['category']} | {ts['count']} | {ts['mean_kl']:.3f} | {ts['median_kl']:.3f} |")

report()

report("### Top 40 lowest-KL tokens (teacher-student agreement)")
report()
report("| Rank | Token | Category | Count | Mean KL | Median KL |")
report("|------|-------|----------|-------|---------|-----------|")
token_stats_low = sorted(token_stats, key=lambda x: x["mean_kl"])
for i, ts in enumerate(token_stats_low[:40]):
    display = repr(ts["token"])
    report(f"| {i+1} | {display} | {ts['category']} | {ts['count']} | {ts['mean_kl']:.3f} | {ts['median_kl']:.3f} |")

report()

# ========== 5. Category-specific top tokens ==========
report("## 5. Highest-KL Tokens by Category")
report()

for cat in ["planning", "structural", "math_number", "math_operator", "math_latex", "continuation"]:
    cat_tokens = [ts for ts in token_stats if ts["category"] == cat]
    cat_tokens.sort(key=lambda x: x["mean_kl"], reverse=True)

    report(f"### {cat} (top 15)")
    report()
    report("| Token | Count | Mean KL | Median KL |")
    report("|-------|-------|---------|-----------|")
    for ts in cat_tokens[:15]:
        display = repr(ts["token"])
        report(f"| {display} | {ts['count']} | {ts['mean_kl']:.3f} | {ts['median_kl']:.3f} |")
    report()

# ========== 6. Position 1-4 token distributions ==========
report("## 6. Positions 1-4 Token Distributions")
report()

for pos in range(1, 5):
    pd = position_data[pos]
    if not pd["tokens"]:
        continue

    tc = Counter(pd["token_strs"])
    token_kl_at_pos = defaultdict(list)
    for ts, kl in zip(pd["token_strs"], pd["kl_values"]):
        token_kl_at_pos[ts].append(kl)

    report(f"### Position {pos} (mean KL={np.mean(pd['kl_values']):.2f}, n={len(pd['tokens'])})")
    report()
    report("| Token | Count | % | Mean KL | Category |")
    report("|-------|-------|---|---------|----------|")
    for token_str, count in tc.most_common(20):
        pct = count / len(pd["tokens"]) * 100
        mean_kl = np.mean(token_kl_at_pos[token_str])
        cat = classify_token(token_str)
        display = repr(token_str)
        report(f"| {display} | {count} | {pct:.1f}% | {mean_kl:.2f} | {cat} |")
    report()

# ========== 7. Summary Statistics ==========
report("## 7. Summary Statistics")
report()

# Sequence length stats
seq_lens = [len(item["response_token_ids"]) for item in data]
report(f"- Mean sequence length: {np.mean(seq_lens):.1f}")
report(f"- Median sequence length: {np.median(seq_lens):.1f}")
report(f"- Min/Max sequence length: {min(seq_lens)}/{max(seq_lens)}")
report()

# Overall KL stats
all_kls = []
for pos in range(max_pos):
    all_kls.extend(position_data[pos]["kl_values"])
report(f"- Total tokens analyzed: {len(all_kls)}")
report(f"- Overall mean KL: {np.mean(all_kls):.4f}")
report(f"- Overall median KL: {np.median(all_kls):.4f}")
report()

# Fraction of KL explained by first N positions
total_kl = sum(all_kls)
for n in [1, 5, 10, 20, 50]:
    first_n_kl = sum(kl for pos in range(n) for kl in position_data[pos]["kl_values"])
    report(f"- KL in first {n} positions: {first_n_kl/total_kl*100:.1f}% of total")

report()
report("## 8. Key Findings")
report()
report("(Filled in after analysis)")

# --- Save report ---
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nReport saved to {OUTPUT_FILE}")
