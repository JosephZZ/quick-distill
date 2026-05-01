"""
Format-mask threshold analysis.

Goal: decide how aggressively to define "format token" before training a
format-mask ablation.

For each student-emitted token, classify into a category, compute its KL
contribution (student_lp - teacher_lp clamped to >=0). Then report:
  - per-category token count and KL share
  - cumulative coverage as categories are added (in fixed "format-aggressive"
    order: structural -> math_latex -> math_operator -> math_number ->
    planning -> continuation)

Use this to pick a mask definition such that masking removes a target
fraction of total KL (e.g. 70-90%).
"""

import json
import sys
import os
import re
from collections import defaultdict, Counter
from transformers import AutoTokenizer

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else \
    "/zhi_backup/ziheng/quick-distillation/docs/kl_position_analysis_v2/raw_logprobs.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/zhi_backup/ziheng/quick-distillation/docs/format_mask_threshold.md"
TOK = "Qwen/Qwen2.5-Math-1.5B"
MAX_TRAJ = 5000

PLANNING = {
    "to","let","we","first","the","since","note","recall","now","next","then",
    "so","thus","hence","therefore","given","consider","suppose","assume",
    "because","if","for","by","from","using","applying","substituting",
    "simplifying","solving","calculating","computing","evaluating","finally",
    "step","answer","solution",
}
MATH_OPS = {"+","-","*","/","=","<",">","^","_","≤","≥","≠","±","×","÷"}
LATEX_HINTS = ("frac","sqrt","sum","int","lim","boxed","text","cdot","times")

def classify(s):
    st = s.strip()
    if st.lower() in PLANNING:
        return "planning"
    if s in ("\n","\r\n","\r") or st == "":
        return "structural"
    if st in ("**","##","#","---","```",":",";",",",".","!","?","(",")","[","]","{","}"):
        return "structural"
    if st.startswith("**") or st.startswith("##"):
        return "structural"
    if st.startswith("\\") and len(st) > 1:
        return "math_latex"
    if st in MATH_OPS:
        return "math_operator"
    if st.replace(".","").replace(",","").isdigit():
        return "math_number"
    if any(x in st for x in LATEX_HINTS):
        return "math_latex"
    if re.match(r'^-?\d+\.?\d*$', st):
        return "math_number"
    if len(st) <= 2 and not st.isalnum():
        return "structural"
    return "continuation"

def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    cat_count = Counter()
    cat_kl    = defaultdict(float)
    total_tok = 0
    total_kl  = 0.0

    n = 0
    with open(DATA_FILE) as f:
        for line in f:
            if n >= MAX_TRAJ: break
            d = json.loads(line)
            ids  = d["response_ids"]
            slps = d["student_lps"]
            tlps = d["teacher_lps"]
            L = min(len(ids), len(slps), len(tlps))
            for i in range(L):
                # |s_lp - t_lp| matches the metric used in the original
                # token_classification_analysis.py / Slide 10 numbers.
                kl = abs(slps[i] - tlps[i])
                tstr = tok.decode([ids[i]])
                c = classify(tstr)
                cat_count[c] += 1
                cat_kl[c]    += kl
                total_tok    += 1
                total_kl     += kl
            n += 1

    lines = []
    def w(s=""): lines.append(s); print(s)

    w(f"# Format-Mask Threshold Analysis")
    w()
    w(f"Data: `{DATA_FILE}`")
    w(f"Trajectories: {n}, total tokens: {total_tok}, total KL: {total_kl:.1f}")
    w()
    w("## Per-category share")
    w()
    w("| Category | n_tokens | tok % | KL sum | KL % | mean KL/tok |")
    w("|---|---:|---:|---:|---:|---:|")
    cats = sorted(cat_count.keys(), key=lambda c: -cat_kl[c])
    for c in cats:
        nT = cat_count[c]
        kl = cat_kl[c]
        w(f"| {c} | {nT} | {nT/total_tok*100:.1f}% | {kl:.1f} | {kl/total_kl*100:.1f}% | {kl/max(nT,1):.3f} |")
    w()

    # Cumulative as we add categories in "format-aggressive" order
    order = ["structural","math_latex","math_operator","math_number","planning","continuation"]
    w("## Cumulative mask coverage (add categories in this order)")
    w()
    w("| Mask = | Masked tokens | Masked tok % | Masked KL | Masked KL % | Remaining KL % |")
    w("|---|---:|---:|---:|---:|---:|")
    cum_n, cum_kl = 0, 0.0
    masked = []
    for c in order:
        if c not in cat_count: continue
        cum_n  += cat_count[c]
        cum_kl += cat_kl[c]
        masked.append(c)
        w(f"| {{{', '.join(masked)}}} | {cum_n} | {cum_n/total_tok*100:.1f}% | {cum_kl:.1f} | {cum_kl/total_kl*100:.1f}% | {(total_kl-cum_kl)/total_kl*100:.1f}% |")
    w()

    w("## Recommendation guides")
    w()
    w("- Aggressive mask (structural+latex+ops+numbers): captures most format-y tokens but may over-remove signal")
    w("- Conservative mask (structural+latex only): keeps numerical/operator content")
    w("- Pick the row whose masked-KL % matches the fraction of signal the user wants removed.")
    w()
    w("If the prefix-vs-fullseq gap is mostly about format noise, a mask that")
    w("removes ~50-80% of KL but leaves reasoning content should *match or exceed*")
    w("fullseq performance. If it does not, format-noise is not the operative variable.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {OUT}")

if __name__ == "__main__":
    main()
