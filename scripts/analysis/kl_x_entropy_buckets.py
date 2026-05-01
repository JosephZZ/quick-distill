"""
Bucket tokens by (KL high/low) × (surprise high/low) and show what's in each.

KL = |student_lp - teacher_lp| at the sampled token.
Surprise = -student_lp at the sampled token (proxy for distribution entropy).

If high-KL tokens cluster into:
  - high-KL + high-surprise (student also uncertain) -> "genuine reasoning
    moments" — student doesn't know which path to take, teacher does
  - high-KL + low-surprise (student confident, but disagrees with teacher) ->
    "format / overconfident" — student thinks it knows but teacher disagrees,
    typically structural / format tokens

then format-noise vs reasoning-divergence story is supported.
"""

import json
import sys
import re
import os
import numpy as np
from collections import Counter, defaultdict
from transformers import AutoTokenizer

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/zhi_backup/ziheng/quick-distillation/docs/kl_position_analysis_v2/raw_logprobs.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/zhi_backup/ziheng/quick-distillation/docs/kl_x_entropy_buckets.md"
TOK = "Qwen/Qwen2.5-Math-1.5B"

PLANNING = {
    "to","let","we","first","the","since","note","recall","now","next","then",
    "so","thus","hence","therefore","given","consider","suppose","assume",
    "because","if","for","by","from","using","applying","substituting",
    "simplifying","solving","calculating","computing","evaluating","finally",
    "step","answer","solution",
}
MATH_OPS = {"+","-","*","/","=","<",">","^","_","≤","≥","≠","±","×","÷"}
LATEX_HINTS = ("frac","sqrt","sum","int","lim","boxed","text","cdot","times")
STRUCT_LIT = {"**","##","#","---","```",":",";",",",".","!","?","(",")","[","]","{","}"}

def classify(s):
    st = s.strip()
    if st.lower() in PLANNING: return "planning"
    if s in ("\n","\r\n","\r") or st == "": return "structural"
    if st in STRUCT_LIT: return "structural"
    if st.startswith("**") or st.startswith("##"): return "structural"
    if st.startswith("\\") and len(st) > 1: return "math_latex"
    if st in MATH_OPS: return "math_operator"
    if st.replace(".","").replace(",","").isdigit(): return "math_number"
    if any(x in st for x in LATEX_HINTS): return "math_latex"
    if re.match(r'^-?\d+\.?\d*$', st): return "math_number"
    if len(st) <= 2 and not st.isalnum(): return "structural"
    return "continuation"

tok = AutoTokenizer.from_pretrained(TOK)

# ---- pass 1: collect per-token records ----
records = []  # (kl, surprise_s, tid, tstr, cat)
with open(DATA) as f:
    for line in f:
        d = json.loads(line)
        ids  = d["response_ids"]
        slps = d["student_lps"]
        tlps = d["teacher_lps"]
        L = min(len(ids), len(slps), len(tlps))
        for i in range(L):
            kl = abs(slps[i] - tlps[i])
            surp = -slps[i]
            tstr = tok.decode([ids[i]])
            cat  = classify(tstr)
            records.append((kl, surp, ids[i], tstr, cat))

n = len(records)
kl_arr   = np.array([r[0] for r in records])
surp_arr = np.array([r[1] for r in records])
kl_med   = float(np.median(kl_arr))
surp_med = float(np.median(surp_arr))
kl_p75   = float(np.percentile(kl_arr, 75))
surp_p75 = float(np.percentile(surp_arr, 75))

lines = []
def w(s=""): lines.append(s); print(s)
w(f"# KL × Entropy(surprise) Bucket Analysis")
w()
w(f"Data: {DATA}, total tokens: {n}")
w(f"KL stats:        median={kl_med:.3f}  p75={kl_p75:.3f}  max={kl_arr.max():.3f}")
w(f"Surprise stats:  median={surp_med:.3f}  p75={surp_p75:.3f}  max={surp_arr.max():.3f}")
w(f"(surprise = -log p(sampled token | student); proxy for distribution entropy)")
w()

# Use p75 for "high" so we're really looking at the upper quartile
def bucket(kl, surp):
    k_high = kl > kl_p75
    s_high = surp > surp_p75
    return ("hiKL_hiE" if k_high and s_high
            else "hiKL_loE" if k_high and not s_high
            else "loKL_hiE" if not k_high and s_high
            else "loKL_loE")

bucket_records = defaultdict(list)
for r in records:
    bucket_records[bucket(r[0], r[1])].append(r)

w("## Bucket sizes (split at KL>p75 and surprise>p75)")
w()
w("| Bucket | n | % of total |")
w("|---|---:|---:|")
for b in ["hiKL_hiE","hiKL_loE","loKL_hiE","loKL_loE"]:
    nb = len(bucket_records[b])
    w(f"| {b} | {nb} | {nb/n*100:.1f}% |")
w()

# Category share within each bucket
w("## Category composition per bucket")
w()
w("| Bucket | planning | structural | math_latex | math_operator | math_number | continuation | mean KL | mean surp |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for b in ["hiKL_hiE","hiKL_loE","loKL_hiE","loKL_loE"]:
    rs = bucket_records[b]
    if not rs: continue
    cnt = Counter(r[4] for r in rs)
    total = len(rs)
    pct = {c: cnt.get(c,0)/total*100 for c in ["planning","structural","math_latex","math_operator","math_number","continuation"]}
    mk = np.mean([r[0] for r in rs])
    ms = np.mean([r[1] for r in rs])
    w(f"| **{b}** | {pct['planning']:.1f}% | {pct['structural']:.1f}% | {pct['math_latex']:.1f}% | {pct['math_operator']:.1f}% | {pct['math_number']:.1f}% | {pct['continuation']:.1f}% | {mk:.2f} | {ms:.2f} |")
w()

# Top tokens within each bucket (especially high-KL buckets)
w("## Top-frequency tokens within high-KL buckets")
w()
for b in ["hiKL_hiE","hiKL_loE"]:
    rs = bucket_records[b]
    w(f"### {b}  (n={len(rs)})")
    w()
    w("| Token | count | % of bucket | mean KL | mean surp | category |")
    w("|---|---:|---:|---:|---:|---|")
    by_tok = defaultdict(list)
    for r in rs:
        by_tok[r[3]].append(r)
    items = sorted(by_tok.items(), key=lambda x: -len(x[1]))[:30]
    for tstr, group in items:
        cnt = len(group)
        cat = classify(tstr)
        mk = np.mean([g[0] for g in group])
        ms = np.mean([g[1] for g in group])
        w(f"| {repr(tstr)} | {cnt} | {cnt/max(len(rs),1)*100:.1f}% | {mk:.2f} | {ms:.2f} | {cat} |")
    w()

w("## Reading guide")
w()
w("- **hiKL_hiE** = student is uncertain AND disagrees with teacher → genuine 'pivot' positions where reasoning happens")
w("- **hiKL_loE** = student is confident but wrong (per teacher) → 'overconfident format / habit' positions; this is where prefix-100 likely helps most")
w("- **loKL_hiE** = student uncertain but teacher agrees with the sample → coverage tokens, low gradient")
w("- **loKL_loE** = both agree, both confident → easy positions, dropping these costs little")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f: f.write("\n".join(lines))
print("\nSaved", OUT)
