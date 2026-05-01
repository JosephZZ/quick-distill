"""
Position × KL × surprise bucket distribution.

Tests the conceptual bridge: does the prefix-100 region concentrate the
"useful gradient" bucket (hiKL_hiE) and avoid the "harmful gradient" bucket
(hiKL_loE)?

If yes, prefix-100 is a CHEAP POSITIONAL PROXY for the principled rule
"select tokens where student is genuinely uncertain AND teacher disagrees".
If no, the prefix advantage isn't reducible to KL × surprise selection and
some other mechanism (cascade, position-specific dynamics) is doing work.

Output: for each position window (0-100, 100-300, 300-500, 500+), share of
each bucket. Uses GLOBAL p75 thresholds so buckets are commensurate with
the existing kl_x_entropy_buckets.md analysis.
"""

import json
import sys
import os
import numpy as np
from collections import defaultdict

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/zhi_backup/ziheng/quick-distillation/docs/kl_position_analysis_v2/raw_logprobs.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/zhi_backup/ziheng/quick-distillation/docs/position_x_bucket.md"

POS_BINS = [(0, 100), (100, 300), (300, 500), (500, 10**9)]
POS_LABELS = ["0-100", "100-300", "300-500", "500+"]

# Pass 1: collect per-(position, kl, surp) records
records = []  # (pos, kl, surp)
with open(DATA) as f:
    for line in f:
        d = json.loads(line)
        slps = d["student_lps"]
        tlps = d["teacher_lps"]
        L = min(len(slps), len(tlps))
        for i in range(L):
            kl = abs(slps[i] - tlps[i])
            surp = -slps[i]
            records.append((i, kl, surp))

n = len(records)
kl_arr   = np.array([r[1] for r in records])
surp_arr = np.array([r[2] for r in records])
kl_p75   = float(np.percentile(kl_arr, 75))
surp_p75 = float(np.percentile(surp_arr, 75))

def bucket(kl, surp):
    k_high = kl > kl_p75
    s_high = surp > surp_p75
    if k_high and s_high: return "hiKL_hiE"
    if k_high and not s_high: return "hiKL_loE"
    if not k_high and s_high: return "loKL_hiE"
    return "loKL_loE"

# Aggregate per position bin
bin_counts = [defaultdict(int) for _ in POS_BINS]
bin_total  = [0] * len(POS_BINS)
bin_kl_sum = [0.0] * len(POS_BINS)
bin_surp_sum = [0.0] * len(POS_BINS)

for pos, kl, surp in records:
    for bi, (lo, hi) in enumerate(POS_BINS):
        if lo <= pos < hi:
            b = bucket(kl, surp)
            bin_counts[bi][b] += 1
            bin_total[bi] += 1
            bin_kl_sum[bi] += kl
            bin_surp_sum[bi] += surp
            break

lines = []
def w(s=""): lines.append(s); print(s)

w("# Position × KL × Surprise Bucket Distribution")
w()
w(f"Data: `{DATA}`")
w(f"Total tokens: {n}")
w(f"Global thresholds: KL_p75={kl_p75:.3f}, surp_p75={surp_p75:.3f}")
w()
w("## Bucket share by position window")
w()
w("Each row: % of tokens IN that position window that fall into each bucket.")
w()
w("| Position | n | hiKL_hiE | hiKL_loE | loKL_hiE | loKL_loE | mean KL | mean surp |")
w("|---|---:|---:|---:|---:|---:|---:|---:|")
for bi, label in enumerate(POS_LABELS):
    total = bin_total[bi]
    if total == 0:
        w(f"| {label} | 0 | - | - | - | - | - | - |")
        continue
    pct = {b: bin_counts[bi].get(b, 0) / total * 100 for b in
           ["hiKL_hiE","hiKL_loE","loKL_hiE","loKL_loE"]}
    mk = bin_kl_sum[bi] / total
    ms = bin_surp_sum[bi] / total
    w(f"| **{label}** | {total} | {pct['hiKL_hiE']:.1f}% | {pct['hiKL_loE']:.1f}% | {pct['loKL_hiE']:.1f}% | {pct['loKL_loE']:.1f}% | {mk:.2f} | {ms:.2f} |")
w()

# Also: cumulative — what fraction of all hiKL_hiE / hiKL_loE tokens fall in each window?
w("## Cumulative bucket coverage by position window")
w()
w("Each row: % of ALL tokens in that bucket (across full sequences) that")
w("fall in this position window. Tells us where each bucket's mass lives.")
w()
total_per_bucket = defaultdict(int)
for bi in range(len(POS_BINS)):
    for b, c in bin_counts[bi].items():
        total_per_bucket[b] += c

w("| Position | hiKL_hiE share | hiKL_loE share | loKL_hiE share | loKL_loE share |")
w("|---|---:|---:|---:|---:|")
for bi, label in enumerate(POS_LABELS):
    parts = []
    for b in ["hiKL_hiE","hiKL_loE","loKL_hiE","loKL_loE"]:
        c = bin_counts[bi].get(b, 0)
        tot = total_per_bucket.get(b, 1)
        parts.append(f"{c/tot*100:.1f}%")
    w(f"| **{label}** | {' | '.join(parts)} |")
w()

# Direct test: position 0-100 vs full distribution
w("## Direct test: prefix-100 vs full sequence")
w()
w(f"Baseline (full sequence) bucket shares are p75-by-construction:")
w(f"  hiKL_hiE ≈ 21%, hiKL_loE ≈ 4%, loKL_hiE ≈ 4%, loKL_loE ≈ 71%")
w()
prefix_total = bin_total[0]
if prefix_total > 0:
    pct = {b: bin_counts[0].get(b, 0) / prefix_total * 100 for b in
           ["hiKL_hiE","hiKL_loE","loKL_hiE","loKL_loE"]}
    w(f"Prefix-100 actual:")
    w(f"  hiKL_hiE = {pct['hiKL_hiE']:.1f}% (enrichment vs baseline 21%: {pct['hiKL_hiE']/21:.2f}x)")
    w(f"  hiKL_loE = {pct['hiKL_loE']:.1f}% (enrichment vs baseline ~4%: {pct['hiKL_loE']/4:.2f}x)")
    w(f"  loKL_hiE = {pct['loKL_hiE']:.1f}% (enrichment vs baseline ~4%: {pct['loKL_hiE']/4:.2f}x)")
    w(f"  loKL_loE = {pct['loKL_loE']:.1f}% (enrichment vs baseline ~71%: {pct['loKL_loE']/71:.2f}x)")
    w()
    w("Reading guide:")
    w("  - If prefix hiKL_hiE > baseline hiKL_hiE  → prefix concentrates 'useful gradient'")
    w("  - If prefix hiKL_loE < baseline hiKL_loE  → prefix avoids 'harmful gradient'")
    w("  - Both true → prefix-100 is a principled positional proxy for KL×entropy selection")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f: f.write("\n".join(lines))
print(f"\nSaved {OUT}")
