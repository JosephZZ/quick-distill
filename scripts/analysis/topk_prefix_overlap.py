"""
Overlap rate between top-100-by-(KL*surprise) and prefix-100 (positions 0-99).

For each trajectory:
  - score[i] = |s_lp[i] - t_lp[i]| * (-s_lp[i])
  - top_idx  = indices of top-K positions by score (K = min(100, response length))
  - prefix   = {0, 1, ..., K-1}
  - overlap  = |top_idx ∩ prefix| / K

Reports per-trajectory mean overlap, and the distribution-level share of
top-100 tokens that fall in positions 0-99.
"""
import json
import sys
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/zhi_backup/ziheng/quick-distillation/docs/kl_position_analysis_v2/raw_logprobs.jsonl"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 100

per_traj_overlap = []
total_topk_tokens = 0
total_topk_in_prefix = 0
traj_lens = []
# also breakdown of where top-100 tokens land
top_pos_bins = {"0-100": 0, "100-300": 0, "300-500": 0, "500+": 0}
# What % of trajectories' top-K is exactly the prefix?
exact_prefix = 0
n_traj = 0

with open(DATA) as f:
    for line in f:
        d = json.loads(line)
        slps = np.asarray(d["student_lps"], dtype=np.float64)
        tlps = np.asarray(d["teacher_lps"], dtype=np.float64)
        L = min(len(slps), len(tlps))
        if L < 5:
            continue
        slps = slps[:L]
        tlps = tlps[:L]
        kl = np.abs(slps - tlps)
        surp = np.maximum(-slps, 0.0)
        score = kl * surp

        k = min(K, L)
        top_idx = np.argpartition(-score, k - 1)[:k]
        prefix = set(range(k))
        top_set = set(top_idx.tolist())

        ov = len(top_set & prefix) / k
        per_traj_overlap.append(ov)
        total_topk_tokens += k
        total_topk_in_prefix += len(top_set & prefix)
        traj_lens.append(L)
        if top_set == prefix:
            exact_prefix += 1
        for pos in top_idx:
            if pos < 100:
                top_pos_bins["0-100"] += 1
            elif pos < 300:
                top_pos_bins["100-300"] += 1
            elif pos < 500:
                top_pos_bins["300-500"] += 1
            else:
                top_pos_bins["500+"] += 1
        n_traj += 1

ov = np.array(per_traj_overlap)
print(f"# top-{K} (KL × surprise) vs prefix-{K} overlap")
print(f"Source: {DATA}")
print(f"Trajectories: {n_traj}")
print(f"Mean response length: {np.mean(traj_lens):.1f}")
print()
print("## Per-trajectory overlap")
print(f"  mean   = {ov.mean()*100:.2f}%")
print(f"  median = {np.median(ov)*100:.2f}%")
print(f"  std    = {ov.std()*100:.2f}%")
print(f"  p10    = {np.percentile(ov, 10)*100:.2f}%")
print(f"  p90    = {np.percentile(ov, 90)*100:.2f}%")
print(f"  trajectories where top-K = prefix exactly: {exact_prefix}/{n_traj} ({exact_prefix/n_traj*100:.1f}%)")
print()
print("## Distribution-level (pooled over all top-K tokens)")
total = sum(top_pos_bins.values())
print(f"  total top-K tokens: {total}")
print(f"  share in 0-100 (prefix-100): {total_topk_in_prefix/total*100:.2f}%")
print()
print("## Where top-K tokens land (pooled across trajectories)")
for k, v in top_pos_bins.items():
    print(f"  {k:8s}: {v:>8d}  ({v/total*100:.2f}%)")
