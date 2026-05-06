"""
Per-trajectory top-100 vs prefix-100 overlap, for several scoring rules:
  - KL alone:        score = |s_lp - t_lp|
  - surprise alone:  score = -s_lp
  - KL * surprise:   score = |s_lp - t_lp| * (-s_lp)

For each rule, compute per-trajectory overlap with positions {0..K-1}.
"""
import json
import sys
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/zhi_backup/ziheng/quick-distillation/docs/kl_position_analysis_v2/raw_logprobs.jsonl"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 100

def topk_indices(score, k):
    if k >= len(score):
        return np.arange(len(score))
    return np.argpartition(-score, k - 1)[:k]

records_kl = []
records_surp = []
records_joint = []
total_kl_in_prefix = 0
total_surp_in_prefix = 0
total_joint_in_prefix = 0
total_topk = 0
top_pos_bins_kl   = {"0-100": 0, "100-300": 0, "300-500": 0, "500+": 0}
top_pos_bins_surp = {"0-100": 0, "100-300": 0, "300-500": 0, "500+": 0}
n_traj = 0
traj_lens = []

with open(DATA) as f:
    for line in f:
        d = json.loads(line)
        slps = np.asarray(d["student_lps"], dtype=np.float64)
        tlps = np.asarray(d["teacher_lps"], dtype=np.float64)
        L = min(len(slps), len(tlps))
        if L < 5:
            continue
        slps = slps[:L]; tlps = tlps[:L]
        kl   = np.abs(slps - tlps)
        surp = np.maximum(-slps, 0.0)
        joint = kl * surp

        k = min(K, L)
        prefix = set(range(k))

        for arr, store, in_prefix_total, bin_dict in [
            (kl,    records_kl,    "kl",    top_pos_bins_kl),
            (surp,  records_surp,  "surp",  top_pos_bins_surp),
            (joint, records_joint, "joint", None),
        ]:
            top = topk_indices(arr, k)
            ov = len(set(top.tolist()) & prefix) / k
            store.append(ov)
            if in_prefix_total == "kl":
                total_kl_in_prefix += len(set(top.tolist()) & prefix)
            elif in_prefix_total == "surp":
                total_surp_in_prefix += len(set(top.tolist()) & prefix)
            else:
                total_joint_in_prefix += len(set(top.tolist()) & prefix)
            if bin_dict is not None:
                for pos in top:
                    if pos < 100:   bin_dict["0-100"] += 1
                    elif pos < 300: bin_dict["100-300"] += 1
                    elif pos < 500: bin_dict["300-500"] += 1
                    else:           bin_dict["500+"] += 1
        total_topk += k
        n_traj += 1
        traj_lens.append(L)

def report(name, rec, total_in_prefix, total, bins=None):
    a = np.array(rec)
    print(f"## {name}")
    print(f"  per-traj mean overlap  = {a.mean()*100:.2f}%")
    print(f"  per-traj median        = {np.median(a)*100:.2f}%")
    print(f"  std / p10 / p90        = {a.std()*100:.1f}% / {np.percentile(a,10)*100:.1f}% / {np.percentile(a,90)*100:.1f}%")
    print(f"  pooled share in 0-100  = {total_in_prefix/total*100:.2f}%")
    if bins is not None:
        tot = sum(bins.values())
        print(f"  position breakdown:")
        for k_, v in bins.items():
            print(f"    {k_:8s}: {v/tot*100:6.2f}%")
    print()

print(f"# Top-{K} vs prefix-{K} overlap (per trajectory)")
print(f"Source: {DATA}")
print(f"Trajectories: {n_traj}, mean length {np.mean(traj_lens):.1f}")
print()
report("score = |s_lp - t_lp|         (KL alone)",      records_kl,    total_kl_in_prefix,    total_topk, top_pos_bins_kl)
report("score = -s_lp                 (surprise alone)", records_surp,  total_surp_in_prefix,  total_topk, top_pos_bins_surp)
report("score = KL * surprise          (joint)",         records_joint, total_joint_in_prefix, total_topk)
