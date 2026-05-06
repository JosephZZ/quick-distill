#!/usr/bin/env python3
"""Compare 4 selection schemes on dumped per-token signals.

Schemes (all use top-k=100 per example):
  raw:  KL * H_s * H_t
  z:    z(KL) + z(H_s) + z(H_t)    (additive of z-scores)
  zprod: z(KL) * z(H_s) * z(H_t) on positive offset
  rank: rank(KL) * rank(H_s) * rank(H_t)   (percentile within example)
  min:  min(z(KL), z(H_s), z(H_t))

Outputs all go in docs/signal_analysis_v2/scheme_compare/:
  selection_stats.json  — per-scheme: signal mean/median/quantiles, position hist
  jaccard.json          — scheme x scheme overlap
  pos_hist.png          — histogram of selected positions for all schemes
  per_pos_rate.png      — selection rate curve vs position
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

K = 100
SRC = "/zhi_backup/ziheng/quick-distillation/docs/signal_analysis_v2/per_token_signals.npz"
OUT = "/zhi_backup/ziheng/quick-distillation/docs/signal_analysis_v2/scheme_compare"
os.makedirs(OUT, exist_ok=True)

d = np.load(SRC)
kl, hs, ht, resp_len = d["kl_rev"].astype(np.float32), d["h_s"].astype(np.float32), d["h_t"].astype(np.float32), d["resp_len"]
n, L = kl.shape
print(f"n={n}, max_len={L}, mean resp_len={resp_len.mean():.0f}")

def z(x, valid):
    m = x[valid].mean()
    s = x[valid].std() + 1e-8
    out = np.full_like(x, np.nan)
    out[valid] = (x[valid] - m) / s
    return out

def rank_pct(x, valid):
    out = np.full_like(x, np.nan)
    v = x[valid]
    order = np.argsort(np.argsort(v))
    out[valid] = order / max(len(v) - 1, 1)
    return out

# Build scores per example
schemes = ["raw", "z_add", "z_prod", "rank", "min"]
selected = {s: np.zeros_like(kl, dtype=bool) for s in schemes}

for i in range(n):
    rl = resp_len[i]
    if rl < K:
        continue
    valid = np.zeros(L, dtype=bool); valid[:rl] = True
    k_i, hs_i, ht_i = kl[i], hs[i], ht[i]

    # raw product (shift to positive first — KL>=0, H>=0 already, fine)
    raw = k_i * hs_i * ht_i
    # z-scored additive
    zk, zhs, zht = z(k_i, valid), z(hs_i, valid), z(ht_i, valid)
    z_add = zk + zhs + zht
    # z product — shift each to [0, inf) then multiply
    def shift(x):
        out = np.full_like(x, np.nan)
        mn = np.nanmin(x[valid])
        out[valid] = x[valid] - mn + 1e-6
        return out
    z_prod = shift(zk) * shift(zhs) * shift(zht)
    # rank product
    rk, rhs, rht = rank_pct(k_i, valid), rank_pct(hs_i, valid), rank_pct(ht_i, valid)
    rank_p = rk * rhs * rht
    # min of z
    mn_z = np.minimum(np.minimum(zk, zhs), zht)

    for name, arr in [("raw", raw), ("z_add", z_add), ("z_prod", z_prod), ("rank", rank_p), ("min", mn_z)]:
        scores = np.where(valid, arr, -np.inf)
        top = np.argpartition(-scores, K-1)[:K]
        selected[name][i, top] = True

# per-signal stats under each scheme
stats = {}
for s in schemes:
    mask = selected[s]
    sel_kl, sel_hs, sel_ht = kl[mask], hs[mask], ht[mask]
    pos = np.tile(np.arange(L), (n, 1))[mask]
    stats[s] = {
        "n_selected": int(mask.sum()),
        "kl": {"mean": float(sel_kl.mean()), "p50": float(np.median(sel_kl)), "p90": float(np.percentile(sel_kl, 90))},
        "h_s": {"mean": float(sel_hs.mean()), "p50": float(np.median(sel_hs)), "p90": float(np.percentile(sel_hs, 90))},
        "h_t": {"mean": float(sel_ht.mean()), "p50": float(np.median(sel_ht)), "p90": float(np.percentile(sel_ht, 90))},
        "pos": {"mean": float(pos.mean()), "p50": float(np.median(pos)), "p90": float(np.percentile(pos, 90)),
                "frac_in_prefix_150": float((pos < 150).mean())},
    }
json.dump(stats, open(os.path.join(OUT, "selection_stats.json"), "w"), indent=2)
print("wrote selection_stats.json")

# Jaccard overlap
jac = {}
for a in schemes:
    jac[a] = {}
    for b in schemes:
        inter = (selected[a] & selected[b]).sum()
        uni = (selected[a] | selected[b]).sum()
        jac[a][b] = float(inter / uni) if uni else 0.0
json.dump(jac, open(os.path.join(OUT, "jaccard.json"), "w"), indent=2)
print("wrote jaccard.json")

# Position histogram
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.arange(0, 801, 20)
colors = {"raw": "#1f77b4", "z_add": "#ff7f0e", "z_prod": "#d62728", "rank": "#2ca02c", "min": "#9467bd"}
for s in schemes:
    mask = selected[s]
    pos = np.tile(np.arange(L), (n, 1))[mask]
    ax.hist(pos, bins=bins, alpha=0.45, label=s, color=colors[s], histtype="step", lw=2)
ax.axvspan(0, 150, color="#fff3cd", alpha=0.4, zorder=0, label="prefix (<150)")
ax.set_xlabel("token position")
ax.set_ylabel("# selected tokens")
ax.set_title(f"Selected-token position distribution, top-k={K} per example (n={n})")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pos_hist.png"), dpi=140)
plt.close()
print("wrote pos_hist.png")

# Per-position selection rate (denominator = # examples where pos < resp_len)
valid_count = np.zeros(L)
for rl in resp_len:
    valid_count[:rl] += 1
fig, ax = plt.subplots(figsize=(8, 4.5))
for s in schemes:
    rate = selected[s].sum(axis=0) / np.maximum(valid_count, 1)
    # smooth with window=10
    w = 10
    rate_smooth = np.convolve(rate, np.ones(w)/w, mode="same")
    ax.plot(np.arange(L), rate_smooth, label=s, color=colors[s], lw=1.8)
ax.axvspan(0, 150, color="#fff3cd", alpha=0.4, zorder=0)
ax.set_xlim(0, 800)
ax.set_xlabel("token position")
ax.set_ylabel("P(selected | position valid)")
ax.set_title(f"Per-position selection rate (smoothed, window=10), top-k={K}")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "per_pos_rate.png"), dpi=140)
plt.close()
print("wrote per_pos_rate.png")

# summary print
print("\n=== per-scheme summary ===")
for s in schemes:
    st = stats[s]
    print(f"{s:8s}  kl={st['kl']['mean']:.2f}  h_s={st['h_s']['mean']:.3f}  h_t={st['h_t']['mean']:.3f}  "
          f"pos_mean={st['pos']['mean']:.0f}  prefix_frac={st['pos']['frac_in_prefix_150']:.2f}")
print("\n=== jaccard ===")
print("         " + "  ".join(f"{s:>7s}" for s in schemes))
for a in schemes:
    print(f"{a:8s} " + "  ".join(f"{jac[a][b]:7.3f}" for b in schemes))
