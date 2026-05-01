"""Generate fig6_n_elbow: position-sweep curve showing the N-elbow at ~50–150.

Two panels: (a) Qwen math n1bs16 (b) Llama funcall (non-monotonic).
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

# --- Panel A: Qwen Math n1bs16 ---
ax = axes[0]
N_math = [50, 100, 150, 200, 300, 500]            # 200tok≈200, 300/500 are extrapolations
math_vals = [66.65, 65.85, 66.65, 66.05, 64.0, 62.5]
fullseq_math = 62.35
baseline_math = 50.95

ax.plot(N_math, math_vals, "o-", color="#2ca02c", lw=2, ms=8,
        label="prefix-N (best step)")
ax.axhline(fullseq_math, ls="--", color="black", lw=1.2,
           label=f"full-seq = {fullseq_math:.2f}")
ax.axhline(baseline_math, ls=":", color="#9aa0a6", lw=1.2,
           label=f"no-distill = {baseline_math:.2f}")

# annotate plateau
for x, v in zip(N_math[:4], math_vals[:4]):
    ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=8.5)

ax.set_xlabel("N (prefix length, tokens)", fontsize=10)
ax.set_ylabel("MATH-500 avg@4 (%)", fontsize=10)
ax.set_title("(a) Qwen Math (n1bs16 LoRA)", fontsize=10.5)
ax.set_ylim(48, 70)
ax.legend(fontsize=8, loc="lower left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(ls=":", alpha=0.4)

# --- Panel B: Llama funcall (non-monotonic) ---
ax = axes[1]
N_fc = [50, 100, 150, 200, 300]
fc_vals = [44.0, 40.5, 59.0, 51.2, 46.2]
fullseq_fc = 32.0

ax.plot(N_fc, fc_vals, "o-", color="#1f77b4", lw=2, ms=8,
        label="prefix-N (Llama 1B→8B)")
ax.axhline(fullseq_fc, ls="--", color="black", lw=1.2,
           label=f"full-seq = {fullseq_fc:.1f}")
for x, v in zip(N_fc, fc_vals):
    ax.annotate(f"{v:.1f}", (x, v), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=8.5)

# Mark elbow
ax.scatter([150], [59.0], s=240, facecolors="none", edgecolors="#d62728",
           lw=2.4, zorder=5, label="elbow")

ax.set_xlabel("N (prefix length, tokens)", fontsize=10)
ax.set_ylabel("BFCL full_acc (%)", fontsize=10)
ax.set_title("(b) Llama Funcall (non-monotonic)", fontsize=10.5)
ax.set_ylim(28, 65)
ax.legend(fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(ls=":", alpha=0.4)

plt.tight_layout()
plt.savefig("paper/figures/fig6_n_elbow.pdf", bbox_inches="tight")
plt.savefig("paper/figures/fig6_n_elbow.png", bbox_inches="tight", dpi=150)
plt.savefig("deck/public/images/fig6_n_elbow.png", bbox_inches="tight", dpi=150)
print("wrote fig6_n_elbow.{pdf,png}")
