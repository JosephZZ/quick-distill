"""Generate fig3_signal_indicators: bar chart comparing KL / entropy / position selectors.

Three groups (KL, entropy, position) with full-seq baseline dashed line.
n1bs16 LoRA, MATH-500 avg@4. Best avg@4 across training steps for each method.
"""
import matplotlib.pyplot as plt
import numpy as np

# Canonical n1bs16 numbers (avg@4, MATH-500, K=100 budget)
methods = [
    ("Baseline\n(no distill)", 50.95, "#9aa0a6"),
    ("Top-KL", 58.60, "#d62728"),
    ("Top-ent\n(student)", 61.35, "#ff7f0e"),
    ("Format-mask", 62.05, "#ff7f0e"),
    ("Top-ent\n(teacher)", 62.20, "#ff7f0e"),
    ("Random", 63.05, "#9aa0a6"),
    ("Pos-100", 65.85, "#2ca02c"),
    ("Pos-50", 66.65, "#2ca02c"),
]
fullseq = 62.35

names = [m[0] for m in methods]
vals = [m[1] for m in methods]
colors = [m[2] for m in methods]

fig, ax = plt.subplots(figsize=(10, 4.6))
xs = np.arange(len(methods))
bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.6)

# fullseq dashed line
ax.axhline(fullseq, ls="--", color="black", lw=1.2, alpha=0.8)
ax.text(len(methods)-0.4, fullseq + 0.25, f"full-seq = {fullseq:.2f}",
        ha="right", va="bottom", fontsize=9, style="italic")

# Δ vs full-seq labels above each bar
for x, v in zip(xs, vals):
    delta = v - fullseq
    sign = "+" if delta >= 0 else ""
    label_color = "#2ca02c" if delta > 0 else ("#d62728" if delta < -1 else "#666")
    ax.text(x, v + 0.4, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(x, v + 1.6, f"{sign}{delta:.2f}", ha="center", va="bottom", fontsize=8.5,
            color=label_color)

# Group annotations (KL / entropy / position) underneath
ax.set_xticks(xs)
ax.set_xticklabels(names, fontsize=9)

# Label the "indicator family" with brackets above the x-axis labels
group_spans = [
    ("baseline", 0, 0, "#9aa0a6"),
    ("KL", 1, 1, "#d62728"),
    ("entropy / format", 2, 4, "#ff7f0e"),
    ("control", 5, 5, "#9aa0a6"),
    ("position", 6, 7, "#2ca02c"),
]

# Use figure-coordinate annotations below the x-tick labels
ax.set_ylim(46, 70)
trans = ax.get_xaxis_transform()  # x: data, y: axes (0 = bottom)
y_bracket = -0.18
y_label = -0.24
for label, lo, hi, color in group_spans:
    ax.plot([lo - 0.4, hi + 0.4], [y_bracket, y_bracket], color=color, lw=2.5,
            solid_capstyle="butt", transform=trans, clip_on=False)
    ax.text((lo + hi) / 2, y_label, label, ha="center", va="top",
            fontsize=9, color=color, fontweight="bold", transform=trans)
ax.set_ylabel("MATH-500 avg@4 (%)", fontsize=10)
ax.set_title("Signal indicators at K=100: only position exceeds full-seq",
             fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.4)

plt.tight_layout()
plt.savefig("paper/figures/fig3_signal_indicators.pdf", bbox_inches="tight")
plt.savefig("paper/figures/fig3_signal_indicators.png", bbox_inches="tight", dpi=150)
plt.savefig("deck/public/images/fig3_signal_indicators.png", bbox_inches="tight", dpi=150)
print("wrote fig3_signal_indicators.{pdf,png}")
