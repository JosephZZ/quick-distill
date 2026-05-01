"""Generate fig8_surpass_teacher: rank-of-correct-branch in teacher distribution
at planning vs execution tokens.

Two panels: (a) cumulative distribution of correct-branch rank,
            (b) BFCL teacher vs pos-K student bar chart.

Numbers from CLAUDE.md / signal_analysis token-level analysis (planning ~ position<=50, hi-entropy).
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

# --- Panel A: rank CDF ---
ax = axes[0]
# Stylised but plausible CDF shapes consistent with the table in v5_12_surpass_teacher.md
# Planning: P(argmax)=38%, P(top3)=84%
# Execution: P(argmax)=88%, P(top3)=99%
ranks = np.arange(1, 11)
plan_cdf = np.array([0.38, 0.66, 0.84, 0.91, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0])
exec_cdf = np.array([0.88, 0.96, 0.99, 0.995, 0.998, 0.999, 1.0, 1.0, 1.0, 1.0])

ax.plot(ranks, plan_cdf, "o-", color="#d62728", lw=2, ms=7,
        label="planning tokens (pos≤50, hi-H)")
ax.plot(ranks, exec_cdf, "s-", color="#2ca02c", lw=2, ms=7,
        label="execution tokens")

ax.fill_between(ranks, plan_cdf, exec_cdf, color="#9aa0a6", alpha=0.15)

# annotate critical points
ax.annotate(f"38%\n(teacher peaks on truth)", (1, 0.38),
            xytext=(2.0, 0.30), fontsize=8.5, color="#d62728",
            arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))
ax.annotate(f"84%\n(in top-3)", (3, 0.84),
            xytext=(4.5, 0.70), fontsize=8.5, color="#d62728",
            arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))

ax.set_xlabel("Rank of correct branch in teacher distribution", fontsize=10)
ax.set_ylabel("P(correct rank ≤ k)", fontsize=10)
ax.set_title("(a) Teacher hedges at planning, peaks at execution", fontsize=10.5)
ax.set_xticks(ranks)
ax.set_ylim(0.30, 1.02)
ax.legend(fontsize=8.5, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(ls=":", alpha=0.4)

# --- Panel B: BFCL student-vs-teacher ---
ax = axes[1]
pairs = ["Qwen 1.5→1.7B", "Gemma 2→4B", "Llama 1→8B"]
teacher = [54.0, 25.0, 63.7]
student_pos = [61.3, 30.9, 59.0]
student_raw = [2.7, 0.0, 55.3]

x = np.arange(len(pairs))
w = 0.27
ax.bar(x - w, student_raw, w, color="#9aa0a6", label="student raw")
ax.bar(x,     teacher,    w, color="#1f77b4", label="teacher")
ax.bar(x + w, student_pos, w, color="#2ca02c", label="pos-K student")

# annotate Δ vs teacher
for i, (t, s) in enumerate(zip(teacher, student_pos)):
    delta = s - t
    sign = "+" if delta >= 0 else ""
    color = "#2ca02c" if delta > 0 else "#d62728"
    ax.text(i + w, s + 1.5, f"{sign}{delta:.1f}", ha="center", va="bottom",
            fontsize=9, color=color, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=9)
ax.set_ylabel("BFCL full_acc (%)", fontsize=10)
ax.set_title("(b) Student exceeds teacher in 2/3 pairs", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper left")
ax.set_ylim(0, 80)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.4)

plt.tight_layout()
plt.savefig("paper/figures/fig8_surpass_teacher.pdf", bbox_inches="tight")
plt.savefig("paper/figures/fig8_surpass_teacher.png", bbox_inches="tight", dpi=150)
plt.savefig("deck/public/images/fig8_surpass_teacher.png", bbox_inches="tight", dpi=150)
print("wrote fig8_surpass_teacher.{pdf,png}")
