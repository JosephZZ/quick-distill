"""
Generate publication-quality figures for:
"Where Teachers Teach: The Positional Structure of Distillation Signal"

Saves each figure as PDF + PNG (300 dpi) to:
  - /Users/joseph/Desktop/Projects/quick-distillation/paper/figures/
  - /Users/joseph/Desktop/Projects/quick-distillation/deck/public/images/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from scipy.stats import linregress

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          12,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})

C_QWEN  = "#2166ac"   # blue
C_LLAMA = "#d6604d"   # orange-red
C_GRAY  = "#888888"
C_GREEN = "#1a9641"
C_RED   = "#d73027"
C_BLUE_LIGHT = "#c6dbef"

DIRS = [
    "/Users/joseph/Desktop/Projects/quick-distillation/paper/figures",
    "/Users/joseph/Desktop/Projects/quick-distillation/deck/public/images",
]


def save(fig, name):
    for d in DIRS:
        fig.savefig(f"{d}/{name}.pdf")
        fig.savefig(f"{d}/{name}.png", dpi=300)
    print(f"  saved {name}.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — KL Decay Curve  (two panels)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_kl_decay():
    bin_centers = [25, 75, 150, 250, 400]   # midpoints of bins
    bin_labels  = ["0–50", "50–100", "100–200", "200–300", "300–500"]

    kl_qwen  = [1.91, 0.94, 0.65, 0.52, 0.43]
    kl_llama = [0.186, 0.184, 0.193, 0.145, 0.148]

    agr_qwen  = [75.4, 81.9, 86.3, 87.8, 88.8]
    agr_llama = [89.1, 88.5, 90.1, 92.2, 91.7]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── Left: mean KL per position bin ────────────────────────────────────────
    ax = axes[0]
    ax.plot(bin_centers, kl_qwen,  "o-", color=C_QWEN,  label="Qwen3-1.7B")
    ax.plot(bin_centers, kl_llama, "s-", color=C_LLAMA, label="Llama-3.2-3B")

    # Optimal N markers
    ax.axvline(100, color=C_QWEN,  linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(200, color=C_LLAMA, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(100, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 2.1,
            " N=100", color=C_QWEN,  fontsize=10, va="top")
    ax.text(200, 0.20,
            " N=200", color=C_LLAMA, fontsize=10, va="bottom")

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Mean KL Divergence")
    ax.set_title("(a) KL Divergence by Position")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(bottom=0)

    # ── Right: agreement rate ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(bin_centers, agr_qwen,  "o-", color=C_QWEN,  label="Qwen3-1.7B")
    ax.plot(bin_centers, agr_llama, "s-", color=C_LLAMA, label="Llama-3.2-3B")

    ax.axvline(100, color=C_QWEN,  linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(200, color=C_LLAMA, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(100, 91, " N=100", color=C_QWEN,  fontsize=10)
    ax.text(202, 89.5, "N=200", color=C_LLAMA, fontsize=10)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Teacher–Student Agreement (%)")
    ax.set_title("(b) Agreement Rate by Position")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.legend(loc="lower right", frameon=False)
    ax.set_ylim(70, 97)

    fig.tight_layout(pad=1.5)
    save(fig, "fig1_kl_decay")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Token-Selection Paradox (scatter)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_token_paradox():
    points = {
        "prefix":      (45.6, 65.85),
        "ent_teacher": (50.5, 63.30),
        "ent_student": (67.0, 62.70),
        "ent_and":     (57.3, 54.50),
        "ent_or":      (64.3, 55.20),
        "top_kl":      (93.2, 53.35),
        "random":      (21.1, 61.25),
        "middle":      (15.8, 61.80),
        "last":        (13.9, 62.30),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    xs = np.array([v[0] for v in points.values()])
    ys = np.array([v[1] for v in points.values()])

    # Trend line
    slope, intercept, r, *_ = linregress(xs, ys)
    xfit = np.linspace(xs.min() - 5, xs.max() + 5, 200)
    ax.plot(xfit, slope * xfit + intercept, "--", color=C_GRAY,
            linewidth=1.4, alpha=0.8, label=f"Linear fit (r={r:.2f})")

    # Text offsets: (dx, dy) in data units
    offsets = {
        "prefix":      (-4,  0.6),
        "ent_teacher": ( 1.5, 0.4),
        "ent_student": ( 1.5,-0.5),
        "ent_and":     ( 1.5, 0.3),
        "ent_or":      ( 1.5,-0.6),
        "top_kl":      (-10,  0.5),
        "random":      ( 1.5, 0.3),
        "middle":      (-10, -0.7),
        "last":        (-10,  0.4),
    }

    for name, (x, y) in points.items():
        if name == "prefix":
            ax.scatter(x, y, marker="*", s=260, color=C_GREEN, zorder=5,
                       edgecolors="white", linewidths=0.5, label="prefix (best)")
        elif name == "top_kl":
            ax.scatter(x, y, marker="X", s=160, color=C_RED, zorder=5,
                       edgecolors="white", linewidths=0.5, label="top_kl (worst)")
        else:
            ax.scatter(x, y, marker="o", s=70, color=C_BLUE_LIGHT,
                       edgecolors=C_QWEN, linewidths=0.8, zorder=4)
        dx, dy = offsets[name]
        ax.annotate(name, (x, y), xytext=(x + dx, y + dy), fontsize=9,
                    color="#333333", ha="left" if dx > 0 else "right")

    # Annotation box
    ax.annotate("More KL coverage ≠\nbetter performance",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=10, color="#555555",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8))

    ax.set_xlabel("KL Coverage (%)")
    ax.set_ylabel("Best avg@4 (%)")
    ax.set_title("Token Selection Strategy vs. Performance")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig2_token_paradox")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Cross-Family Math Results (grouped bars)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_cross_family():
    families = ["Qwen", "Gemma", "Llama"]
    baseline = [50.95, 13.45, 15.20]
    pos_best  = [65.85, 25.80, 22.45]
    fullseq   = [62.35, 11.70, 20.65]

    x      = np.arange(len(families))
    width  = 0.24
    gap    = 0.04

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    bars_base = ax.bar(x - width - gap, baseline, width,
                       color="#aaaaaa", edgecolor="white", linewidth=0.5,
                       label="Baseline (no distill)")
    bars_pos  = ax.bar(x,               pos_best, width,
                       color=C_QWEN,    edgecolor="white", linewidth=0.5,
                       label="Best Positional (LoRA)")
    bars_full = ax.bar(x + width + gap, fullseq,  width,
                       color=C_RED,     edgecolor="white", linewidth=0.5,
                       label="Full-Seq (LoRA)")

    # Qwen full-seq collapse arrow  (real acc was 37.75 after \boxed{} repetition)
    collapse_y = 37.75
    ax.annotate("",
                xy=(x[0] + width + gap, collapse_y + 0.5),
                xytext=(x[0] + width + gap, fullseq[0] - 1),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5))
    ax.text(x[0] + width + gap + 0.02, collapse_y + 1.5,
            f"collapse\n→ {collapse_y}%", color=C_RED, fontsize=8.5,
            ha="left", va="bottom")

    # Gemma full-seq below baseline warning
    ax.text(x[1] + width + gap, fullseq[1] + 0.6, "[!]",
            ha="center", va="bottom", fontsize=9, color=C_RED,
            fontstyle="italic", fontweight="bold")

    # Value labels
    for bars in (bars_base, bars_pos, bars_full):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=12)
    ax.set_ylabel("avg@4 (%)")
    ax.set_title("Math Performance Across Teacher Families (LoRA)")
    ax.set_ylim(0, 75)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig3_cross_family")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Funcall: Student Exceeds Teacher
# ─────────────────────────────────────────────────────────────────────────────
def fig4_funcall():
    families = ["Qwen", "Gemma", "Llama"]
    teacher  = [54.0, 25.0, 63.7]
    baseline = [ 2.7,  0.0, 55.3]
    pos_best = [61.3, 30.9, 59.0]
    fullseq  = [58.2,  3.9, 32.0]

    x     = np.arange(len(families))
    width = 0.18
    gap   = 0.02

    fig, ax = plt.subplots(figsize=(9, 5.2))

    def bar_group(offset, values, color, label, hatch=None, edge="white"):
        bars = ax.bar(x + offset, values, width, color=color,
                      edgecolor=edge, linewidth=0.8,
                      hatch=hatch, label=label)
        return bars

    # Teacher — gray with dashed outline
    bars_teacher = ax.bar(x - 1.5*(width+gap), teacher, width,
                          color="none", edgecolor="#555555",
                          linewidth=1.4, linestyle="--", label="Teacher")
    bars_base = bar_group(-0.5*(width+gap), baseline, "#bbbbbb", "Baseline",
                           edge="#888888")
    bars_pos  = bar_group( 0.5*(width+gap), pos_best, C_QWEN,  "Best Positional")
    bars_full = bar_group( 1.5*(width+gap), fullseq,  C_RED,   "Full-Seq")

    # Annotate student > teacher
    student_exceeds = [
        (0, pos_best[0], teacher[0],  "Qwen pos"),   # Qwen pos > teacher
        (1, pos_best[1], teacher[1],  "Gemma pos"),  # Gemma pos > teacher
    ]
    for i, sy, ty, lbl in student_exceeds:
        bx = x[i] + 0.5*(width+gap) + width/2
        ax.annotate("",
                    xy=(bx, sy + 0.5),
                    xytext=(bx, ty + 1.0),
                    arrowprops=dict(arrowstyle="-|>", color=C_GREEN,
                                   lw=1.5, mutation_scale=12))
        ax.text(bx + 0.04, (sy + ty) / 2, "exceeds\nteacher",
                fontsize=8, color=C_GREEN, va="center")

    # Value labels on pos and fullseq bars
    for bars in (bars_pos, bars_full, bars_base):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    # Teacher value labels
    for bar in bars_teacher:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7.5,
                color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=12)
    ax.set_ylabel("Funcall Score (%)")
    ax.set_title("Function-Calling: Student vs. Teacher Performance")
    ax.set_ylim(0, 75)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig4_funcall")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Cascade Effect
# ─────────────────────────────────────────────────────────────────────────────
def fig5_cascade():
    bin_centers = [25, 75, 150, 250, 350]
    bin_labels  = ["0–50", "50–100", "100–200", "200–300", "300–400"]

    before = [2.064, 0.759, 0.441, 0.382, 0.331]
    after  = [1.015, 0.321, 0.242, 0.238, 0.216]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Shade trained region
    ax.axvspan(0, 200, color=C_BLUE_LIGHT, alpha=0.35, zorder=0, label="Trained region (0–200)")
    # Shade untrained region
    ax.axvspan(200, 400, color="#fee0d2", alpha=0.25, zorder=0, label="Untrained region (200–400)")

    ax.plot(bin_centers, before, "o-", color=C_RED,   label="Before distillation")
    ax.plot(bin_centers, after,  "s-", color=C_QWEN,  label="After pos-200 distillation")

    ax.axvline(200, color="#444444", linestyle=":", linewidth=1.2)
    ax.text(200 + 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 2.2,
            "Position limit (N=200)", color="#444444",
            fontsize=9, rotation=90, va="top", ha="left")

    # Annotation for cascade reduction
    ax.annotate("35–38% KL reduction\nat UNTRAINED positions",
                xy=(300, (before[3] + after[3]) / 2),
                xytext=(310, 0.9),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                fontsize=9.5, ha="left", color="#333333",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#cccccc", lw=0.8))

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Mean KL Divergence")
    ax.set_title("Cascade Effect: KL Reduction Beyond Trained Region")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels)
    ax.set_xlim(0, 410)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, "fig5_cascade")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Figure 1: KL Decay Curve …")
    fig1_kl_decay()
    print("Generating Figure 2: Token Selection Paradox …")
    fig2_token_paradox()
    print("Generating Figure 3: Cross-Family Math Results …")
    fig3_cross_family()
    print("Generating Figure 4: Funcall — Student Exceeds Teacher …")
    fig4_funcall()
    print("Generating Figure 5: Cascade Effect …")
    fig5_cascade()
    print("Done.")
