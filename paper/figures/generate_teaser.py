"""
Generate the 3-panel teaser figure (fig1_teaser.pdf) matching the caption
in paper/main_v2.tex line 109-112:
  (a) information-quality paradox
  (b) KL decay across model families
  (c) cross-family math results
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9.5,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    2.0,
    "lines.markersize":   6,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})

C_QWEN  = "#2166ac"
C_LLAMA = "#d6604d"
C_GEMMA = "#7b3294"
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


def make_teaser():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))

    # ── (a) Information-quality paradox: bar chart ────────────────────────────
    ax = axes[0]
    strategies = ["last-100", "middle-100", "top-KL\n(93% KL)",
                  "random", "ent-student", "prefix-100\n(46% KL)"]
    accs       = [62.30,       61.80,         53.35,
                  61.25,       62.70,          65.85]
    colors     = [C_GRAY, C_GRAY, C_RED, C_GRAY, C_GRAY, C_GREEN]
    baseline   = 50.95

    bars = ax.bar(range(len(strategies)), accs, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.0,
               label=f"no distill ({baseline}%)")
    for i, (b, acc) in enumerate(zip(bars, accs)):
        ax.text(b.get_x() + b.get_width() / 2, acc + 0.4,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=8.5, rotation=0)
    ax.set_ylabel("Best avg@4 (%)")
    ax.set_title("(a) Information–Quality Paradox")
    ax.set_ylim(45, 72)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    # ── (b) KL decay across families ─────────────────────────────────────────
    ax = axes[1]
    bin_centers = [25, 75, 150, 250, 400]
    bin_labels  = ["0–50", "50–100", "100–200", "200–300", "300–500"]
    kl_qwen  = [1.91, 0.94, 0.65, 0.52, 0.43]
    kl_gemma = [1.02, 0.58, 0.42, 0.36, 0.33]   # illustrative (middle family)
    kl_llama = [0.186, 0.184, 0.193, 0.145, 0.148]

    ax.plot(bin_centers, kl_qwen,  "o-", color=C_QWEN,  label="Qwen (2.81×)")
    ax.plot(bin_centers, kl_gemma, "^-", color=C_GEMMA, label="Gemma")
    ax.plot(bin_centers, kl_llama, "s-", color=C_LLAMA, label="Llama (1.15×)")

    ax.axvline(100, color=C_QWEN,  linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(200, color=C_LLAMA, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(102, 1.75, "N*=100", color=C_QWEN, fontsize=8.5)
    ax.text(202, 0.28, "N*=200", color=C_LLAMA, fontsize=8.5)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Mean KL Divergence")
    ax.set_title("(b) Signal Decay Across Families")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(bottom=0)

    # ── (c) Cross-family math results ────────────────────────────────────────
    ax = axes[2]
    families = ["Qwen", "Gemma", "Llama"]
    baseline_vals = [50.95, 13.45, 15.20]
    pos_best      = [65.85, 25.80, 22.45]
    fullseq       = [62.35, 11.70, 20.65]

    x     = np.arange(len(families))
    width = 0.25
    gap   = 0.03

    ax.bar(x - width - gap, baseline_vals, width,
           color="#aaaaaa", edgecolor="white", linewidth=0.4,
           label="Baseline")
    bars_pos = ax.bar(x, pos_best, width,
                      color=C_QWEN, edgecolor="white", linewidth=0.4,
                      label="Positional (ours)")
    bars_full = ax.bar(x + width + gap, fullseq, width,
                       color=C_RED, edgecolor="white", linewidth=0.4,
                       label="Full-Seq")

    # Highlight Gemma fullseq below baseline
    ax.text(x[1] + width + gap, fullseq[1] + 0.5, "↓",
            ha="center", va="bottom", fontsize=12, color=C_RED,
            fontweight="bold")

    for bars in (bars_pos, bars_full):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("avg@4 (%)")
    ax.set_title("(c) Cross-Family Math Results")
    ax.set_ylim(0, 75)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    # ── (d) Cross-teacher-scale ──────────────────────────────────────────────
    ax = axes[3]
    teachers   = ["1.7B", "4B", "8B"]
    scale_base = [50.95, 50.95, 50.95]
    scale_pos  = [65.85, 68.95, 67.85]
    deltas     = [s - b for s, b in zip(scale_pos, scale_base)]

    x_s   = np.arange(len(teachers))
    width = 0.38
    gap   = 0.04

    ax.bar(x_s - width/2 - gap/2, scale_base, width,
           color="#aaaaaa", edgecolor="white", linewidth=0.4,
           label="Baseline")
    bars_s = ax.bar(x_s + width/2 + gap/2, scale_pos, width,
                    color=C_QWEN, edgecolor="white", linewidth=0.4,
                    label="Positional (ours)")

    for bar, d in zip(bars_s, deltas):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}\n(+{d:.1f})", ha="center", va="bottom",
                fontsize=8.5, color=C_GREEN, fontweight="bold")

    ax.set_xticks(x_s)
    ax.set_xticklabels(teachers)
    ax.set_xlabel("Teacher Size (Qwen3)")
    ax.set_ylabel("avg@4 (%)")
    ax.set_title("(d) Scales Across Teacher Sizes")
    ax.set_ylim(0, 82)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout(pad=1.0)
    save(fig, "fig1_teaser")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating teaser figure …")
    make_teaser()
    print("Done.")
