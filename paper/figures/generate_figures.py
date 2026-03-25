#!/usr/bin/env python3
"""Generate all publication-quality figures for the positional distillation paper."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Global style ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,       # TrueType fonts in PDF (editable text)
    'ps.fonttype': 42,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

# Professional color palette
C = {
    'blue':   '#2171b5',
    'orange': '#e6550d',
    'green':  '#31a354',
    'red':    '#de2d26',
    'purple': '#756bb1',
    'grey':   '#636363',
    'teal':   '#17becf',
    'brown':  '#8c564b',
}


# ======================================================================
# Fig 1a: Per-position KL divergence curve
# ======================================================================
def fig1a():
    # Exact data points from the docs (2k trajectories, positions 0-29)
    pos_exact = list(range(30))
    kl_exact = [3.057, 1.979, 1.104, 1.510, 0.921,
                0.962, 0.946, 0.846, 0.853, 0.936,
                1.036, 1.007, 0.843, 1.007, 0.839,
                1.027, 0.764, 1.011, 0.987, 0.943,
                1.031, 1.018, 0.891, 0.954, 0.946,
                1.050, 1.083, 1.095, 1.064, 1.207]

    # Extended data from 59k trajectories
    pos_ext = [30, 40, 50, 60, 70, 75, 80, 90, 100, 120, 140, 150,
               160, 170, 175, 180, 190, 192, 200]
    kl_ext = [1.1, 0.9, 0.693, 0.599, 0.535, 0.507, 0.479, 0.454,
              0.45, 0.43, 0.42, 0.410, 0.396, 0.386, 0.374, 0.375,
              0.366, 0.359, 0.355]

    # Combine
    pos_all = np.array(pos_exact + pos_ext, dtype=float)
    kl_all = np.array(kl_exact + kl_ext, dtype=float)
    order = np.argsort(pos_all)
    pos_all, kl_all = pos_all[order], kl_all[order]

    # Smooth interpolation
    pos_smooth = np.linspace(0, 200, 600)
    spl = make_interp_spline(pos_all, kl_all, k=3)
    kl_smooth = spl(pos_smooth)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Shade first 50 positions
    mask = pos_smooth <= 50
    ax.fill_between(pos_smooth[mask], 0, kl_smooth[mask],
                    color=C['blue'], alpha=0.08, zorder=0)
    ax.annotate('26% of total KL\n$\\rightarrow$ 73% of perf. gain',
                xy=(25, 0.18), fontsize=7, color=C['blue'],
                ha='center', style='italic')

    # Main curve
    ax.plot(pos_smooth, kl_smooth, color=C['blue'], linewidth=1.5, zorder=3)

    # Scatter key points
    key_idx = [0, 1, 3, 4, 9, 19, 29]
    ax.scatter([pos_exact[i] for i in key_idx],
               [kl_exact[i] for i in key_idx],
               color=C['blue'], s=12, zorder=4, edgecolors='white', linewidths=0.3)

    # Annotate position 0
    ax.annotate(f'Pos 0: {kl_exact[0]:.2f}',
                xy=(0, kl_exact[0]), xytext=(22, 3.1),
                fontsize=7.5, color=C['grey'],
                arrowprops=dict(arrowstyle='->', color=C['grey'], lw=0.7))

    ax.set_xlabel('Token position')
    ax.set_ylabel('KL divergence')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 3.5)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1a_kl_curve.pdf'))
    plt.close(fig)
    print('  saved fig1a_kl_curve.pdf')


# ======================================================================
# Fig 1b: MATH-500 avg@4 vs position limit
# ======================================================================
def fig1b():
    # n=16 config (200 problems, n_samples=16)
    pos_n16 = [5, 10, 20, 50, 100, 150, 200]
    perf_n16 = [56.50, 59.50, 60.20, 62.45, 64.25, 65.70, 66.75]

    # n=1 config (3200 problems, n_samples=1)
    pos_n1 = [50, 100, 150, 200]
    perf_n1 = [66.65, 65.85, 66.65, 66.05]

    baseline = 50.95
    fullseq_best = 65.55

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.plot(pos_n16, perf_n16, '-o', color=C['blue'],
            label='$n{=}16$, 200 problems', markersize=5, zorder=3)
    ax.plot(pos_n1, perf_n1, '-s', color=C['orange'],
            label='$n{=}1$, 3200 problems', markersize=5, zorder=3)

    ax.axhline(baseline, color=C['grey'], linestyle='--', linewidth=0.9, alpha=0.7)
    ax.axhline(fullseq_best, color=C['green'], linestyle='-.', linewidth=0.9, alpha=0.7)

    # Labels for reference lines
    ax.text(205, baseline + 0.4, f'Baseline ({baseline}%)',
            fontsize=7, color=C['grey'], ha='right', va='bottom')
    ax.text(205, fullseq_best + 0.4, f'Full-seq ({fullseq_best}%)',
            fontsize=7, color=C['green'], ha='right', va='bottom')

    ax.set_xlabel('Position limit')
    ax.set_ylabel('MATH-500 avg@4 (%)')
    ax.set_xlim(0, 210)
    ax.set_ylim(48, 69)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#cccccc',
              fontsize=7.5)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1b_performance_vs_pos.pdf'))
    plt.close(fig)
    print('  saved fig1b_performance_vs_pos.pdf')


# ======================================================================
# Fig 2: Training stability (two subplots)
# ======================================================================
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # --- Left: MATH-500 avg@4 over steps ---
    steps_math = [50, 100, 150, 200]
    pos100_math = [63.75, 64.45, 65.15, 65.85]
    fullseq_first = [65.55, 64.95, 65.55, 64.95]
    fullseq_last = [65.60, 46.30, 49.70, 43.80]

    ax1.plot(steps_math, pos100_math, '-o', color=C['blue'],
             label='Pos-100 ($n{=}1$)', markersize=5)
    ax1.plot(steps_math, fullseq_first, '-^', color=C['green'],
             label='Full-seq, first-box ($n{=}16$)', markersize=5)
    ax1.plot(steps_math, fullseq_last, '-v', color=C['red'],
             label='Full-seq, last-box ($n{=}16$)', markersize=5)

    ax1.set_xlabel('Training step')
    ax1.set_ylabel('MATH-500 avg@4 (%)')
    ax1.set_xlim(25, 225)
    ax1.set_ylim(38, 70)
    ax1.set_xticks(steps_math)
    ax1.legend(loc='lower left', frameon=True, fancybox=False,
               edgecolor='#cccccc', fontsize=7)
    ax1.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Math reasoning', fontsize=11, pad=8)

    # --- Right: HumanEval pass@1 over steps ---
    steps_code = [50, 100, 150, 200, 250, 300, 350, 400]
    he_pos50 =   [37.8, 39.0, 39.6, 41.5, 40.2, 40.9, 42.1, 40.9]
    he_pos100 =  [37.2, 39.0, 42.1, 37.8, 39.0, 37.8, 37.8, 38.4]
    he_fullseq = [40.2, 31.7, 32.3, 32.9, 27.4, 28.0, 26.8, 26.8]

    ax2.plot(steps_code, he_pos50, '-o', color=C['blue'],
             label='Pos-50', markersize=4)
    ax2.plot(steps_code, he_pos100, '-s', color=C['orange'],
             label='Pos-100', markersize=4)
    ax2.plot(steps_code, he_fullseq, '-v', color=C['red'],
             label='Full-seq', markersize=4)

    ax2.set_xlabel('Training step')
    ax2.set_ylabel('HumanEval pass@1 (%)')
    ax2.set_xlim(25, 425)
    ax2.set_ylim(24, 45)
    ax2.set_xticks([50, 100, 200, 300, 400])
    ax2.legend(loc='lower left', frameon=True, fancybox=False,
               edgecolor='#cccccc', fontsize=7.5)
    ax2.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(b) Code generation', fontsize=11, pad=8)

    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(OUT_DIR, 'fig2_stability.pdf'))
    plt.close(fig)
    print('  saved fig2_stability.pdf')


# ======================================================================
# Fig token composition: Stacked bar chart
# ======================================================================
def fig_token_composition():
    labels = ['0\u20135', '5\u201320', '20\u201350', '50\u2013100', '100\u2013200', '200\u2013500']

    planning      = [32.8, 20.1, 15.3, 11.2,  9.5,  7.9]
    structural    = [ 7.8, 15.2, 22.1, 28.5, 33.1, 39.3]
    math_number   = [ 1.7,  5.3,  8.9, 13.2, 15.8, 18.1]
    math_operator = [ 0.8,  2.1,  3.5,  5.1,  5.8,  6.2]
    math_latex    = [ 3.2,  5.8,  7.2,  8.5,  9.1,  8.8]
    continuation  = [100 - (p + s + mn + mo + ml)
                     for p, s, mn, mo, ml in
                     zip(planning, structural, math_number, math_operator, math_latex)]

    x = np.arange(len(labels))
    width = 0.55

    # Vega-lite inspired palette
    cats = [
        ('Planning',        planning,      '#4c78a8'),
        ('Structural',      structural,    '#f58518'),
        ('Math (number)',   math_number,   '#e45756'),
        ('Math (operator)', math_operator, '#72b7b2'),
        ('Math (LaTeX)',    math_latex,    '#54a24b'),
        ('Continuation',    continuation,  '#b3b3b3'),
    ]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    bottom = np.zeros(len(labels))
    for name, vals, color in cats:
        ax.bar(x, vals, width, bottom=bottom, label=name, color=color,
               edgecolor='white', linewidth=0.4)
        bottom += np.array(vals)

    ax.set_xlabel('Position range')
    ax.set_ylabel('Token type (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=8, handlelength=1.2, handleheight=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_token_composition.pdf'))
    plt.close(fig)
    print('  saved fig_token_composition.pdf')


# ======================================================================
# Run all
# ======================================================================
if __name__ == '__main__':
    print('Generating figures...')
    fig1a()
    fig1b()
    fig2()
    fig_token_composition()
    print(f'Done. All figures saved to {OUT_DIR}')
