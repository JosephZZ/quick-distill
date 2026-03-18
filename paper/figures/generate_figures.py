#!/usr/bin/env python3
"""Generate publication-quality figures for DFT-Distill paper."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# Global style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'lines.linewidth': 1.5,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

# ============================================================
# Data
# ============================================================

# Per-position KL data (combining 2k and 59k trajectory data)
# Positions 0-29 from 2k trajectories (detailed), rest from 59k
kl_positions_detailed = {
    0: 3.057, 1: 1.979, 2: 1.104, 3: 1.510, 4: 0.921,
    5: 0.962, 6: 0.946, 7: 0.846, 8: 0.853, 9: 0.936,
    10: 1.036, 11: 1.007, 12: 0.843, 13: 1.007, 14: 0.839,
    15: 1.027, 16: 0.764, 17: 1.011, 18: 0.987, 19: 0.943,
    20: 1.031, 21: 1.018, 22: 0.891, 23: 0.954, 24: 0.946,
    25: 1.050, 26: 1.083, 27: 1.095, 28: 1.064, 29: 1.207,
}

# Extended data from 59k trajectories (approximate values from table)
kl_positions_extended = {
    30: 1.1, 40: 0.9, 50: 0.693, 60: 0.599, 70: 0.535,
    75: 0.507, 80: 0.479, 90: 0.454, 100: 0.45, 120: 0.43,
    140: 0.42, 150: 0.410, 160: 0.396, 170: 0.386, 175: 0.374,
    180: 0.375, 190: 0.366, 192: 0.359,
}

# Merge and interpolate
all_pos = {}
all_pos.update(kl_positions_detailed)
all_pos.update(kl_positions_extended)

positions = sorted(all_pos.keys())
kl_values = [all_pos[p] for p in positions]

# Interpolate for smooth curve
positions_fine = np.arange(0, 193)
kl_interp = np.interp(positions_fine, positions, kl_values)

# Performance data
pos_limits = [5, 10, 20, 50, 100, 200]
avg4_scores = [56.50, 59.50, 60.20, 62.45, 64.25, 66.75]
kl_pct = [5.2, 7.8, 12.2, 26.2, 43.9, 66.1]
baseline = 50.95
fullseq = 65.55

# Phase boundaries
phases = [
    (0, 4, 'Ultra-early'),
    (5, 19, 'Early'),
    (20, 49, 'Mid-early'),
    (50, 99, 'Mid'),
    (100, 149, 'Mid-late'),
    (150, 199, 'Late'),
]

# Phase-wise mean KL
phase_mean_kl = {
    'Ultra-early': 1.714,
    'Early': 0.934,
    'Mid-early': 0.896,
    'Mid': 0.525,
    'Mid-late': 0.449,
    'Late': 0.383,
}


# ============================================================
# Color palette
# ============================================================
C_BLUE = '#2171b5'
C_BLUE_LIGHT = '#c6dbef'
C_RED = '#cb181d'
C_RED_LIGHT = '#fcbba1'
C_GREEN = '#238b45'
C_ORANGE = '#d94801'
C_GRAY = '#636363'
C_GRAY_LIGHT = '#bdbdbd'

phase_colors = ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26', '#a50f15']
phase_colors_soft = ['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000', '#7f0000']
# Softer academic palette for phase backgrounds
phase_bg = ['#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486']
phase_bg_alpha = 0.12


def fig1a_kl_curve():
    """Figure 1a: Per-position KL divergence curve."""
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    # Shade 0-50 region
    mask_50 = positions_fine <= 50
    ax.fill_between(positions_fine[mask_50], kl_interp[mask_50], alpha=0.20,
                    color=C_BLUE, label=None, zorder=1)
    ax.fill_between(positions_fine[mask_50], kl_interp[mask_50], alpha=0.0,
                    color=C_BLUE, zorder=1)  # invisible, for clean edge

    # Plot the KL curve
    ax.plot(positions_fine, kl_interp, color=C_BLUE, linewidth=1.8, zorder=3)

    # Scatter the actual data points (small)
    ax.scatter(positions[:30], kl_values[:30], s=8, color=C_BLUE, zorder=4, alpha=0.6)

    # Annotate the shaded region
    ax.annotate('26% of total KL',
                xy=(25, 0.5), fontsize=8.5, color=C_BLUE,
                ha='center', style='italic')

    # Annotate position 0
    ax.annotate(f'pos 0: {all_pos[0]:.2f}',
                xy=(0, all_pos[0]), xytext=(18, 2.85),
                fontsize=8, color=C_GRAY,
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=0.8),
                ha='left')

    ax.set_xlabel('Response token position')
    ax.set_ylabel('Mean KL divergence')
    ax.set_xlim(-3, 200)
    ax.set_ylim(0, 3.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'/CGLab/ziheng/projects/dft-distill/paper/figures/fig1a_kl_curve.{ext}')
    plt.close(fig)
    print('Saved fig1a_kl_curve')


def fig1b_performance_vs_pos():
    """Figure 1b: Performance vs position limit."""
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    # Bar chart
    x = np.arange(len(pos_limits))
    bar_width = 0.6
    bars = ax.bar(x, avg4_scores, width=bar_width, color=C_BLUE, alpha=0.85,
                  edgecolor='white', linewidth=0.5, zorder=3)

    # Baseline and full-seq lines
    ax.axhline(y=baseline, color=C_GRAY, linestyle='--', linewidth=1.0, zorder=2)
    ax.axhline(y=fullseq, color=C_RED, linestyle='--', linewidth=1.0, zorder=2)

    ax.text(len(pos_limits) - 0.5, baseline + 0.4, 'Baseline (no distill)',
            fontsize=7.5, color=C_GRAY, ha='right', va='bottom')
    ax.text(len(pos_limits) - 0.5, fullseq + 0.4, 'Full-seq',
            fontsize=7.5, color=C_RED, ha='right', va='bottom')

    # Annotate KL % on top of bars
    for i, (bar, pct) in enumerate(zip(bars, kl_pct)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                f'{pct}%', ha='center', va='bottom', fontsize=7.5, color=C_GRAY)

    # Label "% of KL" above annotations
    ax.text(x[-1], avg4_scores[-1] + 1.8, '(% of KL signal)',
            ha='center', va='bottom', fontsize=7, color=C_GRAY, style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pos_limits])
    ax.set_xlabel('Position limit')
    ax.set_ylabel('Best avg@4 accuracy (%)')
    ax.set_ylim(48, 70)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'/CGLab/ziheng/projects/dft-distill/paper/figures/fig1b_performance_vs_pos.{ext}')
    plt.close(fig)
    print('Saved fig1b_performance_vs_pos')


def fig3_detailed_kl():
    """Figure 3: Detailed per-position KL with phase annotations."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    # Draw phase backgrounds with subtle shading
    phase_bg_colors = ['#e0e0e0', '#d0d0d0', '#c0c0c0', '#b0b0b0', '#a0a0a0', '#909090']
    for i, (start, end, name) in enumerate(phases):
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.08,
                   color=phase_bg_colors[i], zorder=0)

    # Draw phase boundary lines
    for start, end, name in phases[1:]:
        ax.axvline(x=start - 0.5, color=C_GRAY_LIGHT, linewidth=0.6,
                   linestyle=':', zorder=1)

    # Phase labels at the top of the plot
    phase_label_y = 3.38
    for i, (start, end, name) in enumerate(phases):
        mid = (start + end) / 2
        if name == 'Ultra-early':
            label = 'Ultra-\nearly'
            fontsize = 6
        elif name == 'Early':
            label = 'Early'
            fontsize = 6.5
        else:
            label = name
            fontsize = 7
        ax.text(mid, phase_label_y, label, ha='center', va='top', fontsize=fontsize,
                color='#666666', fontstyle='italic')

    # Plot the KL curve
    ax.plot(positions_fine, kl_interp, color=C_BLUE, linewidth=1.8, zorder=3)
    ax.scatter(positions[:30], kl_values[:30], s=10, color=C_BLUE, zorder=4, alpha=0.5)

    # Phase mean KL as horizontal segments
    for i, (start, end, name) in enumerate(phases):
        mean_kl = phase_mean_kl[name]
        ax.plot([start, end], [mean_kl, mean_kl], color=C_RED, linewidth=1.2,
                linestyle='-', alpha=0.5, zorder=2)
        # Label mean - position carefully to avoid overlap
        if name == 'Ultra-early':
            ax.text(end + 2, mean_kl + 0.08, f'{mean_kl:.2f}', fontsize=7,
                    color=C_RED, ha='left', va='bottom', alpha=0.8)
        elif name == 'Early':
            ax.text(end + 2, mean_kl - 0.06, f'{mean_kl:.2f}', fontsize=7,
                    color=C_RED, ha='left', va='top', alpha=0.8)
        elif name == 'Mid-early':
            ax.text(end + 2, mean_kl + 0.05, f'{mean_kl:.2f}', fontsize=7,
                    color=C_RED, ha='left', va='bottom', alpha=0.8)
        else:
            ax.text(end + 2, mean_kl + 0.03, f'{mean_kl:.2f}', fontsize=7,
                    color=C_RED, ha='left', va='bottom', alpha=0.8)

    # Annotate key points
    ax.annotate(f'pos 0: {all_pos[0]:.2f}',
                xy=(0, all_pos[0]), xytext=(35, 2.7),
                fontsize=8.5, color=C_GRAY,
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=0.8))

    ax.annotate(f'pos 3: {all_pos[3]:.2f}',
                xy=(3, all_pos[3]), xytext=(40, 1.9),
                fontsize=8.5, color=C_GRAY,
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=0.8))

    ax.set_xlabel('Response token position')
    ax.set_ylabel('Mean KL divergence')
    ax.set_xlim(-3, 205)
    ax.set_ylim(0, 3.5)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend for phase mean
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_BLUE, linewidth=1.8, label='Per-position KL'),
        Line2D([0], [0], color=C_RED, linewidth=1.2, alpha=0.5, label='Phase mean KL'),
    ]
    ax.legend(handles=legend_elements, loc='center right', frameon=True,
              framealpha=0.9, edgecolor='#cccccc', fontsize=8,
              bbox_to_anchor=(0.98, 0.6))

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'/CGLab/ziheng/projects/dft-distill/paper/figures/fig3_detailed_kl.{ext}')
    plt.close(fig)
    print('Saved fig3_detailed_kl')


if __name__ == '__main__':
    fig1a_kl_curve()
    fig1b_performance_vs_pos()
    fig3_detailed_kl()
    print('All figures generated.')
