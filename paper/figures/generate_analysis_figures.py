"""Generate the 4 analysis figures for the paper."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

# ============================================================
# Figure 1: Signal Distribution (KL + Entropy/Confidence)
# ============================================================
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(10, 3.5))

# (a) KL by position
positions = [0, 1, 2, 3, 4, 10, 20, 50, 100, 150, 192]
kl_values = [3.057, 1.585, 1.161, 0.950, 0.921, 0.934, 0.896, 0.525, 0.449, 0.383, 0.359]
ax1a.plot(positions, kl_values, 'o-', color='#D32F2F', linewidth=2, markersize=4)
ax1a.fill_between(positions, kl_values, alpha=0.15, color='#D32F2F')
ax1a.set_xlabel('Token Position')
ax1a.set_ylabel('Mean KL Divergence')
ax1a.set_title('(a) KL Divergence by Position')
ax1a.set_xlim(-5, 200)
ax1a.grid(True, alpha=0.3)
ax1a.axhline(y=0.4, color='gray', linestyle=':', alpha=0.5)
ax1a.text(120, 0.5, 'Low-signal plateau', fontsize=8, color='gray')

# (b) Teacher confidence metrics by position
pos_ranges = ['0-50', '50-100', '100-200', '200-400', '400-800']
pos_x = [25, 75, 150, 300, 600]
entropy = [1.069, 0.430, 0.248, 0.196, 0.145]
agreement = [58.4, 70.7, 80.1, 83.2, 87.0]
top1 = [72.4, 86.4, 91.9, 93.5, 95.0]

ax1b_twin = ax1b.twinx()
l1, = ax1b.plot(pos_x, entropy, 's-', color='#1565C0', linewidth=2, markersize=5, label='Teacher entropy')
l2, = ax1b_twin.plot(pos_x, agreement, 'D-', color='#2E7D32', linewidth=2, markersize=5, label='Agreement rate (%)')
ax1b.set_xlabel('Token Position')
ax1b.set_ylabel('Teacher Entropy (nats)', color='#1565C0')
ax1b_twin.set_ylabel('Agreement Rate (%)', color='#2E7D32')
ax1b.set_title('(b) Teacher Confidence by Position')
ax1b.tick_params(axis='y', labelcolor='#1565C0')
ax1b_twin.tick_params(axis='y', labelcolor='#2E7D32')
ax1b.set_xlim(0, 650)
ax1b_twin.set_ylim(50, 95)
lines = [l1, l2]
ax1b.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=8)
ax1b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fig_analysis_1_signal.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_analysis_1_signal.png', bbox_inches='tight', dpi=300)
print("Saved fig_analysis_1_signal")
plt.close()


# ============================================================
# Figure 2: Continuation experiment
# Already exists as fig_continuation.pdf - reuse
# ============================================================
# (Reuse existing fig_continuation.pdf alongside the example in LaTeX)


# ============================================================
# Figure 4: Coding degradation only (single panel)
# ============================================================
fig3, ax3 = plt.subplots(1, 1, figsize=(5.5, 3.5))

steps_coding = [50, 100, 150, 200, 250, 300, 350, 400]
pos100_he = [37.2, 39.0, 42.1, 37.8, 39.0, 37.8, 37.8, 38.4]
fullseq_he = [40.2, 31.7, 32.3, 32.9, 27.4, 28.0, 26.8, 26.8]

ax3.plot(steps_coding, pos100_he, 'o-', color='#2196F3', linewidth=2, markersize=5, label='First 100')
ax3.plot(steps_coding, fullseq_he, 'D-', color='#FF9800', linewidth=2, markersize=5, label='Full-seq')
ax3.axhline(y=32.93, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.text(405, 33.5, 'Baseline', fontsize=8, color='gray')

ax3.annotate('$-$33%', xy=(350, 26.8), xytext=(280, 22),
             fontsize=9, color='#FF9800', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#FF9800'))

ax3.set_xlabel('Training Step')
ax3.set_ylabel('HumanEval pass@1 (%)')
ax3.legend(fontsize=9)
ax3.set_ylim(20, 48)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fig_analysis_3_positions.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_analysis_3_positions.png', bbox_inches='tight', dpi=300)
print("Saved fig_analysis_3_positions")
plt.close()


# ============================================================
# Figure: Position limit sweep curve
# ============================================================
fig_sweep, ax_sw = plt.subplots(1, 1, figsize=(5.5, 3.5))

# n16bs16 data (best avg@4 across steps, consistent config)
N_values = [5, 10, 20, 50, 100, 150, 200]
best_avg4 = [56.50, 59.50, 60.20, 62.45, 64.25, 65.70, 66.75]

ax_sw.plot(N_values, best_avg4, 'o-', color='#2196F3', linewidth=2, markersize=7, zorder=3)

# Baseline and full-seq reference lines
ax_sw.axhline(y=50.95, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax_sw.text(155, 51.6, 'Baseline (50.95%)', fontsize=8, color='gray')

ax_sw.axhline(y=65.55, color='#FF9800', linestyle='--', linewidth=1.5, alpha=0.7)
ax_sw.text(155, 66.1, 'Full-seq (65.55%)', fontsize=8, color='#FF9800')

ax_sw.set_xlabel('Position Limit $N$')
ax_sw.set_ylabel('MATH-500 avg@4 (%)')
ax_sw.set_xticks(N_values)
ax_sw.set_xticklabels([str(n) for n in N_values])
ax_sw.set_ylim(48, 70)
ax_sw.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fig_pos_sweep.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_pos_sweep.png', bbox_inches='tight', dpi=300)
print("Saved fig_pos_sweep")
plt.close()


# ============================================================
# Figure: Token selection strategy histogram
# ============================================================
fig_tok, ax_tok = plt.subplots(1, 1, figsize=(6, 3.5))

strategies = ['Prefix\n(first 100)', 'Random\n100', 'Top-Ent\nTeacher', 'Top-Ent\nStudent', 'Full-seq', 'Top-KL\n100']
avg4_values = [65.85, 63.05, 62.20, 61.35, 62.35, 58.60]
colors_tok = ['#2196F3', '#4CAF50', '#FF9800', '#FFC107', '#9E9E9E', '#F44336']

bars_tok = ax_tok.bar(range(len(strategies)), avg4_values, color=colors_tok, width=0.65,
                       edgecolor='white', linewidth=0.5)
ax_tok.set_xticks(range(len(strategies)))
ax_tok.set_xticklabels(strategies, fontsize=8)

ax_tok.axhline(y=50.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax_tok.text(5.5, 51.5, 'Baseline', fontsize=8, color='gray')

for bar, val in zip(bars_tok, avg4_values):
    ax_tok.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax_tok.set_ylabel('MATH-500 avg@4 (%)')
ax_tok.set_ylim(50, 69)
ax_tok.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('paper/figures/fig_token_select.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_token_select.png', bbox_inches='tight', dpi=300)
print("Saved fig_token_select")
plt.close()


# ============================================================
# Figure 4: Continuation + Position region bar chart
# ============================================================
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 3.0),
                                   gridspec_kw={'wspace': 0.35})

# --- Left: Continuation (student prefix → teacher continues only) ---
prefix_lengths = [0, 100, 200, 300]
student_prefix_teacher_cont = [65.30, 62.70, 56.30, 51.75]

ax4a.plot(prefix_lengths, student_prefix_teacher_cont, 'o-', color='#2196F3',
          linewidth=2, markersize=7, zorder=3)

ax4a.axhline(y=65.30, color='#2196F3', linestyle=':', alpha=0.5, linewidth=1)
ax4a.text(305, 65.80, 'Teacher\nbaseline', fontsize=8, color='#2196F3', alpha=0.7)
ax4a.axhline(y=50.95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4a.text(305, 51.45, 'Student\nbaseline', fontsize=8, color='gray', alpha=0.7)

ax4a.set_xlabel('Student prefix length (tokens)')
ax4a.set_ylabel('MATH-500 avg@4 (%)')
ax4a.set_xticks([0, 100, 200, 300])
ax4a.set_xlim(-10, 380)
ax4a.set_ylim(42, 70)
ax4a.grid(True, alpha=0.3)

# --- Right: Bar chart of best checkpoint values per strategy ---
strategies = ['First\n100', 'Random\n100', 'Full-\nseq', 'Base-\nline', 'Last\n100', 'Middle\n100']
best_values = [65.85, 63.05, 62.35, 50.95, 50.35, 47.80]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E', '#9C27B0', '#F44336']

bars = ax4b.bar(range(len(strategies)), best_values, color=colors, width=0.7, edgecolor='white', linewidth=0.5)
ax4b.set_xticks(range(len(strategies)))
ax4b.set_xticklabels(strategies, fontsize=8)

ax4b.axhline(y=50.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)

for bar, val in zip(bars, best_values):
    ax4b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax4b.set_ylabel('MATH-500 avg@4 (%)')
ax4b.set_ylim(42, 70)
ax4b.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('paper/figures/fig_analysis_4_degradation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_analysis_4_degradation.png', bbox_inches='tight', dpi=300)
print("Saved fig_analysis_4_degradation")
plt.close()

print("\nAll figures generated!")
