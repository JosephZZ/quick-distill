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
# Figure 3: Training on different positions
# ============================================================
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 3.5))

methods = ['First 100\n(prefix)', 'Random\n100', 'Full-seq\n(all)', 'Last 100\n(tail)', 'Middle 100\n(center)']
avg4 = [65.85, 63.05, 62.35, 50.35, 47.80]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#F44336']
bars = ax3.bar(methods, avg4, color=colors, width=0.6, edgecolor='white', linewidth=1.5)

ax3.axhline(y=50.95, color='gray', linestyle='--', linewidth=1.5, label='Baseline (no distill)')
ax3.text(4.3, 51.5, 'Baseline\n50.95%', fontsize=8, color='gray', ha='right')

for bar, val in zip(bars, avg4):
    ax3.text(bar.get_x() + bar.get_width()/2., val + 0.5, f'{val:.1f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_ylabel('MATH-500 avg@4 (%)')
ax3.set_title('(a) Training on Different Position Regions (k=100)')
ax3.set_ylim(40, 70)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('paper/figures/fig_analysis_3_positions.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_analysis_3_positions.png', bbox_inches='tight', dpi=300)
print("Saved fig_analysis_3_positions")
plt.close()


# ============================================================
# Figure 4: Degradation over training
# ============================================================
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 3.5))

# (a) Math
steps = [50, 100, 150, 200]
pos50_math = [62.35, 66.05, 66.65, 64.85]
pos100_math = [63.75, 64.45, 65.15, 65.85]
fullseq_math = [61.00, 62.00, 62.35, 61.20]

ax4a.plot(steps, pos50_math, 'o-', color='#2196F3', linewidth=2, markersize=6, label='Pos-50')
ax4a.plot(steps, pos100_math, 's-', color='#4CAF50', linewidth=2, markersize=6, label='Pos-100')
ax4a.plot(steps, fullseq_math, 'D-', color='#FF9800', linewidth=2, markersize=6, label='Full-seq')
ax4a.axhline(y=50.95, color='gray', linestyle=':', alpha=0.5)
ax4a.text(205, 51.5, 'Baseline', fontsize=8, color='gray')
ax4a.set_xlabel('Training Step')
ax4a.set_ylabel('MATH-500 avg@4 (%)')
ax4a.set_title('(a) Math: Stable Training')
ax4a.legend(fontsize=8)
ax4a.set_ylim(48, 70)
ax4a.grid(True, alpha=0.3)

# (b) Coding - degradation
steps_coding = [50, 100, 150, 200, 250, 300, 350, 400]
pos50_he = [37.8, 39.0, 39.6, 41.5, 40.2, 40.9, 42.1, 40.9]
pos100_he = [37.2, 39.0, 42.1, 37.8, 39.0, 37.8, 37.8, 38.4]
fullseq_he = [40.2, 31.7, 32.3, 32.9, 27.4, 28.0, 26.8, 26.8]

ax4b.plot(steps_coding, pos50_he, 'o-', color='#2196F3', linewidth=2, markersize=5, label='Pos-50')
ax4b.plot(steps_coding, pos100_he, 's-', color='#4CAF50', linewidth=2, markersize=5, label='Pos-100')
ax4b.plot(steps_coding, fullseq_he, 'D-', color='#FF9800', linewidth=2, markersize=5, label='Full-seq')
ax4b.axhline(y=32.93, color='gray', linestyle=':', alpha=0.5)
ax4b.text(405, 33.5, 'Baseline', fontsize=8, color='gray')

# Annotate the degradation
ax4b.annotate('Degradation\n(-33%)', xy=(350, 26.8), xytext=(280, 22),
              fontsize=8, color='#FF9800', fontweight='bold',
              arrowprops=dict(arrowstyle='->', color='#FF9800'))

ax4b.set_xlabel('Training Step')
ax4b.set_ylabel('HumanEval pass@1 (%)')
ax4b.set_title('(b) Coding: Full-seq Degrades')
ax4b.legend(fontsize=8)
ax4b.set_ylim(20, 48)
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fig_analysis_4_degradation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_analysis_4_degradation.png', bbox_inches='tight', dpi=300)
print("Saved fig_analysis_4_degradation")
plt.close()

print("\nAll figures generated!")
