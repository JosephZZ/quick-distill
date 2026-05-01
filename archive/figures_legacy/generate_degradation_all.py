"""Generate degradation figure showing all experiments step 50-200."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

steps = [50, 100, 150, 200]

# === Panel 1: 1.7B Teacher MATH-500 ===
ax = axes[0]
# V1 verified data
pos50_math = [58.05, 60.65, 61.70, 60.85]
pos100_math = [60.45, 62.50, 63.15, 62.35]
fullseq_math = [61.30, 62.35, 37.75, 47.35]
baseline_math = 50.95

ax.plot(steps, pos50_math, 'o-', color='#4CAF50', linewidth=2, markersize=5, label='Pos-50')
ax.plot(steps, pos100_math, 's-', color='#2196F3', linewidth=2, markersize=5, label='Pos-100')
ax.plot(steps, fullseq_math, 'D-', color='#FF9800', linewidth=2, markersize=5, label='Full-seq')
ax.axhline(y=baseline_math, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(205, baseline_math + 0.5, 'Baseline', fontsize=7, color='gray')
ax.set_xlabel('Training Step')
ax.set_ylabel('MATH-500 avg@4 (%)')
ax.set_title('(a) 1.7B Teacher — Math')
ax.set_ylim(30, 68)
ax.set_xticks(steps)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3)

# === Panel 2: 4B Teacher MATH-500 ===
ax = axes[1]
pos50_4b = [57.05, 61.40, 61.70, 61.00]
pos100_4b = [59.95, 62.00, 64.05, 64.20]
fullseq_4b = [62.70, 64.15, 65.00, 45.75]  # s50,s100,s150,s200

ax.plot(steps, pos50_4b, 'o-', color='#4CAF50', linewidth=2, markersize=5, label='Pos-50')
ax.plot(steps, pos100_4b, 's-', color='#2196F3', linewidth=2, markersize=5, label='Pos-100')
ax.plot(steps, fullseq_4b, 'D-', color='#FF9800', linewidth=2, markersize=5, label='Full-seq')
ax.axhline(y=baseline_math, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(205, baseline_math + 0.5, 'Baseline', fontsize=7, color='gray')
ax.set_xlabel('Training Step')
ax.set_title('(b) 4B Teacher — Math')
ax.set_ylim(30, 68)
ax.set_xticks(steps)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3)

# === Panel 3: 1.7B Teacher HumanEval ===
ax = axes[2]
pos50_he = [36.59, 37.80]  # only 2 steps
pos100_he = [37.20, 39.02, 39.02, 39.63]
fullseq_he_steps = [50, 100, 150]
fullseq_he = [39.02, 35.37, 32.32]
baseline_he = 32.93

ax.plot([50, 100], pos50_he, 'o-', color='#4CAF50', linewidth=2, markersize=5, label='Pos-50')
ax.plot(steps, pos100_he, 's-', color='#2196F3', linewidth=2, markersize=5, label='Pos-100')
ax.plot(fullseq_he_steps, fullseq_he, 'D-', color='#FF9800', linewidth=2, markersize=5, label='Full-seq')
ax.axhline(y=baseline_he, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(205, baseline_he + 0.3, 'Baseline', fontsize=7, color='gray')
ax.set_xlabel('Training Step')
ax.set_ylabel('HumanEval pass@1 (%)')
ax.set_title('(c) 1.7B Teacher — Coding')
ax.set_ylim(25, 45)
ax.set_xticks(steps)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fig_degradation_all.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_degradation_all.png', bbox_inches='tight', dpi=300)
print("Saved fig_degradation_all")
plt.close()
