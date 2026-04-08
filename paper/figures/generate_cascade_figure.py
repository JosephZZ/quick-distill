"""Generate cascade effect figure: two KL curves (raw vs pos-200 distilled)."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

# Data from kl_position_analysis.md (100 problems x 1 trajectory)
# Using bin centers for smooth curve
bin_centers = [25, 75, 125, 175, 250, 350, 450, 550, 650]
raw_kl =     [2.064, 0.759, 0.495, 0.387, 0.382, 0.331, 0.317, 0.260, 0.252]
pos200_kl =  [1.015, 0.321, 0.246, 0.237, 0.238, 0.216, 0.289, 0.259, 0.243]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))

# Plot curves
ax.plot(bin_centers, raw_kl, 'o-', color='#9E9E9E', linewidth=2, markersize=5,
        label='Before distillation', zorder=3)
ax.plot(bin_centers, pos200_kl, 's-', color='#2196F3', linewidth=2, markersize=5,
        label='After pos-200 distillation', zorder=3)

# Fill between to show reduction
ax.fill_between(bin_centers, raw_kl, pos200_kl, alpha=0.15, color='#2196F3')

# Vertical line at position 200 (training boundary)
ax.axvline(x=200, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(185, 1.85, 'Training\nboundary', fontsize=8, color='#D32F2F',
        ha='right', va='top', fontweight='bold')

# Annotate trained vs untrained regions
ax.annotate('', xy=(10, 2.2), xytext=(195, 2.2),
            arrowprops=dict(arrowstyle='<->', color='#2196F3', lw=1.5))
ax.text(100, 2.28, 'Trained positions', fontsize=8, color='#2196F3',
        ha='center', fontweight='bold')

ax.annotate('', xy=(205, 2.2), xytext=(690, 2.2),
            arrowprops=dict(arrowstyle='<->', color='#FF9800', lw=1.5))
ax.text(450, 2.28, 'Untrained positions', fontsize=8, color='#FF9800',
        ha='center', fontweight='bold')

# Add reduction % annotations at key points
for i, (x, raw, dist) in enumerate(zip(bin_centers, raw_kl, pos200_kl)):
    if raw > 0 and x <= 400:
        pct = (raw - dist) / raw * 100
        y_mid = (raw + dist) / 2
        if x <= 200:
            ax.text(x + 8, y_mid, f'−{pct:.0f}%', fontsize=7, color='#1565C0',
                    va='center', fontweight='bold')
        else:
            ax.text(x + 8, y_mid, f'−{pct:.0f}%', fontsize=7, color='#E65100',
                    va='center', fontweight='bold')

ax.set_xlabel('Token Position')
ax.set_ylabel('Mean KL Divergence')
ax.set_xlim(0, 700)
ax.set_ylim(0, 2.5)
ax.legend(fontsize=9, loc='center right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fig_cascade.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_cascade.png', bbox_inches='tight', dpi=300)
print("Saved fig_cascade")
