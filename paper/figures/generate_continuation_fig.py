"""Generate Figure: Continuation experiment results."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

# Data from prefix_continuation_analysis.md
prefix_lengths = [0, 100, 200, 300]

# Student prefix → Teacher continuation
student_prefix_teacher_cont = [65.30, 62.70, 56.30, 51.75]  # 0 = teacher baseline

# Teacher prefix → Student continuation
teacher_prefix_student_cont = [50.95, 47.00, 47.45, 52.65]  # 0 = student baseline

ax.plot(prefix_lengths, student_prefix_teacher_cont, 'o-', color='#2196F3',
        linewidth=2, markersize=7, label='Student prefix → Teacher continues', zorder=3)
ax.plot(prefix_lengths, teacher_prefix_student_cont, 's--', color='#FF9800',
        linewidth=2, markersize=7, label='Teacher prefix → Student continues', zorder=3)

# Baselines
ax.axhline(y=65.30, color='#2196F3', linestyle=':', alpha=0.5, linewidth=1)
ax.text(305, 65.80, 'Teacher baseline', fontsize=8, color='#2196F3', alpha=0.7)
ax.axhline(y=50.95, color='#FF9800', linestyle=':', alpha=0.5, linewidth=1)
ax.text(305, 51.45, 'Student baseline', fontsize=8, color='#FF9800', alpha=0.7)

ax.set_xlabel('Prefix length (tokens)')
ax.set_ylabel('MATH-500 avg@4 (%)')
ax.set_xticks([0, 100, 200, 300])
ax.set_xlim(-10, 380)
ax.set_ylim(42, 70)
ax.legend(loc='center right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fig_continuation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_continuation.png', bbox_inches='tight', dpi=300)
print("Saved fig_continuation.pdf/png")
