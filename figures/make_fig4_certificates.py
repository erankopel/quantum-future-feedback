"""Regenerates Fig. 4 (nested analytic certificates) with the drift-corrected
information threshold. Half-widths in units of pi, truncated conservatively."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

r = {'NPT': 0.01632, 'Stability': 0.00760, 'Coherence': 0.00365,
     'Information (with input drift)': 0.00150, 'Combined chosen': 0.00130}
colors = {'NPT': '#4472C4', 'Stability': '#ED7D31', 'Coherence': '#A5427A',
          'Information (with input drift)': '#70AD47', 'Combined chosen': '#FFC000'}
fig, ax = plt.subplots(figsize=(8, 6.2))
for name, q in r.items():
    ax.add_patch(Rectangle((-q, -q), 2*q, 2*q, fill=False, lw=2.5,
                           ec=colors[name], label=f'{name}: {q:.5f} pi'))
ax.scatter([0], [0], c='black', s=35, zorder=5)
ax.set_xlim(-.018, .018); ax.set_ylim(-.018, .018); ax.set_aspect('equal')
ax.grid(alpha=.2)
ax.set_xlabel('(theta - theta0) / pi'); ax.set_ylabel('(phi - phi0) / pi')
ax.set_title('Nested analytic certificates in the fixed (theta, phi) slice',
             fontweight='bold', color='#173F5F')
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
fig.savefig('fig4_certificates.png', dpi=300, bbox_inches='tight', facecolor='white')
