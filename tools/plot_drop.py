import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

categories = ['Irrelevant\n($\mathcal{F}^\circ, \leq \epsilon$)', 
              'Frozen\n($H \cup \mathcal{N}, =0$)', 
              'Critical Overlap\n($S_{share\_crit}$)']

unconstrained_vals = [0.15, 0.0, 0.75] 
soft_masked_vals = [0.15, 0.0, 0.25]    
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

rects1 = ax.bar(x - width/2, unconstrained_vals, width, label='Unconstrained Unlearning', color='gray', alpha=0.7, edgecolor='black')
rects2 = ax.bar(x + width/2, soft_masked_vals, width, label='Soft-Masked (Ours)', color='royalblue', alpha=0.9, edgecolor='black')

worst_drop_unc = sum(unconstrained_vals)
worst_drop_soft = sum(soft_masked_vals)
ax.axhline(worst_drop_unc, color='gray', linestyle='--', linewidth=1.5, zorder=0)
ax.text(-0.4, worst_drop_unc + 0.02, 'WorstDrop (Unconstrained)', color='gray', fontweight='bold')

ax.axhline(worst_drop_soft, color='royalblue', linestyle='--', linewidth=1.5, zorder=0)
ax.text(-0.4, worst_drop_soft + 0.02, 'WorstDrop (Soft-Masked)', color='royalblue', fontweight='bold')

ax.annotate('', xy=(2 + width/2, 0.27), xytext=(2 - width/2, 0.73),
            arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))
ax.text(2, 0.5, r'$\times (1-p)$', color='red', fontsize=14, fontweight='bold', ha='center')

ax.set_ylabel('Contribution to WorstDrop Error', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight='bold')
ax.legend(loc='upper right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('drop_decomposition.pdf', format='pdf', dpi=300)
plt.show()
