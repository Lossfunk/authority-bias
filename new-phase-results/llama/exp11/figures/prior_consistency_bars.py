import matplotlib.pyplot as plt
import numpy as np

# frac_dr_pos from each slice (point estimate)
# Prior-consistent % = (1 - frac_dr_pos) * 100

slices = ["All items\n(n=1813)", "Wrong prior\n(all)", "Wrong prior\n(low conf)", "Wrong prior\n(high conf,\ntop 25%)", "Wrong prior\n(top 10%)"]

# Expert frac_dr_pos: all_items, full_wrong, low_conf, high_conf_top25, top10
expert_fdr = [0.528, 0.447, 0.456, 0.500, 0.556]
# Note frac_dr_pos
note_fdr = [0.586, 0.383, 0.459, 0.154, 0.000]

expert_pc = [(1 - f) * 100 for f in expert_fdr]
note_pc = [(1 - f) * 100 for f in note_fdr]

# CI for frac_dr_pos (low, high) -> prior-consistent CI is (1-high, 1-low)
expert_fdr_ci = [
    (0.501, 0.554),  # all items
    (0.397, 0.497),  # full wrong
    (0.392, 0.520),  # low conf
    (0.367, 0.634),  # high conf top25
    (0.200, 0.875),  # top10
]
note_fdr_ci = [
    (0.556, 0.615),  # all items
    (0.333, 0.435),  # full wrong
    (0.393, 0.526),  # low conf
    (0.049, 0.276),  # high conf top25
    (0.000, 0.000),  # top10
]

expert_pc_lo = [(1 - ci[1]) * 100 for ci in expert_fdr_ci]
expert_pc_hi = [(1 - ci[0]) * 100 for ci in expert_fdr_ci]
note_pc_lo = [(1 - ci[1]) * 100 for ci in note_fdr_ci]
note_pc_hi = [(1 - ci[0]) * 100 for ci in note_fdr_ci]

expert_err_lo = [pc - lo for pc, lo in zip(expert_pc, expert_pc_lo)]
expert_err_hi = [hi - pc for pc, hi in zip(expert_pc, expert_pc_hi)]
note_err_lo = [pc - lo for pc, lo in zip(note_pc, note_pc_lo)]
note_err_hi = [hi - pc for pc, hi in zip(note_pc, note_pc_hi)]

x = np.arange(len(slices))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, expert_pc, width, label='Expert', color='#5B9BD5',
               yerr=[expert_err_lo, expert_err_hi], capsize=4, error_kw={'linewidth': 1.5})
bars2 = ax.bar(x + width/2, note_pc, width, label='Note', color='#C0504D',
               yerr=[note_err_lo, note_err_hi], capsize=4, error_kw={'linewidth': 1.5})

# Add percentage labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{height:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=11, color='#5B9BD5')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{height:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=11, color='#C0504D')

# 50% reference line
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax.text(len(slices) - 0.5, 51, '50% = coin flip', ha='right', va='bottom',
        fontsize=9, color='gray', style='italic')

ax.set_ylabel('% of items where instruction is prior-consistent\n(dr < 0: suppresses correct > wrong)', fontsize=11)
ax.set_title('Prior-Consistency Rate by Model Confidence Slice\n"Be Correct" Instruction — Llama-3.1-8B-Instruct', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(slices, fontsize=10)
ax.legend(fontsize=12, loc='upper left')
ax.set_ylim(0, 115)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/majortimberwolf/Projects/lossfunk/persona-vectors/new-phase-results/exp11/figures/prior_consistency_bars.png', dpi=150, bbox_inches='tight')
plt.savefig('/Users/majortimberwolf/Projects/lossfunk/persona-vectors/new-phase-results/exp11/figures/prior_consistency_bars.pdf', bbox_inches='tight')
print("Saved!")
