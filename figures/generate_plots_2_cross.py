"""Cross-system figures for Part 2 (DDI) — 2.0 / 2.1 / 2.2 / 2.3 head-to-head.

All numbers traceable to the three NOTES.md files. Champions:
    2.0 rule-based                       test M=26.9  m=20.8
    2.1 ML  two-stage on mod_best2       test M=66.8  m=62.5
    2.2 NN  mod9 rel-pos seed 777        test M=65.6  m=62.7
    2.3 LLM FT Llama r=32 prompts02      test M=39.8  m=40.9

Figures (PDF + PNG in report/):
    fig2_cross_system     test micro & macro F1 across the four systems
    fig2_cross_perclass   per-class test F1 across the three trained systems
    fig2_cross_devtest    devel vs test for the three champions
"""
import numpy as np
from generate_plots_2_common import (
    plt, save, CLASSES, CLASS_COLORS,
    C_BASE, C_ML, C_NN, C_LLAMA, C_BEST, C_GOOD, C_BAD, GREY)

SYS      = ['2.0\nrule-based', '2.1 ML\ntwo-stage', '2.2 NN\nrel-pos s777', '2.3 LLM\nLlama r=32 p02']
SYS_COL  = [C_BASE, C_ML, C_NN, C_LLAMA]
test_M   = [26.9, 66.8, 65.6, 39.8]
test_m   = [20.8, 62.5, 62.7, 40.9]

# =============================================================================
# 1. fig2_cross_system — test micro & macro F1 (two panels)
# =============================================================================
fig, (axM, axm) = plt.subplots(1, 2, figsize=(13, 5))
for ax, vals, title in [(axM, test_M, 'Test macro-F1 (M.avg)'),
                        (axm, test_m, 'Test micro-F1 (m.avg)')]:
    bars = ax.bar(SYS, vals, color=SYS_COL, edgecolor='white')
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.9, f'{v:.1f}', ha='center',
                fontsize=11, fontweight='bold' if v == max(vals) else 'normal',
                color=C_BEST if v == max(vals) else 'black')
    ax.set_ylabel('Test F1 (%)'); ax.set_ylim(0, 80)
    ax.set_title(title)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(9.5)

axM.annotate('ML ≈ NN at the top\n(within 1.2 pp)',
             xy=(1.5, 66.2), xytext=(1.5, 76), ha='center', fontsize=9.5,
             color=C_GOOD, fontweight='bold')
axM.annotate('LLM trails by ~27 pp\n(85% null + relational task)',
             xy=(3, 39.8), xytext=(3, 52), ha='center', fontsize=9,
             color=C_BAD, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_BAD, alpha=0.7))

fig.suptitle('Cross-system comparison on the DDI test set — '
             'classic ML & NN beat the fine-tuned 3B LLM (opposite of NER in Part 1)',
             fontsize=12.5, y=1.02)
save(fig, 'fig2_cross_system')


# =============================================================================
# 2. fig2_cross_perclass — per-class test F1 across the three trained systems
# =============================================================================
perclass = {
    '2.1 ML (two-stage)'     : {'advise': 64.8, 'effect': 63.0, 'int': 80.0, 'mechanism': 59.3},
    '2.2 NN (rel-pos s777)'  : {'advise': 62.4, 'effect': 63.0, 'int': 76.1, 'mechanism': 60.7},
    '2.3 LLM (Llama r=32 p02)': {'advise': 45.4, 'effect': 37.9, 'int': 33.3, 'mechanism': 42.6},
}
syscolors = [C_ML, C_NN, C_LLAMA]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(CLASSES)); w = 0.27
for i, (name, data) in enumerate(perclass.items()):
    vals = [data[c] for c in CLASSES]
    ax.bar(x + (i-1)*w, vals, w, color=syscolors[i], edgecolor='white', label=name)
    for j, v in enumerate(vals):
        best = max(perclass[s][CLASSES[j]] for s in perclass)
        ax.text(x[j]+(i-1)*w, v+1.0, f'{v:.1f}', ha='center', fontsize=8.5,
                fontweight='bold' if v == best else 'normal')
ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel('Test F1 (%)')
ax.set_title('Per-class test F1 across systems — ML/NN dominate every class;\n'
             'the LLM is most competitive on advise & mechanism, weakest on the rare int',
             fontsize=11)
ax.set_ylim(0, 95)
ax.legend(loc='upper right', fontsize=9)
save(fig, 'fig2_cross_perclass')


# =============================================================================
# 3. fig2_cross_devtest — devel vs test for the three champions (generalisation)
# =============================================================================
champs = ['2.1 ML\ntwo-stage', '2.2 NN\nrel-pos s777', '2.3 LLM\nLlama r=32 p02']
dev_M  = [65.9, 64.7, 41.6]
tst_M  = [66.8, 65.6, 39.8]
ccol   = [C_ML, C_NN, C_LLAMA]

fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(champs)); w = 0.36
ax.bar(x-w/2, dev_M, w, color=ccol, alpha=0.55, edgecolor='white', label='devel')
ax.bar(x+w/2, tst_M, w, color=ccol, edgecolor='white', label='test')
for i in range(len(champs)):
    ax.text(i-w/2, dev_M[i]+0.7, f'{dev_M[i]:.1f}', ha='center', fontsize=9.5)
    ax.text(i+w/2, tst_M[i]+0.7, f'{tst_M[i]:.1f}', ha='center', fontsize=9.5, fontweight='bold')
    d = tst_M[i]-dev_M[i]
    ax.annotate(f'Δ={d:+.1f}', xy=(i, max(dev_M[i], tst_M[i])+3), ha='center',
                fontsize=9, color=C_GOOD if d >= 0 else C_BAD, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(champs, fontsize=10)
ax.set_ylabel('macro-F1 (%)')
ax.set_title('Champion generalisation — devel → test\n'
             'ML & NN generalise positively; the LLM drops slightly but holds rank',
             fontsize=11)
ax.set_ylim(0, 78)
ax.legend(loc='upper right', fontsize=9)
save(fig, 'fig2_cross_devtest')


print('All fig2_cross_*.pdf/.png written.')
