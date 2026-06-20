"""Generate plots for System 1.1 report section."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Color palette ──
C_CRF = '#2E86AB'    # blue
C_SVM = '#A23B72'    # magenta
C_MEM = '#F18F01'    # orange
C_BG  = '#F5F5F0'
GREY  = '#888888'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': C_BG,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# =============================================
# 1. Feature evolution plot (devel set)
# =============================================
mods = ['Base', 'Mod1', 'Mod2', 'Mod3', 'Mod4', 'Mod5', 'Mod6', 'Mod7', 'Mod8', 'Mod9\n(wo8)']
crf_scores = [67.9, 67.9, 67.9, 67.9, 67.5, 67.8, 68.3, 68.0, 67.5, 68.3]
svm_scores = [67.1, 67.5, 67.4, 67.5, 67.4, 67.1, 67.2, 67.2, None, 67.7]
mem_scores = [65.5, 65.9, 65.6, 66.2, 65.8, 66.3, 66.5, 66.5, 66.1, 66.2]

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(mods))

ax.plot(x, crf_scores, 'o-', color=C_CRF, linewidth=2.2, markersize=7, label='CRF', zorder=3)
# SVM has a None for mod8
svm_x = [i for i, v in enumerate(svm_scores) if v is not None]
svm_y = [v for v in svm_scores if v is not None]
ax.plot(svm_x, svm_y, 's--', color=C_SVM, linewidth=2.2, markersize=7, label='SVM', zorder=3)
ax.plot(x, mem_scores, '^:', color=C_MEM, linewidth=2.2, markersize=7, label='MEM', zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(mods, fontsize=9)
ax.set_ylabel('Micro-avg F1 (%)')
ax.set_title('Feature Engineering Progress (Devel Set)')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_ylim(64.5, 69.5)
ax.axhline(y=67.9, color=GREY, linestyle=':', alpha=0.4)

# Annotate best
ax.annotate('68.3%', xy=(6, 68.3), xytext=(6, 68.8), fontsize=9, fontweight='bold',
            color=C_CRF, ha='center', arrowprops=dict(arrowstyle='->', color=C_CRF, lw=1.2))
ax.annotate('68.3%', xy=(9, 68.3), xytext=(9, 68.8), fontsize=9, fontweight='bold',
            color=C_CRF, ha='center', arrowprops=dict(arrowstyle='->', color=C_CRF, lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_feature_evolution.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig_feature_evolution.png'), bbox_inches='tight', dpi=150)
plt.close()
print("1. Feature evolution plot saved.")

# =============================================
# 2. Hyperparameter heatmap for CRF (mod6 devel - best feature set on devel)
# =============================================
c1_vals = [0.01, 0.1, 0.5, 1.0]
c2_vals = [0.1, 0.5, 1.0, 5.0]
# Best iter per (c1, c2) from mod6 devel results
best_scores_mod6 = np.array([
    [68.5, 68.2, 68.2, 67.7],  # c1=0.01
    [68.4, 68.2, 68.2, 67.9],  # c1=0.1
    [67.8, 68.0, 67.9, 67.8],  # c1=0.5
    [67.8, 68.0, 67.9, 67.8],  # c1=1.0
])

fig, ax = plt.subplots(figsize=(5.5, 4))
im = ax.imshow(best_scores_mod6, cmap='YlOrRd', aspect='auto', vmin=67.0, vmax=68.6)
ax.set_xticks(range(len(c2_vals)))
ax.set_xticklabels([str(v) for v in c2_vals])
ax.set_yticks(range(len(c1_vals)))
ax.set_yticklabels([str(v) for v in c1_vals])
ax.set_xlabel('c2 (L2 regularization)')
ax.set_ylabel('c1 (L1 regularization)')
ax.set_title('CRF Best F1 per (c1, c2) — mod6 Devel')

for i in range(len(c1_vals)):
    for j in range(len(c2_vals)):
        val = best_scores_mod6[i, j]
        color = 'white' if val > 68.2 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('F1 (%)')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_crf_heatmap.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig_crf_heatmap.png'), bbox_inches='tight', dpi=150)
plt.close()
print("2. CRF heatmap saved.")

# =============================================
# 3. Test set: Devel vs Test comparison (grouped bar)
# =============================================
labels = [
    'CRF\nmod6\nbest',
    'CRF\nmod9',
    'CRF\nmod8',
    'SVM rbf\nmod8',
    'SVM lin\nmod8',
    'MEM\nmod8',
]
devel_f1 = [68.5, 68.3, 67.7, 67.7, 67.3, 66.2]
test_f1  = [63.4, 62.3, 68.2, 67.9, 67.9, 67.8]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels))
w = 0.35

bars_d = ax.bar(x - w/2, devel_f1, w, label='Devel F1', color='#5DADE2', edgecolor='white', linewidth=0.8)
bars_t = ax.bar(x + w/2, test_f1, w, label='Test F1', color='#E74C3C', edgecolor='white', linewidth=0.8)

for bar, val in zip(bars_d, devel_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val}',
            ha='center', va='bottom', fontsize=9, color='#2C3E50')
for bar, val in zip(bars_t, test_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val}',
            ha='center', va='bottom', fontsize=9, color='#C0392B', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Micro-avg F1 (%)')
ax.set_title('Devel vs Test Performance — Best Models')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(58, 72)

# Highlight overfitting
ax.annotate('overfitting\n(-5.1 pts)', xy=(0, 63.4), xytext=(0.6, 60.5),
            fontsize=8, color='red', fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='red', lw=1))

# Highlight best
ax.annotate('BEST', xy=(2, 68.2), xytext=(2, 70), fontsize=10, fontweight='bold',
            color='#27AE60', ha='center',
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_devel_vs_test.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig_devel_vs_test.png'), bbox_inches='tight', dpi=150)
plt.close()
print("3. Devel vs test comparison saved.")

# =============================================
# 4. SVM C sensitivity (mod8 test)
# =============================================
c_vals   = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0]
rbf_f1   = [67.3, 67.9, 67.9, 67.7, 67.8, 67.8, 67.7, 67.7, 67.7, 67.7, 67.7, 67.7]

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(range(len(c_vals)), rbf_f1, 'o-', color=C_SVM, linewidth=2, markersize=6, label='SVM rbf')
ax.axhline(y=67.9, color=C_SVM, linestyle=':', alpha=0.4)
ax.fill_between(range(len(c_vals)), [67.6]*len(c_vals), [68.0]*len(c_vals), alpha=0.1, color=C_SVM)

ax.set_xticks(range(len(c_vals)))
ax.set_xticklabels([str(v) for v in c_vals], fontsize=8, rotation=45)
ax.set_xlabel('C (regularization)')
ax.set_ylabel('Test F1 (%)')
ax.set_title('SVM rbf — Sensitivity to C (mod8, Test Set)')
ax.set_ylim(66.8, 68.3)
ax.legend()

# Mark best
ax.annotate('67.9%', xy=(1, 67.9), xytext=(1, 68.15), fontsize=9, fontweight='bold',
            color=C_SVM, ha='center', arrowprops=dict(arrowstyle='->', color=C_SVM))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_svm_c_sensitivity.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig_svm_c_sensitivity.png'), bbox_inches='tight', dpi=150)
plt.close()
print("4. SVM C sensitivity saved.")

# =============================================
# 5. Final results summary bar chart
# =============================================
models = ['CRF\n(mod8)', 'SVM rbf\n(mod8)', 'SVM linear\n(mod8)', 'MEM\n(mod8)', 'CRF\n(mod6)', 'CRF\n(mod9)']
scores = [68.2, 67.9, 67.9, 67.8, 63.4, 62.3]
colors = [C_CRF, C_SVM, C_SVM, C_MEM, '#8FBFDB', '#8FBFDB']

fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.barh(range(len(models)), scores, color=colors, edgecolor='white', linewidth=0.8, height=0.6)

for i, (bar, val) in enumerate(zip(bars, scores)):
    xpos = bar.get_width() + 0.2
    fw = 'bold' if i == 0 else 'normal'
    ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{val}%', va='center', fontsize=11, fontweight=fw)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=10)
ax.set_xlabel('Test F1 (%)')
ax.set_title('Final Test Set Results — All Models')
ax.set_xlim(58, 72)
ax.invert_yaxis()

# Add a vertical line at baseline
ax.axvline(x=67.9, color=GREY, linestyle=':', alpha=0.4, label='Baseline (67.9%)')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_final_results.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig_final_results.png'), bbox_inches='tight', dpi=150)
plt.close()
print("5. Final results summary saved.")

# =============================================
# 6. Per-entity-type F1 breakdown (best models, test)
# =============================================
entity_types = ['brand', 'drug', 'drug_n', 'group']
crf_f1_type  = [93.3, 92.3, 12.6, 74.5]
svm_f1_type  = [91.5, 92.4, 13.8, 74.0]
mem_f1_type  = [92.2, 92.0, 14.0, 73.2]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(entity_types))
w = 0.25

bars1 = ax.bar(x - w, crf_f1_type, w, label='CRF', color=C_CRF, edgecolor='white')
bars2 = ax.bar(x,     svm_f1_type, w, label='SVM', color=C_SVM, edgecolor='white')
bars3 = ax.bar(x + w, mem_f1_type, w, label='MEM', color=C_MEM, edgecolor='white')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(entity_types, fontsize=11)
ax.set_ylabel('F1 (%)')
ax.set_title('Per-Entity-Type F1 on Test Set (mod8 features)')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0, 105)

# Highlight drug_n — arrow origin placed in clear air above the drug_n group
ax.annotate('drug_n — all\nmodels struggle',
            xy=(2, 14), xytext=(1.3, 40),
            fontsize=9, color='red', fontstyle='italic', ha='center',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_per_entity_f1.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig_per_entity_f1.png'), bbox_inches='tight', dpi=150)
plt.close()
print("6. Per-entity-type F1 saved.")

# =============================================
# 7. Precision vs Recall scatter (best CRF, per entity type)
# =============================================
# CRF mod8 best
prec = [92.5, 94.6, 32.0, 74.7]
rec  = [94.2, 90.1,  7.8, 74.3]
labels_pr = ['brand', 'drug', 'drug_n', 'group']
sizes = [275, 2151, 102, 700]  # #expected entities

fig, ax = plt.subplots(figsize=(7, 5))
# Offsets keep labels inside the axes frame, with arrows linking to points
label_offsets = {'brand':  (-80,  10),   # upper-left of point
                 'drug':   ( 25,  -5),   # right of bubble
                 'group':  ( 15,  10),   # upper-right of point
                 'drug_n': ( 15,   0)}   # right of point
for i, (p, r, lbl, sz) in enumerate(zip(prec, rec, labels_pr, sizes)):
    ax.scatter(r, p, s=sz/5, alpha=0.7, color=[C_CRF, C_SVM, '#E74C3C', C_MEM][i],
               edgecolor='black', linewidth=0.5, zorder=3)
    dx, dy = label_offsets[lbl]
    ax.annotate(f'{lbl}\nF1={crf_f1_type[i]:.1f}%', xy=(r, p),
                xytext=(dx, dy), textcoords='offset points',
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color='#888', lw=0.6, alpha=0.7))

ax.plot([0, 100], [0, 100], '--', color=GREY, alpha=0.3, label='P = R')
ax.set_xlabel('Recall (%)')
ax.set_ylabel('Precision (%)')
ax.set_title('Precision vs Recall per Entity Type\n(Best CRF, mod8, Test Set)')
ax.set_xlim(0, 105)
ax.set_ylim(0, 105)
ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_pr_scatter.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig_pr_scatter.png'), bbox_inches='tight', dpi=150)
plt.close()
print("7. P/R scatter saved.")

print("\nAll plots generated!")
