"""Figures for System 2.1 — DDI with Machine Learning (MEM / two-stage).

All numbers traceable to code/2.1.DDI-ML/NOTES.md and the .stats files in
code/2.1.DDI-ML/results/. Hard-coded here so the script is self-contained.

Figures (PDF + PNG in report/):
    fig21_feature_mods      feature-mod additive sweep (devel macro-F1)
    fig21_mem_vs_svm        MEM vs SVM per-class F1 (devel)
    fig21_threshold_sweep   two-stage stage-1 threshold sweep (devel M & m)
    fig21_confmat           confusion matrix heat-map (mod_best2, devel)
    fig21_per_class_test    champion per-class P/R/F1 (test)
    fig21_strata            error stratification by source & sentence length
    fig21_progression       2.0 -> ref -> mod_best2 -> two-stage progression
"""
import numpy as np
from generate_plots_2_common import (
    plt, save, CLASSES, CLASS_COLORS, palette,
    C_ML, C_BASE, C_BEST, C_GOOD, C_BAD, GREY)

# =============================================================================
# 1. fig21_feature_mods — additive feature-mod sweep (devel macro-F1)
# =============================================================================
mods   = ['ref', 'mod1', 'mod2', 'mod3', 'mod4', 'mod5', 'mod_best\n(2+3+4)', 'mod_best2\n(2+3)']
M_dev  = [64.3, 62.4, 64.2, 64.0, 64.1, 63.0, 65.2, 65.4]
ref_M  = 64.3

fig, ax = plt.subplots(figsize=(10, 4.8))
colors = palette(6, start=4) + [C_BEST, C_BEST]   # start=4 avoids the champion pink
bars = ax.bar(mods, M_dev, color=colors, edgecolor='white')
ax.axhline(ref_M, color=GREY, linestyle=':', alpha=0.7, label=f'ref baseline ({ref_M})')
for b, v in zip(bars, M_dev):
    d = v - ref_M
    ax.text(b.get_x()+b.get_width()/2, v+0.12, f'{v:.1f}', ha='center', fontsize=9,
            fontweight='bold' if v >= 65.2 else 'normal',
            color=C_BEST if v >= 65.2 else 'black')
    ax.text(b.get_x()+b.get_width()/2, 60.4, f'{d:+.1f}', ha='center', fontsize=8,
            color=C_GOOD if d > 0 else (C_BAD if d < 0 else GREY))
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Feature engineering — additive mods over the shipped feature set\n'
             'Only mod2+mod3 together (mod_best2) beat ref; 4/5 single mods are neutral-to-negative',
             fontsize=11.5)
ax.set_ylim(60, 67)
ax.legend(loc='upper left', fontsize=9)
save(fig, 'fig21_feature_mods')


# =============================================================================
# 2. fig21_mem_vs_svm — MEM vs SVM per-class F1 (devel, ref features)
# =============================================================================
mem = {'advise': 62.6, 'effect': 65.4, 'int': 69.2, 'mechanism': 60.0}
svm = {'advise': 49.2, 'effect': 60.2, 'int': 61.5, 'mechanism': 49.0}

fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(CLASSES))
w = 0.36
ax.bar(x-w/2, [mem[c] for c in CLASSES], w, color=C_ML, edgecolor='white',
       label='MEM (LogisticRegression)  M=64.3')
ax.bar(x+w/2, [svm[c] for c in CLASSES], w, color=GREY, edgecolor='white',
       label='SVM (rbf)  M=55.0')
for i, c in enumerate(CLASSES):
    ax.text(i-w/2, mem[c]+0.6, f'{mem[c]:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i+w/2, svm[c]+0.6, f'{svm[c]:.1f}', ha='center', fontsize=9, color=GREY)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylabel('Devel F1 (%)')
ax.set_title('MEM beats SVM on every class — SVM is too precision-biased (recall < 45%)')
ax.set_ylim(0, 80)
ax.legend(loc='upper right', fontsize=9)
save(fig, 'fig21_mem_vs_svm')


# =============================================================================
# 3. fig21_threshold_sweep — two-stage stage-1 threshold sweep (devel)
# =============================================================================
thr   = [0.20, 0.30, 0.35, 0.37, 0.40, 0.42, 0.45, 0.50]
M_thr = [63.3, 64.8, 65.3, 66.0, 65.8, 65.7, 64.7, 64.4]
m_thr = [62.1, 63.9, 64.7, 65.2, 65.2, 65.1, 64.7, 64.8]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(thr, M_thr, 'o-', color=C_ML,   linewidth=2.2, markersize=8, label='macro-F1')
ax.plot(thr, m_thr, 's--', color=C_BEST, linewidth=2.0, markersize=7, label='micro-F1')
best_i = M_thr.index(max(M_thr))
ax.scatter(thr[best_i], M_thr[best_i], s=260, facecolors='none',
           edgecolors=C_BEST, linewidths=2.5, zorder=5)
ax.annotate(f'champion\nt={thr[best_i]}  M={M_thr[best_i]:.1f}',
            xy=(thr[best_i], M_thr[best_i]), xytext=(thr[best_i]+0.04, 66.6),
            fontsize=9, color=C_BEST, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_BEST, alpha=0.7))
for t, m in zip(thr, M_thr):
    ax.annotate(f'{m:.1f}', xy=(t, m), xytext=(0, 7), textcoords='offset points',
                ha='center', fontsize=8, color=C_ML)
ax.set_xlabel('Stage-1  P(positive)  decision threshold')
ax.set_ylabel('Devel F1 (%)')
ax.set_title('Two-stage classifier — stage-1 threshold sweep (mod_best2 features)\n'
             'Optimum well below 0.5 because the positive class is itself rare (~15% of training)',
             fontsize=11)
ax.set_ylim(61, 67.5)
ax.legend(loc='lower center', fontsize=9)
save(fig, 'fig21_threshold_sweep')


# =============================================================================
# 4. fig21_confmat — confusion matrix heat-map (mod_best2, devel)
# =============================================================================
labels = ['advise', 'effect', 'int', 'mechanism', 'null']
cm = np.array([
    [  92,    7,   1,    2,   41],
    [  13,  181,   0,    2,  120],
    [   0,    0,  26,    1,   16],
    [  12,    2,   0,  139,  108],
    [  23,   34,   6,   44, 4007],
])
# row-normalised for colour (so the dominant null row doesn't wash everything out)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7.6, 6))
im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        v = cm[i, j]
        ax.text(j, i, f'{v}', ha='center', va='center',
                color='white' if cm_norm[i, j] > 0.5 else 'black',
                fontsize=10, fontweight='bold' if i == j else 'normal')
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('PREDICTED'); ax.set_ylabel('GOLD')
ax.set_title('Confusion matrix — mod_best2 single-stage (devel)\n'
             'Recall failures dominate: 29-41% of every positive class leaks to null',
             fontsize=11)
# highlight the diagonal
for i in range(5):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                               edgecolor=C_BEST, linewidth=2.0))
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('row-normalised fraction')
save(fig, 'fig21_confmat')


# =============================================================================
# 5. fig21_per_class_test — champion per-class P/R/F1 (test, two-stage t=0.37)
# =============================================================================
P = {'advise': 68.8, 'effect': 60.1, 'int': 93.3, 'mechanism': 54.8}
R = {'advise': 61.2, 'effect': 66.2, 'int': 70.0, 'mechanism': 64.5}
F = {'advise': 64.8, 'effect': 63.0, 'int': 80.0, 'mechanism': 59.3}

fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(CLASSES))
w = 0.27
ax.bar(x-w, [P[c] for c in CLASSES], w, color=C_ML, alpha=0.55, edgecolor='white', label='Precision')
ax.bar(x  , [R[c] for c in CLASSES], w, color=C_ML, alpha=0.8,  edgecolor='white', label='Recall')
ax.bar(x+w, [F[c] for c in CLASSES], w, color=C_BEST, edgecolor='white', label='F1')
for i, c in enumerate(CLASSES):
    for off, d, col in [(-w, P, C_ML), (0, R, C_ML), (w, F, C_BEST)]:
        ax.text(i+off, d[c]+0.8, f'{d[c]:.0f}', ha='center', fontsize=8,
                fontweight='bold' if col == C_BEST else 'normal')
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylabel('Test (%)')
ax.set_title('Champion per-class on test — two-stage on mod_best2 (M=66.8)\n'
             'int is the easiest positive class (concentrated trigger lexicon); mechanism the hardest',
             fontsize=11)
ax.set_ylim(0, 105)
ax.legend(loc='upper left', fontsize=9, ncols=3)
save(fig, 'fig21_per_class_test')


# =============================================================================
# 6. fig21_strata — error stratification (source + sentence length), devel
# =============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

# Panel A — document source
src = ['DrugBank\n(4 454)', 'MedLine\n(423)']
src_M = [68.0, 43.0]; src_m = [66.9, 35.5]
xS = np.arange(len(src)); w = 0.35
axL.bar(xS-w/2, src_M, w, color=C_ML, edgecolor='white', label='macro-F1')
axL.bar(xS+w/2, src_m, w, color=C_ML, alpha=0.5, edgecolor='white', label='micro-F1')
for i, (a, b) in enumerate(zip(src_M, src_m)):
    axL.text(i-w/2, a+0.8, f'{a:.1f}', ha='center', fontsize=9, fontweight='bold')
    axL.text(i+w/2, b+0.8, f'{b:.1f}', ha='center', fontsize=9, color=GREY)
axL.set_xticks(xS); axL.set_xticklabels(src)
axL.set_ylabel('Devel F1 (%)'); axL.set_ylim(0, 80)
axL.set_title('By document source — MedLine ~25 pp harder')
axL.legend(loc='upper right', fontsize=9)

# Panel B — sentence length
ln = ['short\n(≤10)', 'medium\n(11-25)', 'long\n(26-50)', 'very long\n(>50)']
ln_M = [52.0, 70.7, 41.2, 45.7]
xL = np.arange(len(ln))
barsL = axR.bar(xL, ln_M, color=palette(4, start=4), edgecolor='white')
for b, v in zip(barsL, ln_M):
    axR.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.1f}', ha='center', fontsize=9,
             fontweight='bold' if v == max(ln_M) else 'normal',
             color=C_BEST if v == max(ln_M) else 'black')
axR.set_xticks(xL); axR.set_xticklabels(ln)
axR.set_ylabel('Devel macro-F1 (%)'); axR.set_ylim(0, 80)
axR.set_title('By sentence length — medium is the sweet spot')

fig.suptitle('System 2.1 error stratification (devel) — two genuine failure modes: MedLine prose & long sentences',
             fontsize=12, y=1.02)
save(fig, 'fig21_strata')


# =============================================================================
# 7. fig21_progression — campaign progression on test macro-F1
# =============================================================================
stages = ['2.0\nrule-based', '2.1 ref-MEM\n(devel only)', '2.1 mod_best2\nsingle-stage', '2.1 two-stage\n(t=0.37)']
prog_M = [26.9, 64.3, 65.7, 66.8]
prog_is_test = [True, False, True, True]   # ref-MEM only run on devel

fig, ax = plt.subplots(figsize=(9, 4.8))
xs = np.arange(len(stages))
colors = [C_BASE, GREY, C_ML, C_BEST]
bars = ax.bar(xs, prog_M, color=colors, edgecolor='white')
for b, v, t in zip(bars, prog_M, prog_is_test):
    lbl = f'{v:.1f}' + ('' if t else '*')
    ax.text(b.get_x()+b.get_width()/2, v+0.8, lbl, ha='center', fontsize=10,
            fontweight='bold' if v == max(prog_M) else 'normal',
            color=C_BEST if v == max(prog_M) else 'black')
ax.annotate('+39.9 pp test macro\nover rule-based',
            xy=(3, 66.8), xytext=(1.6, 45), ha='center', fontsize=10,
            color=C_GOOD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_GOOD, alpha=0.7))
ax.set_xticks(xs); ax.set_xticklabels(stages, fontsize=9.5)
ax.set_ylabel('macro-F1 (%)')
ax.set_title('System 2.1 progression  (test macro-F1; * = devel-only for ref-MEM)')
ax.set_ylim(0, 78)
save(fig, 'fig21_progression')


print('All fig21_*.pdf/.png written.')
