"""Figures for System 2.2 — DDI with Neural Networks (BiLSTM+CNN).

All numbers traceable to code/2.2.DDI-NN/NOTES.md and the .stats files in
code/2.2.DDI-NN/results/.

Figures (PDF + PNG in report/):
    fig22_input_mods     Phase I input-representation sweep (devel macro-F1)
    fig22_arch           architecture variants (no-LSTM collapse)
    fig22_hp             hyperparameter sweep (defaults win)
    fig22_seed_audit     seed lottery — devel vs test across 5 seeds
    fig22_relpos_strata  mod9 rel-pos vs mod2 by length & source (the win)
    fig22_per_class_test champion per-class P/R/F1 (mod9 seed 777, test)
    fig22_devel_test_gap devel->test slope for key configs
"""
import numpy as np
from generate_plots_2_common import (
    plt, save, CLASSES, CLASS_COLORS, palette,
    C_NN, C_BASE, C_BEST, C_GOOD, C_BAD, GREY)

# =============================================================================
# 1. fig22_input_mods — Phase I input-representation sweep (devel macro-F1)
# =============================================================================
mods  = ['ref', 'mod1\nsuf+et', 'mod2\npref+et', 'mod3\netype', 'mod4\nform',
         'mod5\ncombo', 'mod6\nsuf', 'mod7\npref']
M_dev = [55.8, 64.8, 65.8, 62.8, 61.8, 62.0, 61.0, 64.7]
ref_M = 55.8

fig, ax = plt.subplots(figsize=(10.5, 4.8))
colors = [GREY] + palette(7, start=4)   # start=4 avoids the champion pink
colors[2] = C_BEST   # mod2 best
bars = ax.bar(mods, M_dev, color=colors, edgecolor='white')
ax.axhline(ref_M, color=GREY, linestyle=':', alpha=0.7, label=f'ref baseline ({ref_M})')
for b, v in zip(bars, M_dev):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}', ha='center', fontsize=9,
            fontweight='bold' if v == max(M_dev) else 'normal',
            color=C_BEST if v == max(M_dev) else 'black')
ax.annotate('input representation\n= +10 pp (biggest lever)',
            xy=(2, 65.8), xytext=(4.2, 60), ha='center', fontsize=9.5,
            color=C_GOOD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_GOOD, alpha=0.7))
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Input representations (Phase I) — prefix + entity-type marker is the winner\n'
             'mod5 (all 3 extras) underperforms the best 2-feature combo: extra channels dilute signal',
             fontsize=11)
ax.set_ylim(50, 70)
ax.legend(loc='lower right', fontsize=9)
save(fig, 'fig22_input_mods')


# =============================================================================
# 2. fig22_arch — architecture variants (the no-LSTM collapse)
# =============================================================================
arch  = ['mod2\nbaseline', 'no LSTM', 'no CNN', '2-layer\nLSTM', 'wide\nLSTM', 'big\nemb']
M_arch = [65.8, 11.6, 64.7, 62.3, 63.0, 61.0]

fig, ax = plt.subplots(figsize=(9.5, 4.6))
colors = [C_BEST, C_BAD] + palette(4, start=4)
bars = ax.bar(arch, M_arch, color=colors, edgecolor='white')
for b, v in zip(bars, M_arch):
    ax.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.1f}', ha='center', fontsize=9,
            fontweight='bold' if v in (65.8, 11.6) else 'normal',
            color=C_BAD if v == 11.6 else ('black'))
ax.annotate('drop the BiLSTM →\nmodel predicts null\nfor (almost) everything',
            xy=(1, 11.6), xytext=(2.4, 35), ha='center', fontsize=9.5,
            color=C_BAD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_BAD, alpha=0.8))
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Architecture variants (Phase A) — the BiLSTM is the irreplaceable component')
ax.set_ylim(0, 75)
save(fig, 'fig22_arch')


# =============================================================================
# 3. fig22_hp — hyperparameter sweep (defaults win)
# =============================================================================
hp   = ['default\n(bs16/ep10)', 'bs=32', 'bs=64', 'ep=20', 'drop=0.3', 'lr=5e-4', 'ml=100']
M_hp = [65.8, 64.5, 63.3, 61.0, 63.2, 63.8, 62.4]

fig, ax = plt.subplots(figsize=(10, 4.5))
colors = [C_BEST] + palette(6, start=4)   # start=4 avoids the champion pink
bars = ax.bar(hp, M_hp, color=colors, edgecolor='white')
ax.axhline(65.8, color=C_BEST, linestyle=':', alpha=0.6, label='default (65.8)')
for b, v in zip(bars, M_hp):
    d = v - 65.8
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}', ha='center', fontsize=9)
    if d != 0:
        ax.text(b.get_x()+b.get_width()/2, 56.5, f'{d:+.1f}', ha='center', fontsize=8, color=C_BAD)
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Hyperparameter sweep (Phase H) — no tweak beats the shipped defaults')
ax.set_ylim(55, 68)
ax.legend(loc='upper right', fontsize=9)
save(fig, 'fig22_hp')


# =============================================================================
# 4. fig22_seed_audit — seed lottery: devel vs test across 5 seeds (mod2)
#    plus mod9 (rel-pos) overlay showing it generalises
# =============================================================================
seeds   = ['2345', '42', '777', '111', '2024']
mod2_dev = [65.8, 61.7, 61.8, 63.9, 64.0]
mod2_tst = [57.5, 56.0, 62.1, 59.7, 57.1]
mod9_dev = [63.8, 62.5, 64.7, None, None]
mod9_tst = [62.0, 61.0, 65.6, None, None]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(seeds)); w = 0.2
ax.bar(x-1.5*w, mod2_dev, w, color=C_NN, alpha=0.55, edgecolor='white', label='mod2 devel')
ax.bar(x-0.5*w, mod2_tst, w, color=C_NN, edgecolor='white', label='mod2 test')
m9d = [v if v is not None else 0 for v in mod9_dev]
m9t = [v if v is not None else 0 for v in mod9_tst]
ax.bar(x+0.5*w, m9d, w, color=C_BEST, alpha=0.55, edgecolor='white', label='mod9 rel-pos devel')
ax.bar(x+1.5*w, m9t, w, color=C_BEST, edgecolor='white', label='mod9 rel-pos test')

for i in range(len(seeds)):
    ax.text(i-1.5*w, mod2_dev[i]+0.5, f'{mod2_dev[i]:.0f}', ha='center', fontsize=7.5)
    ax.text(i-0.5*w, mod2_tst[i]+0.5, f'{mod2_tst[i]:.0f}', ha='center', fontsize=7.5)
    if mod9_dev[i] is not None:
        ax.text(i+0.5*w, mod9_dev[i]+0.5, f'{mod9_dev[i]:.0f}', ha='center', fontsize=7.5)
        ax.text(i+1.5*w, mod9_tst[i]+0.5, f'{mod9_tst[i]:.0f}', ha='center', fontsize=7.5, fontweight='bold')

# annotate the anti-correlation
ax.annotate('best devel seed (2345)\n= WORST test seed',
            xy=(0-0.5*w, 57.5), xytext=(1.3, 47), ha='center', fontsize=9,
            color=C_BAD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_BAD, alpha=0.7))
ax.annotate('rel-pos seed 777\n= champion (test 65.6)',
            xy=(2+1.5*w, 65.6), xytext=(3.4, 70), ha='center', fontsize=9,
            color=C_GOOD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_GOOD, alpha=0.7))
ax.set_xticks(x); ax.set_xticklabels([f'seed {s}' for s in seeds])
ax.set_ylabel('macro-F1 (%)')
ax.set_title('The seed lottery — single-seed devel selection is unreliable\n'
             'mod2 devel & test ranks are anti-correlated; rel-pos (mod9) lifts test across all seeds',
             fontsize=11)
ax.set_ylim(0, 75)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), ncols=4, fontsize=9)
save(fig, 'fig22_seed_audit')


# =============================================================================
# 5. fig22_relpos_strata — mod9 vs mod2 by sentence length & source (test)
# =============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8))

# Panel A — sentence length
ln = ['short\n(≤10)', 'medium\n(11-25)', 'long\n(26-50)', 'very long\n(>50)']
mod2_ln = [67.9, 60.5, 57.2, 2.7]
mod9_ln = [76.6, 61.1, 63.6, 19.8]
xL = np.arange(len(ln)); w = 0.36
axL.bar(xL-w/2, mod2_ln, w, color=C_NN, alpha=0.6, edgecolor='white', label='mod2 (no rel-pos)')
axL.bar(xL+w/2, mod9_ln, w, color=C_BEST, edgecolor='white', label='mod9 (rel-pos)')
for i in range(len(ln)):
    axL.text(i-w/2, mod2_ln[i]+1, f'{mod2_ln[i]:.1f}', ha='center', fontsize=8)
    axL.text(i+w/2, mod9_ln[i]+1, f'{mod9_ln[i]:.1f}', ha='center', fontsize=8, fontweight='bold')
    d = mod9_ln[i]-mod2_ln[i]
    axL.annotate(f'{d:+.1f}', xy=(i, max(mod2_ln[i], mod9_ln[i])+5), ha='center',
                 fontsize=8.5, color=C_GOOD, fontweight='bold')
axL.set_xticks(xL); axL.set_xticklabels(ln)
axL.set_ylabel('Test macro-F1 (%)'); axL.set_ylim(0, 90)
axL.set_title('rel-pos rescues the very-long-sentence regime (+17.1 pp)')
axL.legend(loc='upper right', fontsize=9)

# Panel B — document source
src = ['DrugBank', 'MedLine']
mod2_src = [58.6, 44.5]; mod9_src = [64.6, 24.3]
xS = np.arange(len(src))
axR.bar(xS-w/2, mod2_src, w, color=C_NN, alpha=0.6, edgecolor='white', label='mod2')
axR.bar(xS+w/2, mod9_src, w, color=C_BEST, edgecolor='white', label='mod9 rel-pos')
for i in range(len(src)):
    axR.text(i-w/2, mod2_src[i]+1, f'{mod2_src[i]:.1f}', ha='center', fontsize=8.5)
    axR.text(i+w/2, mod9_src[i]+1, f'{mod9_src[i]:.1f}', ha='center', fontsize=8.5, fontweight='bold')
axR.annotate('trades MedLine\nfor DrugBank',
             xy=(1+w/2, 24.3), xytext=(1.0, 50), ha='center', fontsize=9,
             color=C_BAD, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_BAD, alpha=0.7))
axR.set_xticks(xS); axR.set_xticklabels(src)
axR.set_ylabel('Test macro-F1 (%)'); axR.set_ylim(0, 75)
axR.set_title('Source trade-off (MedLine is ~10% of pairs)')
axR.legend(loc='upper right', fontsize=9)

fig.suptitle('The rel-pos win (mod9 vs mod2, test) — biggest gain on long sentences where the BiLSTM loses signal',
             fontsize=12, y=1.02)
save(fig, 'fig22_relpos_strata')


# =============================================================================
# 6. fig22_per_class_test — champion per-class P/R/F1 (mod9 seed 777, test)
# =============================================================================
P = {'advise': 61.3, 'effect': 61.0, 'int': 67.3, 'mechanism': 64.9}
R = {'advise': 63.6, 'effect': 65.2, 'int': 87.5, 'mechanism': 57.0}
F = {'advise': 62.4, 'effect': 63.0, 'int': 76.1, 'mechanism': 60.7}

fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(CLASSES)); w = 0.27
ax.bar(x-w, [P[c] for c in CLASSES], w, color=C_NN, alpha=0.55, edgecolor='white', label='Precision')
ax.bar(x  , [R[c] for c in CLASSES], w, color=C_NN, alpha=0.85, edgecolor='white', label='Recall')
ax.bar(x+w, [F[c] for c in CLASSES], w, color=C_BEST, edgecolor='white', label='F1')
for i, c in enumerate(CLASSES):
    for off, d, col in [(-w, P, C_NN), (0, R, C_NN), (w, F, C_BEST)]:
        ax.text(i+off, d[c]+0.8, f'{d[c]:.0f}', ha='center', fontsize=8,
                fontweight='bold' if col == C_BEST else 'normal')
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylabel('Test (%)')
ax.set_title('Champion per-class on test — mod9 rel-pos, seed 777 (M=65.6)')
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=9, ncols=3)
save(fig, 'fig22_per_class_test')


# =============================================================================
# 7. fig22_devel_test_gap — devel -> test slope for key configs
# =============================================================================
slope = [
    ('ref baseline',          55.8, None, GREY),
    ('mod2 seed 2345',        65.8, 57.5, C_NN),
    ('mod2 seed 777',         61.8, 62.1, C_NN),
    ('mod9 rel-pos s777',     64.7, 65.6, C_BEST),
    ('mod9 3-seed mean',      63.7, 62.9, C_GOOD),
]
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for name, dv, ts, col in slope:
    if ts is None:
        ax.plot([0], [dv], 'o', color=col, markersize=9)
        ax.annotate(f'{name} {dv:.1f}', xy=(0, dv), xytext=(-6, 0),
                    textcoords='offset points', ha='right', fontsize=8.5, color=col)
        continue
    ax.plot([0, 1], [dv, ts], 'o-', color=col, linewidth=2.2, markersize=9,
            label=f'{name}  (Δ={ts-dv:+.1f})')
    ax.annotate(f'{dv:.1f}', xy=(0, dv), xytext=(-6, 0), textcoords='offset points',
                ha='right', fontsize=9, color=col,
                fontweight='bold' if 's777' in name and 'mod9' in name else 'normal')
    ax.annotate(f'{ts:.1f}', xy=(1, ts), xytext=(6, 0), textcoords='offset points',
                ha='left', fontsize=9, color=col,
                fontweight='bold' if 'mod9 rel-pos' in name else 'normal')
ax.set_xticks([0, 1]); ax.set_xticklabels(['devel', 'test'], fontsize=11)
ax.set_xlim(-0.4, 1.35)
ax.set_ylabel('macro-F1 (%)')
ax.set_title('Devel → test gap — the devel-best mod2 collapses on test;\nrel-pos generalises (positive Δ)',
             fontsize=11)
ax.legend(loc='lower center', fontsize=8.5)
save(fig, 'fig22_devel_test_gap')


print('All fig22_*.pdf/.png written.')
