"""Generate plots for System 1.3 (NERC with Large Language Models) report section.

Data sources:
  - code/1.3.NERC-LLM/results/*.stats   (one per sbatch run, per-class P/R/F1)
  - code/1.3.NERC-LLM/README_MODIFICATIONS.md §6.1 / 6.2 / 6.3
  - Boada squeue wall-clocks (Runtime columns in §6)

Every number is traceable to those files. Hard-coded here so the script
is self-contained and can regenerate the figures without SSHing the cluster.

Figures produced (all saved as PDF *and* PNG in the report/ directory):
    fig13_shots_sweep              k-sweep curve (Llama / prompts01 / FS)
    fig13_prompt_ablation          prompts x shots grouped bars
    fig13_per_class_fs             per-class F1 under 4 few-shot configs
    fig13_balanced_tradeoff        random vs balanced FS per-class deltas
    fig13_fs_vs_ft_headline        horizontal bars across all flagship configs
    fig13_rank_vs_epochs           r=8/10ep vs r=32/10ep vs r=8/15ep
    fig13_ft_per_class_heat        heat-map (configs x classes) on devel
    fig13_devel_test_gap           slope chart devel->test for FT configs
    fig13_drugn_story              drug_n F1 across all phases (devel+test)
    fig13_final_test_bars          stacked/grouped per-class bars on test
    fig13_timeline                 Phase B...I chronological progression
    fig13_cross_system             CRF (1.1) vs BiLSTM (1.2) vs FT-LLM (1.3)
    fig13_cost_scatter             training GPU-hours vs devel m-F1
    fig13_fs_vs_ft_perclass        side-by-side per-class FS -> FT transition
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Color palette (consistent with 1.1 / 1.2) ────────────────────────
C_CRF   = '#2E86AB'    # blue     — 1.1 CRF reference
C_NN    = '#009E73'    # green    — 1.2 BiLSTM reference
C_LLAMA = '#D55E00'    # red/orange — Llama 3.2 3B
C_QWEN  = '#5E3C99'    # purple  — Qwen 2.5 3B
C_FS    = '#F18F01'    # orange   — few-shot
C_FT    = '#2E86AB'    # blue     — fine-tune
C_BEST  = '#CC0066'    # pink/red — best model highlight
C_BAL   = '#0072B2'    # deep blue — balanced sampler
C_BAD   = '#D55E00'    # red-orange — negative delta / overfit
C_BG    = '#F5F5F0'
GREY    = '#888888'

# Per-class colors (fixed so figures cross-reference)
CLASS_COLORS = {
    'brand'  : '#E69F00',   # amber
    'drug'   : '#2E86AB',   # blue
    'drug_n' : '#CC0066',   # hot-pink  (rare class — visually distinct)
    'group'  : '#009E73',   # green
}
CLASSES = ['brand', 'drug', 'drug_n', 'group']

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': C_BG,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})


def _save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(OUT, f'{name}.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


# =============================================================================
# 1. fig13_shots_sweep — number-of-shots k sweep (llama32B3 / prompts01 / quant)
#    Jobs: 414918 (k=0), 414919 (k=3), 414791 (k=5), 414920 (k=10), 414921 (k=15)
# =============================================================================
shots  = [0, 3, 5, 10, 15]
mF1    = [2.1, 36.3, 46.9, 45.3, 50.3]
MF1    = [1.9, 34.5, 40.1, 41.5, 39.6]
span   = [2.9, 54.3, 58.8, 60.9, 62.9]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(shots, mF1,  'o-', color=C_LLAMA, linewidth=2.2, markersize=8,
        label='micro-F1')
ax.plot(shots, MF1,  's--', color=C_QWEN,  linewidth=2.0, markersize=7,
        label='macro-F1')
ax.plot(shots, span, '^:',  color=GREY,    linewidth=1.5, markersize=6,
        label='span-F1 (no class)')

for k, m in zip(shots, mF1):
    ax.annotate(f'{m:.1f}', xy=(k, m), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, color=C_LLAMA, fontweight='bold')

ax.axhline(50.3, color=C_LLAMA, linestyle=':', alpha=0.3)
ax.annotate('best FS m-F1\n(k=15, 50.3 %)', xy=(15, 50.3), xytext=(11.5, 57),
            ha='center', fontsize=9, color=C_LLAMA,
            arrowprops=dict(arrowstyle='->', color=C_LLAMA, alpha=0.6))

ax.set_xticks(shots)
ax.set_xlabel('Number of few-shot demonstrations  (k)')
ax.set_ylabel('Devel F1 (%)')
ax.set_title('Few-shot k sweep — Llama 3.2 3B Instruct / prompts01 / 4-bit quant')
ax.set_ylim(0, 70)
ax.legend(loc='lower right', framealpha=0.95)
_save(fig, 'fig13_shots_sweep')


# =============================================================================
# 2. fig13_prompt_ablation — prompts {01,02,03} x {5, 15} shots grouped bars
#    Jobs: 414791 (p01/5), 414922 (p02/5), 414923 (p03/5),
#          414921 (p01/15), 415438 (p02/15), 415439 (p03/15)
# =============================================================================
prompts_lbl = ['prompts01\n(generic)', 'prompts02\n(strict)', 'prompts03\n(terse)']
mF1_5       = [46.9, 46.4, 51.3]
mF1_15      = [50.3, 50.9, 54.2]
MF1_5       = [40.1, 34.2, 33.5]
MF1_15      = [39.6, 36.4, 35.3]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(prompts_lbl))
w = 0.35

# Panel A: micro-F1
axL.bar(x - w/2, mF1_5,  w, color=C_FS,    edgecolor='white', label='5 shots')
axL.bar(x + w/2, mF1_15, w, color=C_LLAMA, edgecolor='white', label='15 shots')
for i, (m5, m15) in enumerate(zip(mF1_5, mF1_15)):
    axL.text(i - w/2, m5  + 0.3, f'{m5:.1f}',  ha='center', fontsize=9)
    axL.text(i + w/2, m15 + 0.3, f'{m15:.1f}', ha='center', fontsize=9,
             fontweight='bold' if m15 >= 54.2 else 'normal',
             color=C_BEST if m15 >= 54.2 else 'black')
axL.set_xticks(x); axL.set_xticklabels(prompts_lbl, fontsize=9)
axL.set_ylabel('Devel micro-F1 (%)')
axL.set_title('Micro-F1 — prompt × shots')
axL.set_ylim(0, 62)
axL.legend(loc='lower right')

# Panel B: macro-F1
axR.bar(x - w/2, MF1_5,  w, color=C_FS,    edgecolor='white', label='5 shots')
axR.bar(x + w/2, MF1_15, w, color=C_LLAMA, edgecolor='white', label='15 shots')
for i, (m5, m15) in enumerate(zip(MF1_5, MF1_15)):
    axR.text(i - w/2, m5  + 0.3, f'{m5:.1f}',  ha='center', fontsize=9)
    axR.text(i + w/2, m15 + 0.3, f'{m15:.1f}', ha='center', fontsize=9)
axR.set_xticks(x); axR.set_xticklabels(prompts_lbl, fontsize=9)
axR.set_ylabel('Devel macro-F1 (%)')
axR.set_title('Macro-F1 — refined prompts collapse rare-class recall')
axR.set_ylim(0, 50)

fig.suptitle('Prompt × Shots ablation  (Llama 3.2 3B / 4-bit quant)',
             fontsize=13, y=1.02)
_save(fig, 'fig13_prompt_ablation')


# =============================================================================
# 3. fig13_per_class_fs — per-class F1 across 4 few-shot configs
#    Numbers pulled from .stats files on Boada (via /tmp/boada_perclass.py)
# =============================================================================
fs_configs = ['llama-p01-5',  'llama-p01-15', 'llama-p03-15', 'llama-p01-15\nbalanced']
fs_brand   = [46.9,            50.6,            39.8,           52.1]
fs_drug    = [56.0,            60.2,            64.2,           53.2]
fs_drugn   = [29.7,            14.3,             0.0,           29.9]
fs_group   = [27.8,            33.2,            37.3,           32.0]

fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(fs_configs))
w = 0.2
ax.bar(x - 1.5*w, fs_brand,  w, color=CLASS_COLORS['brand'],  label='brand',   edgecolor='white')
ax.bar(x - 0.5*w, fs_drug,   w, color=CLASS_COLORS['drug'],   label='drug',    edgecolor='white')
ax.bar(x + 0.5*w, fs_drugn,  w, color=CLASS_COLORS['drug_n'], label='drug_n', edgecolor='white')
ax.bar(x + 1.5*w, fs_group,  w, color=CLASS_COLORS['group'],  label='group',   edgecolor='white')

for i, vals in enumerate(zip(fs_brand, fs_drug, fs_drugn, fs_group)):
    for j, v in enumerate(vals):
        ax.text(i + (j - 1.5)*w, v + 1.2, f'{v:.1f}', ha='center', fontsize=7.5,
                fontweight='bold' if v == 0 else 'normal',
                color=C_BEST if (v == 0) else 'black')

ax.set_xticks(x); ax.set_xticklabels(fs_configs, fontsize=9)
ax.set_ylabel('Devel F1 (%)')
ax.set_title('Few-shot per-class F1 — terser prompts lift the common classes but kill drug_n\n'
             'p03/15 reaches drug F1 = 64.2 % but drug_n collapses to 0 % → rare-class blind spot',
             fontsize=11.5)
ax.set_ylim(0, 85)
ax.legend(loc='upper right', ncols=4, fontsize=9)
_save(fig, 'fig13_per_class_fs')


# =============================================================================
# 4. fig13_balanced_tradeoff — random vs balanced FS (p01/15 only)
#    Panel A: headline metrics    Panel B: per-class deltas (diverging bars)
# =============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.5))

# Panel A — grouped bars: micro and macro side-by-side (no overlay)
samplers   = ['random\n(15 shots)', 'balanced\n(15 shots)']
micro_vals = [50.3, 45.2]
macro_vals = [39.6, 41.8]
xA = np.arange(len(samplers))
wA = 0.35
bars_m = axL.bar(xA - wA/2, micro_vals, wA, color=[C_FS, C_BAL],
                 edgecolor='white', label='micro-F1')
bars_M = axL.bar(xA + wA/2, macro_vals, wA, color=[C_FS, C_BAL], alpha=0.45,
                 edgecolor='white', label='macro-F1')
for i, v in enumerate(micro_vals):
    axL.text(i - wA/2, v + 0.6, f'{v:.1f}', ha='center', fontweight='bold', fontsize=9.5)
for i, v in enumerate(macro_vals):
    axL.text(i + wA/2, v + 0.6, f'{v:.1f}', ha='center', fontsize=9, color=GREY)
axL.set_xticks(xA); axL.set_xticklabels(samplers)
axL.set_ylabel('Devel F1 (%)')
axL.set_title('Headline: micro ↓ 5.1 pp, macro ↑ 2.2 pp')
axL.set_ylim(0, 58)
axL.legend(loc='upper right', fontsize=9, framealpha=0.95)

# Panel B — per-class deltas (horizontal diverging bars)
# Labels placed OUTSIDE the axes region opposite the y-axis tick labels
# so they never collide with the class names on the left.
classes = CLASSES
rand = {'brand': 50.6, 'drug': 60.2, 'drug_n': 14.3, 'group': 33.2}
bal  = {'brand': 52.1, 'drug': 53.2, 'drug_n': 29.9, 'group': 32.0}
deltas = [bal[c] - rand[c] for c in classes]
colors = [C_BAL if d > 0 else C_BAD for d in deltas]
axR.barh(classes, deltas, color=colors, edgecolor='white', height=0.6)
# Put every label to the RIGHT of the zero line, with horizontal offset,
# so negative-bar labels clear the y-axis class names.
for i, (c, d) in enumerate(zip(classes, deltas)):
    sign = '+' if d >= 0 else ''
    if d >= 0:
        axR.text(d + 0.6, i, f'{sign}{d:.1f} pp',
                 va='center', ha='left',
                 fontsize=10, fontweight='bold', color=C_BAL)
    else:
        # label goes to the RIGHT of the zero-line (outside the negative bar)
        axR.text(0.6, i, f'{sign}{d:.1f} pp',
                 va='center', ha='left',
                 fontsize=10, fontweight='bold', color=C_BAD)
axR.axvline(0, color='black', linewidth=0.9)
axR.set_xlabel('balanced − random  (pp F1)')
axR.set_title('Per-class delta: drug_n rescued at the cost of the common classes')
axR.set_xlim(-12, 22)

fig.suptitle('Phase G — class-balanced few-shot sampler  (prompts01 / 15 shots)',
             fontsize=13, y=1.02)
_save(fig, 'fig13_balanced_tradeoff')


# =============================================================================
# 5. fig13_fs_vs_ft_headline — horizontal bars across ALL flagship configs
# =============================================================================
head_configs = [
    'zero-shot',
    'FS random 5',
    'FS random 15',
    'FS random 15 (p03, best FS micro)',
    'FS balanced 15',
    'FT llama r=8 / 10ep  (baseline)',
    'FT qwen  r=8 / 10ep  (Phase F)',
    'FT llama r=8 / 15ep  (Phase G)',
    'FT llama r=32 / 10ep (best FT)',
    'FT qwen  r=32 / 10ep (best macro)',
]
head_mF1 = [2.1, 46.9, 50.3, 54.2, 45.2, 82.5, 84.2, 84.8, 88.0, 87.1]
head_MF1 = [1.9, 40.1, 39.6, 35.3, 41.8, 68.2, 73.5, 74.6, 78.1, 78.5]
head_kind = ['FS', 'FS', 'FS', 'FS', 'FS', 'FT-L', 'FT-Q', 'FT-L', 'FT-L', 'FT-Q']
kind_color = {'FS': C_FS, 'FT-L': C_LLAMA, 'FT-Q': C_QWEN}
head_colors = [kind_color[k] for k in head_kind]

fig, ax = plt.subplots(figsize=(10.5, 5.5))
y = np.arange(len(head_configs))
ax.barh(y - 0.2, head_mF1, 0.4, color=head_colors, edgecolor='white', label='micro-F1')
ax.barh(y + 0.2, head_MF1, 0.4, color=head_colors, alpha=0.45,
        edgecolor='white', label='macro-F1')

for i, (m, M) in enumerate(zip(head_mF1, head_MF1)):
    fw = 'bold' if m == max(head_mF1) else 'normal'
    ax.text(m + 0.4, i - 0.2, f'{m:.1f}', va='center', fontsize=8.5, fontweight=fw,
            color=C_BEST if m == max(head_mF1) else 'black')
    ax.text(M + 0.4, i + 0.2, f'{M:.1f}', va='center', fontsize=8.5,
            color=C_BEST if M == max(head_MF1) else GREY)

# Highlight the single best config
best_idx = head_mF1.index(max(head_mF1))
ax.axhspan(best_idx - 0.45, best_idx + 0.45, color=C_BEST, alpha=0.08, zorder=0)

ax.set_yticks(y); ax.set_yticklabels(head_configs, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Devel F1 (%)')
ax.set_title('From zero-shot to best fine-tune — headline devel F1 per configuration')
ax.set_xlim(0, 100)

legend_patches = [
    mpatches.Patch(color=C_FS,    label='few-shot'),
    mpatches.Patch(color=C_LLAMA, label='fine-tune (Llama 3.2 3B)'),
    mpatches.Patch(color=C_QWEN,  label='fine-tune (Qwen 2.5 3B)'),
]
ax.legend(handles=legend_patches, loc='upper right',
          bbox_to_anchor=(0.98, 0.98), fontsize=9, framealpha=0.95)
_save(fig, 'fig13_fs_vs_ft_headline')


# =============================================================================
# 6. fig13_rank_vs_epochs — LoRA rank (8/32) x epochs (10/15)
#    415464+415481 (r=32/10ep), 414926+415067 (r=8/10ep), 415465+415516 (r=8/15ep)
# =============================================================================
labels = ['r=8\n10 epochs\n(baseline)', 'r=32\n10 epochs\n(Phase G)',
          'r=8\n15 epochs\n(Phase G)']
mF1    = [82.5, 88.0, 84.8]
MF1    = [68.2, 78.1, 74.6]
drugn  = [19.7, 45.2, 42.9]

fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(labels))
w = 0.27
ax.bar(x - w, mF1,   w, color=C_LLAMA,           edgecolor='white', label='micro-F1')
ax.bar(x    , MF1,   w, color=C_LLAMA, alpha=0.6, edgecolor='white', label='macro-F1')
ax.bar(x + w, drugn, w, color=CLASS_COLORS['drug_n'], edgecolor='white',
       label='drug_n F1')

for i, (m, M, d) in enumerate(zip(mF1, MF1, drugn)):
    ax.text(i - w, m + 0.5, f'{m:.1f}', ha='center', fontsize=9,
            fontweight='bold' if i == 1 else 'normal',
            color=C_BEST if i == 1 else 'black')
    ax.text(i    , M + 0.5, f'{M:.1f}', ha='center', fontsize=9)
    ax.text(i + w, d + 0.5, f'{d:.1f}', ha='center', fontsize=9)

# Annotate the large r=32 gain
ax.annotate('rank ↑ helps MORE\nthan epochs ↑',
            xy=(1, 88.0), xytext=(1.55, 96),
            ha='center', fontsize=9.5, color=C_BEST, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_BEST, alpha=0.7))

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Devel F1 (%)')
ax.set_title('LoRA rank vs training epochs  (llama32B3 / prompts01 / quant)')
ax.set_ylim(0, 105)
ax.legend(loc='upper left', fontsize=9)
_save(fig, 'fig13_rank_vs_epochs')


# =============================================================================
# 7. fig13_ft_per_class_heat — heat-map of configurations × classes (devel)
# =============================================================================
rows = ['FT-llama r=8 / 10ep',   'FT-qwen  r=8 / 10ep',
        'FT-llama r=8 / 15ep',   'FT-llama r=32 / 10ep', 'FT-qwen  r=32 / 10ep']
cols = CLASSES
mat = np.array([
    [87.1, 89.6, 19.7, 76.5],   # llama r=8
    [89.4, 89.0, 41.2, 74.2],   # qwen  r=8
    [87.6, 89.7, 42.9, 78.3],   # llama r=8 / 15ep
    [92.3, 91.6, 45.2, 83.5],   # llama r=32
    [90.9, 90.6, 51.9, 80.8],   # qwen  r=32  (best macro)
])

fig, ax = plt.subplots(figsize=(8.2, 4.5))
im = ax.imshow(mat, cmap='viridis', aspect='auto', vmin=0, vmax=100)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                color='white' if v < 65 else 'black',
                fontsize=10, fontweight='bold')
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, fontsize=10)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows, fontsize=10)
ax.set_title('Per-class F1 on devel — fine-tune configurations')
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('F1 (%)')

# mark absolute best per column
best_rows = mat.argmax(axis=0)
for j, ri in enumerate(best_rows):
    ax.add_patch(plt.Rectangle((j - 0.48, ri - 0.48), 0.96, 0.96,
                               fill=False, edgecolor=C_BEST, linewidth=2.4))
_save(fig, 'fig13_ft_per_class_heat')


# =============================================================================
# 8. fig13_devel_test_gap — slope chart for FT configs (devel → test)
# =============================================================================
slope = [
    ('FS llama p01/15',       50.3, 48.8, C_FS),
    ('FT llama r=8 / 10ep',   82.5, 83.3, C_LLAMA),
    ('FT llama r=32 / 10ep',  88.0, 86.0, C_BEST),
    ('FT qwen  r=32 / 10ep',  87.1, 85.3, C_QWEN),
]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
# hand-tuned vertical offsets so labels don't collide at the dense 63/58-ish bands
devel_offsets = {
    'FS llama p01/15':       ( -6,  0),
    'FT llama r=8 / 10ep':   ( -6,  0),
    'FT llama r=32 / 10ep':  ( -6,  6),   # push up — avoids 63.0 Qwen label
    'FT qwen  r=32 / 10ep':  ( -6, -8),   # push down
}
test_offsets = {
    'FS llama p01/15':       (  6,  0),
    'FT llama r=8 / 10ep':   (  6,  0),
    'FT llama r=32 / 10ep':  (  6,  6),   # push up — avoids 58.4 Qwen label
    'FT qwen  r=32 / 10ep':  (  6, -8),   # push down
}
for name, dv, ts, col in slope:
    ax.plot([0, 1], [dv, ts], 'o-', color=col, linewidth=2.2, markersize=9,
            label=f'{name}  (Δ={ts-dv:+.1f} pp)')
    dx, dy = devel_offsets.get(name, (-6, 0))
    ax.annotate(f'{dv:.1f}', xy=(0, dv), xytext=(dx, dy),
                textcoords='offset points', ha='right', fontsize=9, color=col,
                fontweight='bold' if name.startswith('FT llama r=32') else 'normal')
    tx, ty = test_offsets.get(name, (6, 0))
    ax.annotate(f'{ts:.1f}', xy=(1, ts), xytext=(tx, ty),
                textcoords='offset points', ha='left',  fontsize=9, color=col,
                fontweight='bold' if name.startswith('FT llama r=32') else 'normal')

ax.set_xticks([0, 1]); ax.set_xticklabels(['devel', 'test'], fontsize=11)
ax.set_xlim(-0.28, 1.28)
ax.set_ylabel('micro-F1 (%)')
ax.set_title('Devel → test generalisation gap')
ax.legend(loc='lower left', fontsize=9, framealpha=0.95)
_save(fig, 'fig13_devel_test_gap')


# =============================================================================
# 9. fig13_drugn_story — drug_n F1 across phases (devel + test side by side)
# =============================================================================
drugn_configs = [
    'FS random 5',       'FS random 15',   'FS balanced 15',
    'FT llama r=8',      'FT qwen  r=8',   'FT llama r=8 15ep',
    'FT llama r=32',     'FT qwen  r=32',
]
drugn_dev = [29.7, 14.3, 29.9, 19.7, 41.2, 42.9, 45.2, 51.9]
drugn_ts  = [None, None, None, 32.2, None, None, 51.3, 46.9]

fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(len(drugn_configs))
bars = ax.bar(x, drugn_dev, color=CLASS_COLORS['drug_n'], edgecolor='white',
              label='devel drug_n F1', alpha=0.85)

# overlay test dots where available (pushed RIGHT of the devel bar)
for i, t in enumerate(drugn_ts):
    if t is not None:
        ax.plot(i + 0.25, t, 'D', color='black', markersize=9,
                markerfacecolor='white', markeredgewidth=2)
        ax.annotate(f'test\n{t:.1f}', xy=(i + 0.25, t), xytext=(11, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=8.5, color='black', fontweight='bold')

for i, v in enumerate(drugn_dev):
    ax.text(i, v + 1.0, f'{v:.1f}', ha='center', fontsize=9,
            fontweight='bold' if v == max(drugn_dev) else 'normal',
            color=C_BEST if v == max(drugn_dev) else 'black')

ax.set_xticks(x); ax.set_xticklabels(drugn_configs, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('drug_n F1 (%)')
ax.set_title('The rare-class story: drug_n F1 — from 14.3 % (best FS) to 51.9 % on devel, 51.3 % on test')
ax.set_ylim(0, 68)

# annotate devel→test deltas in a clear band above the bars (no crossing arrows)
ax.annotate('Llama r=32 devel → test:  45.2 → 51.3  (+6.1 pp, generalises)',
            xy=(0.02, 0.94), xycoords='axes fraction', ha='left',
            fontsize=9, color='#007A33', fontweight='bold')
ax.annotate('Qwen  r=32 devel → test:  51.9 → 46.9  (−5.0 pp, slight overfit)',
            xy=(0.02, 0.87), xycoords='axes fraction', ha='left',
            fontsize=9, color=C_BAD, fontweight='bold')

ax.annotate('♦  = test-set F1  (Phases E, H, I)',
            xy=(0.02, 0.80), xycoords='axes fraction', ha='left',
            fontsize=8.5, color='black')
_save(fig, 'fig13_drugn_story')


# =============================================================================
# 10. fig13_final_test_bars — per-class F1 on test for the 4 systems run on test
# =============================================================================
test_systems = ['FS 15\n(Phase E)',  'FT r=8\n(Phase E)',
                'FT r=32\n(Phase H)', 'Qwen r=32\n(Phase I)']
# per-class F1 on test
test_brand  = [36.0, 83.0, 84.4, 92.1]
test_drug   = [61.1, 89.5, 91.0, 89.3]
test_drugn  = [ 0.0, 32.2, 51.3, 46.9]
test_group  = [28.5, 74.8, 78.4, 75.7]
# micro-F1 overlay
test_micro  = [48.8, 83.3, 86.0, 85.3]

fig, ax = plt.subplots(figsize=(10.5, 5))
x = np.arange(len(test_systems))
w = 0.18
ax.bar(x - 1.5*w, test_brand, w, color=CLASS_COLORS['brand'],  label='brand',  edgecolor='white')
ax.bar(x - 0.5*w, test_drug,  w, color=CLASS_COLORS['drug'],   label='drug',   edgecolor='white')
ax.bar(x + 0.5*w, test_drugn, w, color=CLASS_COLORS['drug_n'], label='drug_n', edgecolor='white')
ax.bar(x + 1.5*w, test_group, w, color=CLASS_COLORS['group'],  label='group',  edgecolor='white')

for i in range(len(test_systems)):
    for j, v in enumerate([test_brand[i], test_drug[i], test_drugn[i], test_group[i]]):
        ax.text(i + (j - 1.5)*w, v + 0.9, f'{v:.1f}', ha='center', fontsize=7.6)

# micro-F1 reference markers — placed ABOVE all per-class bars (top row)
# with horizontal label to the right so nothing collides with value labels.
MICRO_Y = 102.0
for i, m in enumerate(test_micro):
    ax.plot(i, MICRO_Y, 'o', color='black', markersize=11, markerfacecolor='white',
            markeredgewidth=2)
    ax.text(i, MICRO_Y + 3.2, f'm-F1 {m:.1f}', ha='center', fontsize=9,
            fontweight='bold',
            color=C_BEST if m == max(test_micro) else 'black')

# Winner callout
winner = np.argmax(test_micro)
ax.axvspan(winner - 0.42, winner + 0.42, color=C_BEST, alpha=0.06)

ax.set_xticks(x); ax.set_xticklabels(test_systems, fontsize=9.5)
ax.set_ylabel('Test F1 (%)')
ax.set_title('Test-set per-class F1 — final head-to-head between our four test-run systems')
ax.set_ylim(0, 115)
# legend below the axes so it doesn't compete with the m-F1 row
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncols=4,
          fontsize=9, framealpha=0.95)
_save(fig, 'fig13_final_test_bars')


# =============================================================================
# 11. fig13_timeline — experiments B…I on a horizontal time axis
# =============================================================================
phases = [
    ('B',  1, 'smoke',              46.9, C_FS),
    ('C',  2, 'FS grid (best p03)', 54.2, C_FS),
    ('D',  3, 'LoRA baseline',      82.5, C_LLAMA),
    ('E',  4, 'test — r=8',         83.3, C_LLAMA),
    ('F',  5, 'Qwen FT / pXX x15',  84.2, C_QWEN),
    ('G',  6, 'r=32 / ep15 / bal',  88.0, C_BEST),
    ('H',  7, 'test — r=32',        86.0, C_BEST),
    ('I',  8, 'test — Qwen r=32',   85.3, C_QWEN),
]
fig, ax = plt.subplots(figsize=(12, 4.5))
xs  = [p[1] for p in phases]
ys  = [p[3] for p in phases]
cs  = [p[4] for p in phases]
ax.plot(xs, ys, '-', color=GREY, linewidth=1.5, alpha=0.5, zorder=0)
# alternate label above/below marker so nothing stacks on the dotted lines
# or runs into a neighbouring dot. Index-based so positions are deterministic.
label_offsets = {
    'B': (0,  14),  'C': (0,  14),
    'D': (0, -40),  'E': (0,  14),
    'F': (0, -42),  'G': (0,  14),
    'H': (0, -42),  'I': (0, -42),
}
for ph, x_, desc, y_, col in phases:
    ax.scatter(x_, y_, color=col, edgecolor='white', s=180, zorder=2)
    dx, dy = label_offsets[ph]
    ax.annotate(f'Phase {ph}\n{desc}\n{y_:.1f} %',
                xy=(x_, y_), xytext=(dx, dy),
                textcoords='offset points', ha='center', fontsize=8.5,
                color=col, fontweight='bold')
ax.axhline(54.2, color=C_FS,    linestyle=':', alpha=0.5,
           label='best few-shot (54.2 %)')
ax.axhline(88.0, color=C_BEST,  linestyle=':', alpha=0.5,
           label='best FT devel (88.0 %)')

ax.set_xlabel('Experiment phase  (chronological order)')
ax.set_ylabel('Headline micro-F1 (%)')
ax.set_title('Phase B → I — campaign progression  (devel m-F1 unless otherwise noted)')
ax.set_ylim(40, 100)
ax.set_xlim(0.4, 8.6)
ax.set_xticks(xs)
ax.set_xticklabels([p[0] for p in phases])
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncols=2,
          fontsize=9, framealpha=0.95)
_save(fig, 'fig13_timeline')


# =============================================================================
# 12. fig13_cross_system — 1.1 CRF  vs  1.2 BiLSTM  vs  1.3 FT-LLM on TEST
#     (two panels: micro-F1 and macro-F1, both reported by util/evaluator.py)
# =============================================================================
cross_sys  = ['System 1.1\nCRF mod8',
              'System 1.2\nBiLSTM champ_big',
              'System 1.3\nFS  best (p01/15)',
              'System 1.3\nFT  r=8',
              'System 1.3\nFT  r=32 (best)']
cross_mic  = [86.8, 87.4, 48.8, 83.3, 86.0]
cross_mac  = [68.2, 69.1, 31.4, 69.9, 76.3]
cross_col  = [C_CRF, C_NN, C_FS, C_LLAMA, C_BEST]

fig, (axM, axL) = plt.subplots(1, 2, figsize=(12.5, 5))

for ax, vals, title in [(axM, cross_mic, 'Test micro-F1 (m.avg)'),
                         (axL, cross_mac, 'Test macro-F1 (M.avg)')]:
    bars = ax.bar(cross_sys, vals, color=cross_col, edgecolor='white')
    for b, v in zip(bars, vals):
        fw = 'bold' if v == max(vals) else 'normal'
        ax.text(b.get_x() + b.get_width()/2, v + 0.8, f'{v:.1f}',
                ha='center', fontsize=10, fontweight=fw)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Test F1 (%)')
    ax.set_title(title)
    ax.axhline(vals[1], color=C_NN, linestyle=':', alpha=0.4)
    # gap annotation
    gap = vals[-1] - vals[1]
    gap_col = '#007A33' if gap >= 0 else C_BAD
    ax.annotate(f'Δ = {gap:+.1f} pp\nvs 1.2 BiLSTM',
                xy=(4, vals[-1]), xytext=(3.5, 20),
                ha='center', fontsize=9, color=gap_col, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=gap_col, alpha=0.7))
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(9)
        lbl.set_rotation(15)
        lbl.set_ha('right')

fig.suptitle('Cross-system comparison on the DDI test set — fine-tuned 3B LLM matches dedicated NERC systems\n'
             '(micro within 1.4 pp of 1.2 BiLSTM; macro +7.2 pp — LLM wins on the rare class)',
             fontsize=12, y=1.02)
_save(fig, 'fig13_cross_system')


# =============================================================================
# 15. fig13_cross_system_perclass — per-class F1 across 1.1 / 1.2 / 1.3 on TEST
#     pulled from 1.1/23_CRF .stats, 1.2/champ_big_s42 .stats, 1.3/FT-r=32 .stats
# =============================================================================
cross_perclass = {
    'System 1.1 (CRF mod8)'            : {'brand': 93.3, 'drug': 92.3, 'drug_n': 12.6, 'group': 74.5},
    'System 1.2 (BiLSTM champ_big_s42)': {'brand': 89.1, 'drug': 92.0, 'drug_n': 15.6, 'group': 79.8},
    'System 1.3 (FT Llama r=32)'       : {'brand': 84.4, 'drug': 91.0, 'drug_n': 51.3, 'group': 78.4},
}
fig, ax = plt.subplots(figsize=(10.5, 5))
x = np.arange(len(CLASSES))
w = 0.27
syscolors = [C_CRF, C_NN, C_BEST]
for i, (name, data) in enumerate(cross_perclass.items()):
    vals = [data[c] for c in CLASSES]
    bars = ax.bar(x + (i - 1) * w, vals, w, color=syscolors[i],
                  edgecolor='white', label=name)
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 1) * w, v + 1.2, f'{v:.1f}',
                ha='center', fontsize=8.5,
                fontweight='bold' if v == max([cross_perclass[s][CLASSES[j]]
                                               for s in cross_perclass]) else 'normal')
ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel('Test F1 (%)')
ax.set_title('Per-class F1 on test across the three systems — where does 1.3 win, where does it lose?')
ax.set_ylim(0, 118)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=9, framealpha=0.95)

# callout: LLM wins drug_n — arrow points at the TOP of the 1.3 bar
# (so the label and the "51.3" value marker never compete for the same pixels)
ax.annotate('1.3 wins drug_n\n+35.7 pp vs 1.2,  +38.7 pp vs 1.1',
            xy=(2 + 0.27, 56), xytext=(2.7, 95),
            ha='center', fontsize=9.5, color='#007A33', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#007A33', alpha=0.8))
_save(fig, 'fig13_cross_system_perclass')


# =============================================================================
# 13. fig13_cost_scatter — training GPU-hours  vs  devel m-F1
# =============================================================================
# (train-hours, devel mF1, label, marker, color, label_offset_xy)
# label_offset_xy hand-tuned to disambiguate the dense upper cluster
points = [
    (0.00, 46.9, 'FS 5',             'o', C_FS,    (0.22, -0.1)),
    (0.00, 50.3, 'FS 15',            'o', C_FS,    (0.22, -0.1)),
    (0.00, 54.2, 'FS p03/15',        'o', C_FS,    (0.22, -0.1)),
    (0.00, 45.2, 'FS bal',           'o', C_BAL,   (0.22, -0.1)),
    (5.13, 82.5, 'FT r=8 / 10ep',    's', C_LLAMA, (0.00, -2.6)),
    (5.95, 84.2, 'FT qwen r=8',      's', C_QWEN,  (0.15,  0.3)),
    (7.96, 84.8, 'FT r=8 / 15ep',    's', C_LLAMA, (-0.55, -2.6)),
    (5.33, 88.0, 'FT r=32 / 10ep',   '*', C_BEST,  (-0.65,  1.4)),
    (6.50, 87.1, 'FT qwen r=32',     '*', C_QWEN,  (0.15,  1.4)),
]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
for h, f, name, mk, col, (xo, yo) in points:
    xj = h + np.random.uniform(-0.03, 0.03) if h == 0 else h
    ax.scatter(xj, f, marker=mk, s=190 if mk == '*' else 120,
               color=col, edgecolor='white', linewidth=1.2, zorder=2)
    ax.annotate(name, xy=(h, f), xytext=(h + xo, f + yo), fontsize=8.8,
                color=col,
                fontweight='bold' if name.startswith('FT r=32') else 'normal')
ax.set_xlabel('Training wall-clock  (GPU-hours, RTX-3080)')
ax.set_ylabel('Devel micro-F1 (%)')
ax.set_title('Cost vs. accuracy — rank-32 LoRA is the cheapest large-gain move')
ax.set_xlim(-0.6, 9.3)
ax.set_ylim(40, 95)

# legend markers
leg_patches = [
    plt.scatter([], [], marker='o', s=110, color=C_FS, edgecolor='white', label='few-shot (no training)'),
    plt.scatter([], [], marker='s', s=110, color=C_LLAMA, edgecolor='white', label='fine-tune'),
    plt.scatter([], [], marker='*', s=170, color=C_BEST, edgecolor='white', label='fine-tune, rank-32'),
]
ax.legend(handles=leg_patches, loc='lower right', fontsize=9, framealpha=0.95)
_save(fig, 'fig13_cost_scatter')


# =============================================================================
# 14. fig13_fs_vs_ft_perclass — direct bar-pair: best FS (p03/15) vs best FT (r=32) per class
# =============================================================================
# best FS per-class F1 = prompts03/15 (job 415439)
fs_pc    = {'brand': 39.8, 'drug': 64.2, 'drug_n':  0.0, 'group': 37.3}
ft_pc    = {'brand': 92.3, 'drug': 91.6, 'drug_n': 45.2, 'group': 83.5}

fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(CLASSES))
w = 0.36
ax.bar(x - w/2, [fs_pc[c] for c in CLASSES], w, color=C_FS, edgecolor='white',
       label='best few-shot (prompts03 / 15, m-F1 54.2 %)')
ax.bar(x + w/2, [ft_pc[c] for c in CLASSES], w, color=C_BEST, edgecolor='white',
       label='best fine-tune (Llama r=32 / 10ep, m-F1 88.0 %)')

for i, c in enumerate(CLASSES):
    fv, tv = fs_pc[c], ft_pc[c]
    ax.text(i - w/2, fv + 1, f'{fv:.1f}', ha='center', fontsize=9)
    ax.text(i + w/2, tv + 1, f'{tv:.1f}', ha='center', fontsize=9, fontweight='bold')
    # delta annotation
    d = tv - fv
    ax.annotate(f'+{d:.1f} pp', xy=(i, max(fv, tv) + 6), ha='center',
                fontsize=9, color='#007A33', fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylabel('Devel F1 (%)')
ax.set_title('Few-shot → fine-tune gains per class  (same base model, same 4-bit quant)')
ax.set_ylim(0, 115)
ax.legend(loc='upper right', fontsize=9)
_save(fig, 'fig13_fs_vs_ft_perclass')


print('All fig13_*.pdf/.png written to', OUT)
