"""Generate plots for System 1.2 (NERC with Neural Networks) report section.

Data sources:
  - code/1.2.NERC-NN/results/BATCH.log, COMBO.log, EXTRA.log, CHAMP.log,
    BIG_SEEDS.log (one line per experiment with macro-F1)
  - code/1.2.NERC-NN/results/test-*.stats (per-class test eval)
  - README_MODIFICATIONS.md §5.1–5.14

Every number is traceable to those files. Hard-coded here so the script
is self-contained and regenerates the figures without re-running the
training pipeline.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Color palette (1.1 + NN-specific extensions) ────────────────────
C_CRF   = '#2E86AB'    # blue        — reused from 1.1 for CRF
C_SVM   = '#A23B72'    # magenta     — reused from 1.1 for SVM
C_MEM   = '#F18F01'    # orange      — reused from 1.1 for MEM
C_NN    = '#009E73'    # green       — 1.2 NN headline color
C_BIG   = '#5E3C99'    # purple      — champ_big variant
C_BAD   = '#D55E00'    # red-orange  — negative results / regressions
C_BG    = '#F5F5F0'
GREY    = '#888888'

# Category colors for single-axis sweep
C_INPUT = '#0072B2'    # inputs
C_ARCH  = '#CC79A7'    # architecture
C_HYPER = '#E69F00'    # hyperparameter

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': C_BG,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# =============================================================================
# 1. Round-1 single-axis sweep (exp01 … exp10)
# =============================================================================
exp_ids    = ['exp01\nbaseline', 'exp02\n+pos', 'exp03\n+pref', 'exp04\n+lemma',
              'exp05\n+all_in', 'exp06\nlstm2', 'exp07\nlayernorm', 'exp08\nadamw',
              'exp09\nbig', 'exp10\ngelu+dr']
exp_scores = [54.2, 63.3, 57.9, 51.5, 58.0, 59.7, 60.5, 53.3, 58.1, 62.7]
exp_cats   = ['base', 'input', 'input', 'input', 'input', 'arch', 'arch',
              'hyper', 'arch', 'hyper']

cat_color = {'base': GREY, 'input': C_INPUT, 'arch': C_ARCH, 'hyper': C_HYPER}
bar_colors = [cat_color[c] for c in exp_cats]

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(exp_ids))
bars = ax.bar(x, exp_scores, color=bar_colors, edgecolor='white', linewidth=0.8)

# Baseline reference line
ax.axhline(y=54.2, color=GREY, linestyle=':', alpha=0.6, label='baseline (54.2%)')

for bar, val, cat in zip(bars, exp_scores, exp_cats):
    delta = val - 54.2
    sign = '+' if delta >= 0 else ''
    txt = f'{val:.1f}\n({sign}{delta:.1f})'
    fw = 'bold' if val >= 62 else 'normal'
    color = '#27AE60' if delta > 0 else C_BAD if delta < 0 else 'black'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, txt,
            ha='center', va='bottom', fontsize=8, fontweight=fw, color=color)

ax.set_xticks(x)
ax.set_xticklabels(exp_ids, fontsize=8)
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Round-1 Single-Axis Sweep — ∆ vs. baseline')
ax.set_ylim(48, 70)

# Legend — placed lower-left to avoid overlapping the tallest bar labels
legend_patches = [
    mpatches.Patch(color=GREY,   label='baseline'),
    mpatches.Patch(color=C_INPUT, label='input features'),
    mpatches.Patch(color=C_ARCH,  label='architecture'),
    mpatches.Patch(color=C_HYPER, label='hyperparameter'),
]
ax.legend(handles=legend_patches, loc='lower left', framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_round1_sweep.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_round1_sweep.png'), bbox_inches='tight', dpi=150)
plt.close()
print("1. Round-1 sweep saved.")

# =============================================================================
# 2. Round-2 data-preprocessing sweeps
# =============================================================================
data_labels = ['suf_len=3', 'suf_len=5\n(default)', 'suf_len=7',
               'pref_len=2', 'pref_len=3\n(default)', 'pref_len=4',
               'bs=16', 'bs=32\n(default)', 'bs=64',
               'max=100', 'max=150\n(default)', 'max=200']
data_scores = [57.6, 54.2, 51.2, 59.3, 57.9, 58.3, 57.3, 54.2, 51.6, 57.1, 54.2, 53.9]
group_of    = ['suf', 'suf', 'suf', 'pref', 'pref', 'pref', 'bs', 'bs', 'bs', 'ml', 'ml', 'ml']
is_default  = [False, True, False, False, True, False, False, True, False, False, True, False]

group_colors = {'suf': '#1F77B4', 'pref': '#FF7F0E', 'bs': '#2CA02C', 'ml': '#9467BD'}
bar_colors2 = ['#CCCCCC' if d else group_colors[g] for g, d in zip(group_of, is_default)]

fig, ax = plt.subplots(figsize=(12, 4.8))
x = np.arange(len(data_labels))
bars = ax.bar(x, data_scores, color=bar_colors2, edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, data_scores):
    delta = val - 54.2
    sign = '+' if delta >= 0 else ''
    color = '#27AE60' if delta > 0 else C_BAD if delta < 0 else 'black'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}\n({sign}{delta:.1f})',
            ha='center', va='bottom', fontsize=8, color=color)

ax.axhline(y=54.2, color=GREY, linestyle=':', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(data_labels, fontsize=9)
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Round-2 Data-Preprocessing Sweeps (one-knob-at-a-time, vs. baseline 54.2%)')
ax.set_ylim(48, 64)

# Group separators
for sep_x in [2.5, 5.5, 8.5]:
    ax.axvline(x=sep_x, color='white', linewidth=2)
    ax.axvline(x=sep_x, color=GREY, linestyle='--', alpha=0.3)

# Category labels
for center, label in zip([1, 4, 7, 10],
                          ['suffix length', 'prefix length', 'batch size', 'max_len']):
    ax.text(center, 62.5, label, ha='center', fontsize=10, fontweight='bold', color='#555')

legend_patches2 = [
    mpatches.Patch(color='#CCCCCC', label='baseline default'),
    mpatches.Patch(color=group_colors['suf'],  label='suffix variants'),
    mpatches.Patch(color=group_colors['pref'], label='prefix variants'),
    mpatches.Patch(color=group_colors['bs'],   label='batch-size variants'),
    mpatches.Patch(color=group_colors['ml'],   label='max-len variants'),
]
ax.legend(handles=legend_patches2, loc='lower left', framealpha=0.9, fontsize=8, ncol=1)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_data_sweeps.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_data_sweeps.png'), bbox_inches='tight', dpi=150)
plt.close()
print("2. Data sweeps saved.")

# =============================================================================
# 3. Pretrained word embeddings — negative result
# =============================================================================
pre_labels = ['baseline\n(random)', 'pretrained\nfine-tune', 'pretrained\nfrozen',
              'pretrained\n+pos', 'final04\n(no pre)', 'final04\n+pretrained']
pre_scores = [54.2, 51.9, 59.7, 52.4, 67.4, 61.7]
pre_colors = [GREY, C_BAD, '#F1C40F', C_BAD, C_NN, C_BAD]

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(pre_labels))
bars = ax.bar(x, pre_scores, color=pre_colors, edgecolor='white', linewidth=0.8)

for i, (bar, val) in enumerate(zip(bars, pre_scores)):
    if i == 0:
        delta_txt = 'baseline'
    elif i in (1, 2, 3):
        d = val - 54.2
        delta_txt = f'∆={d:+.1f}\nvs base'
    elif i == 4:
        delta_txt = 'final04\nno pre'
    else:
        d = val - 67.4
        delta_txt = f'∆={d:+.1f}\nvs final04'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
            delta_txt, ha='center', va='top', fontsize=8,
            color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(pre_labels, fontsize=9)
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Pretrained Word Embeddings (spaCy en_core_web_md 300d) — Negative Result')
ax.set_ylim(45, 72)

# Baseline + final04 reference lines with labels in the clear area above bar 3
ax.axhline(y=54.2, color=GREY, linestyle=':', alpha=0.5,
           label='baseline (random) = 54.2')
ax.axhline(y=67.4, color=C_NN, linestyle=':', alpha=0.5,
           label='final04 (no pretrained) = 67.4')
ax.legend(loc='upper left', framealpha=0.85, fontsize=8)

# Explanation moved to figure caption in the report — no in-plot annotation.

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_pretrained.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_pretrained.png'), bbox_inches='tight', dpi=150)
plt.close()
print("3. Pretrained embeddings saved.")

# =============================================================================
# 4. Epoch sensitivity curves (champion vs champ_big at seed 2345)
# =============================================================================
epochs_champ  = [8,    15,   18,   20,   22,   25]
scores_champ  = [67.4, 68.3, 70.0, 70.0, 69.2, 69.1]  # §5.3 + §5.10.B + §5.6
epochs_big    = [20]
scores_big    = [70.6]                                # §5.10.C

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(epochs_champ, scores_champ, 'o-', color=C_NN, linewidth=2.3, markersize=9,
        label='champion (hidden=200, seed 2345)', zorder=3)
ax.scatter(epochs_big, scores_big, s=180, color=C_BIG, marker='*',
           edgecolor='black', linewidth=0.8, zorder=4,
           label='champ_big (hidden=300, seed 2345)')

# Annotations on every point
for ep, sc in zip(epochs_champ, scores_champ):
    ax.annotate(f'{sc:.1f}', xy=(ep, sc), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold', color=C_NN)
ax.annotate(f'{scores_big[0]:.1f}', xy=(epochs_big[0], scores_big[0]),
            xytext=(10, -3), textcoords='offset points',
            fontsize=10, fontweight='bold', color=C_BIG)

ax.set_xlabel('Epochs')
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Longer Training Wins (up to ~20 epochs, then saturates)')
ax.set_xticks(epochs_champ)
ax.set_ylim(66.5, 72)
ax.legend(loc='lower right', framealpha=0.9)

# Highlight under/over zones
ax.axvspan(7, 14, alpha=0.08, color=C_BAD, label='_under-trained')
ax.axvspan(21, 26, alpha=0.08, color=C_BAD)
ax.axvspan(17, 21, alpha=0.08, color='#27AE60')
ax.text(11, 71.2, 'under-trained', fontsize=9, color=C_BAD, style='italic', ha='center')
ax.text(19, 71.2, 'sweet spot',    fontsize=9, color='#27AE60', style='italic', ha='center', fontweight='bold')
ax.text(23.5, 71.2, 'saturating',  fontsize=9, color=C_BAD, style='italic', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_epoch_curve.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_epoch_curve.png'), bbox_inches='tight', dpi=150)
plt.close()
print("4. Epoch curve saved.")

# =============================================================================
# 5. Seed stability: champion (200) vs champ_big (300) devel + test
# =============================================================================
# Regular champion (hidden=200) — 6 variants
champ_devel = [70.0, 68.1, 68.4, 68.0, 70.0, 69.2]
champ_test  = [68.8, 68.8, 68.2, 66.9, 68.5, 69.3]
champ_tag   = ['s2345', 's111', 's777', 's42', 'e18', 'e22']

# champ_big — 3 seeds
big_devel = [70.6, 70.7, 67.6]
big_test  = [70.1, 69.1, 70.6]
big_tag   = ['s2345', 's42', 's777']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

# --- Panel A: devel ---
ax = axes[0]
rng = np.random.default_rng(0)
jitter_c = rng.uniform(-0.10, 0.10, size=len(champ_devel))
jitter_b = rng.uniform(-0.10, 0.10, size=len(big_devel))

ax.scatter([0 + j for j in jitter_c], champ_devel, s=110, color=C_NN,
           edgecolor='black', linewidth=0.6, alpha=0.85, label='champion (h=200)', zorder=3)
ax.scatter([1 + j for j in jitter_b], big_devel, s=160, color=C_BIG, marker='*',
           edgecolor='black', linewidth=0.6, alpha=0.9, label='champ_big (h=300)', zorder=3)

# Per-tag offsets chosen so coincident y-values don't stack their labels
offs_dev_champ = {'s2345': (8, 5), 's111': (8, -8), 's777': (8, 0),
                  's42': (8, 0), 'e18': (8, -8), 'e22': (8, 0)}
offs_dev_big   = {'s2345': (9, -8), 's42': (9, 5), 's777': (9, 0)}

for v, j in zip(champ_tag, jitter_c):
    idx = champ_tag.index(v)
    ax.annotate(v, xy=(0 + j, champ_devel[idx]), xytext=offs_dev_champ[v],
                textcoords='offset points', fontsize=7, color='#333')
for v, j in zip(big_tag, jitter_b):
    idx = big_tag.index(v)
    ax.annotate(v, xy=(1 + j, big_devel[idx]), xytext=offs_dev_big[v],
                textcoords='offset points', fontsize=7, color=C_BIG, fontweight='bold')

# means
m_c = np.mean(champ_devel); m_b = np.mean(big_devel)
ax.hlines(m_c, -0.35, 0.35, colors=C_NN, linewidth=2.2)
ax.hlines(m_b,  0.65, 1.35, colors=C_BIG, linewidth=2.2)
ax.text(-0.37, m_c, f'µ={m_c:.1f}', ha='right', va='center', fontsize=9, color=C_NN, fontweight='bold')
ax.text( 1.37, m_b, f'µ={m_b:.1f}', ha='left',  va='center', fontsize=9, color=C_BIG, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(['champion\n(h=200, n=6)', 'champ_big\n(h=300, n=3)'])
ax.set_ylabel('macro-F1 (%)')
ax.set_title('Devel set — seed + epoch variants')
ax.set_xlim(-0.6, 1.7)
ax.set_ylim(66, 72)
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

# --- Panel B: test ---
ax = axes[1]
ax.scatter([0 + j for j in jitter_c], champ_test, s=110, color=C_NN,
           edgecolor='black', linewidth=0.6, alpha=0.85, zorder=3)
ax.scatter([1 + j for j in jitter_b], big_test, s=160, color=C_BIG, marker='*',
           edgecolor='black', linewidth=0.6, alpha=0.9, zorder=3)

offs_tst_champ = {'s2345': (8, 5), 's111': (8, -8), 's777': (8, 0),
                  's42': (8, 0), 'e18': (8, 0), 'e22': (8, 0)}
offs_tst_big   = {'s2345': (9, 0), 's42': (9, 0), 's777': (9, 0)}

for v, j in zip(champ_tag, jitter_c):
    idx = champ_tag.index(v)
    ax.annotate(v, xy=(0 + j, champ_test[idx]), xytext=offs_tst_champ[v],
                textcoords='offset points', fontsize=7, color='#333')
for v, j in zip(big_tag, jitter_b):
    idx = big_tag.index(v)
    ax.annotate(v, xy=(1 + j, big_test[idx]), xytext=offs_tst_big[v],
                textcoords='offset points', fontsize=7, color=C_BIG, fontweight='bold')

m_c = np.mean(champ_test); m_b = np.mean(big_test)
ax.hlines(m_c, -0.35, 0.35, colors=C_NN, linewidth=2.2)
ax.hlines(m_b,  0.65, 1.35, colors=C_BIG, linewidth=2.2)
ax.text(-0.37, m_c, f'µ={m_c:.1f}', ha='right', va='center', fontsize=9, color=C_NN, fontweight='bold')
ax.text( 1.37, m_b, f'µ={m_b:.1f}', ha='left',  va='center', fontsize=9, color=C_BIG, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(['champion\n(h=200, n=6)', 'champ_big\n(h=300, n=3)'])
ax.set_title('Test set — same models')
ax.set_xlim(-0.6, 1.7)

# Highlight the +1.5 improvement
ax.annotate('', xy=(1, m_b), xytext=(0, m_c),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
ax.text(0.5, (m_b + m_c)/2, f'+{m_b - m_c:.1f} pt\non test',
        ha='center', fontsize=10, color='#27AE60', fontweight='bold')

fig.suptitle('Seed Variance Audit — Bigger LSTM is a Robust Test-Set Winner',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_seed_variance.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_seed_variance.png'), bbox_inches='tight', dpi=150)
plt.close()
print("5. Seed variance saved.")

# =============================================================================
# 6. Devel vs Test scatter — showing weak / inverted correlation
# =============================================================================
# All 9 configs with both devel and test
all_labels = champ_tag + big_tag
all_devel  = champ_devel + big_devel
all_test   = champ_test  + big_test
all_is_big = [False]*len(champ_devel) + [True]*len(big_devel)

fig, ax = plt.subplots(figsize=(7.5, 6))
for d, t, lbl, is_b in zip(all_devel, all_test, all_labels, all_is_big):
    color = C_BIG if is_b else C_NN
    marker = '*' if is_b else 'o'
    size = 240 if is_b else 120
    ax.scatter(d, t, s=size, color=color, marker=marker,
               edgecolor='black', linewidth=0.7, alpha=0.85, zorder=3)
    tag = ('big_' if is_b else '') + lbl
    ax.annotate(tag, xy=(d, t), xytext=(6, 4),
                textcoords='offset points', fontsize=8,
                color=color, fontweight='bold' if is_b else 'normal')

# Diagonal d=t
lo, hi = 66, 72
ax.plot([lo, hi], [lo, hi], '--', color=GREY, alpha=0.4, label='devel = test')

# Linear fits separately
champ_arr_d = np.array(champ_devel); champ_arr_t = np.array(champ_test)
big_arr_d   = np.array(big_devel);   big_arr_t   = np.array(big_test)

# pearson correlations
def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1])

r_champ = pearson(champ_devel, champ_test)
r_big   = pearson(big_devel, big_test)
r_all   = pearson(all_devel, all_test)

ax.text(0.02, 0.97,
        f'Pearson r:\n  champion  = {r_champ:+.2f}\n  champ_big = {r_big:+.2f}\n  combined  = {r_all:+.2f}',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

ax.set_xlabel('Devel macro-F1 (%)')
ax.set_ylabel('Test macro-F1 (%)')
ax.set_title('Devel does NOT predict Test inside ±1 pt\n(champ_big_s777 ↘ lowest devel, highest test)')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)

# legend
legend_elems = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_NN,
               markeredgecolor='black', markersize=10, label='champion (h=200)'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=C_BIG,
               markeredgecolor='black', markersize=14, label='champ_big (h=300)'),
    plt.Line2D([0], [0], color=GREY, linestyle='--', alpha=0.4, label='devel = test'),
]
ax.legend(handles=legend_elems, loc='lower right', framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_devel_vs_test.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_devel_vs_test.png'), bbox_inches='tight', dpi=150)
plt.close()
print("6. Devel vs test scatter saved.")

# =============================================================================
# 7. Per-entity-type F1 on test — final champion vs previous final vs 1.1 CRF
# =============================================================================
entity_types = ['brand', 'drug', 'drug_n', 'group']
# champ_big_s42 (new champion)
nn_new  = [89.1, 92.0, 15.6, 79.8]
# final04 (round-1 champion on test, from §5.3)
nn_old  = [87.5, 91.0, 15.3, 80.9]
# 1.1 CRF best on test — run 23, mod8 c1=c2=0.1 iter=50 (canonical 1.1 champion)
crf_11  = [93.3, 92.3, 12.6, 74.5]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(entity_types))
w = 0.26

b1 = ax.bar(x - w, crf_11, w, label='1.1 CRF (mod8)', color=C_CRF, edgecolor='white')
b2 = ax.bar(x,     nn_old, w, label='1.2 NN round-1 (final04)', color='#8FBFDB', edgecolor='white')
b3 = ax.bar(x + w, nn_new, w, label='1.2 NN final (champ_big_s42)', color=C_NN, edgecolor='white')

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(entity_types, fontsize=11)
ax.set_ylabel('Test F1 (%)')
ax.set_title('Per-Entity-Type Test F1 — 1.1 CRF vs. 1.2 NN (round-1 → final)')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0, 105)

# Per-class callouts removed — the caption already explains the pattern
# and the bar deltas make the story readable at a glance.

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_per_entity.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_per_entity.png'), bbox_inches='tight', dpi=150)
plt.close()
print("7. Per-entity saved.")

# =============================================================================
# 8. Progress timeline — rounds 1 → 5 (best devel + best test per round)
# =============================================================================
rounds = ['R1\nexp01-10\n+ final01-05', 'R2\nexp11-22\n+ long/seed/best',
          'R3\nfinal2_01-05', 'R4\nchamp_seed/e/big', 'R5\nbig_seeds']
round_best_devel = [67.4, 69.0, 70.2, 70.6, 70.7]
round_best_test  = [68.7, None, 68.8, None, 69.1]
round_mean_test  = [None, None, None, None, 69.9]  # champ_big mean across 3 seeds

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(rounds))

ax.plot(x, round_best_devel, 'o-', color=C_NN, linewidth=2.5, markersize=12,
        label='Best devel F1 (round)', zorder=3)

# Only plot test dots where we have them
test_x = [i for i, v in enumerate(round_best_test) if v is not None]
test_y = [round_best_test[i] for i in test_x]
ax.plot(test_x, test_y, 's-', color=C_BAD, linewidth=2.5, markersize=11,
        label='Best test F1 (single-seed)', zorder=3)

# Mean test (round 5 only)
mean_x = [i for i, v in enumerate(round_mean_test) if v is not None]
mean_y = [round_mean_test[i] for i in mean_x]
ax.plot(mean_x, mean_y, 'D', color=C_BIG, markersize=13,
        label='Mean test over 3 seeds (champ_big)', zorder=4,
        markeredgecolor='black', markeredgewidth=0.6)

for i, v in enumerate(round_best_devel):
    ax.annotate(f'{v:.1f}', xy=(i, v), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=10,
                fontweight='bold', color=C_NN)
for i, v in zip(test_x, test_y):
    ax.annotate(f'{v:.1f}', xy=(i, v), xytext=(0, -18),
                textcoords='offset points', ha='center', fontsize=10,
                fontweight='bold', color=C_BAD)
for i, v in zip(mean_x, mean_y):
    ax.annotate(f'{v:.1f}', xy=(i, v), xytext=(12, 0),
                textcoords='offset points', fontsize=10,
                fontweight='bold', color=C_BIG)

# 1.1 baseline reference
ax.axhline(y=68.2, color=C_CRF, linestyle=':', alpha=0.6, linewidth=1.5)
ax.text(0.05, 68.35, '1.1 CRF test = 68.2', fontsize=9, color=C_CRF, ha='left', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(rounds, fontsize=9)
ax.set_ylabel('macro-F1 (%)')
ax.set_title('Progress Timeline — System 1.2 (Rounds 1 → 5)')
ax.set_ylim(66.5, 72)
ax.legend(loc='lower right', framealpha=0.9)

# Story callouts removed — dotted reference line + the caption carry the story.

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_timeline.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_timeline.png'), bbox_inches='tight', dpi=150)
plt.close()
print("8. Timeline saved.")

# =============================================================================
# 9. Final cross-system comparison (grouped horizontal bars)
# =============================================================================
systems = ['1.1 CRF\n(mod8)', '1.2 NN R1\n(final04)', '1.2 NN final\n(champ_big_s42)',
           '1.2 NN final\n(3-seed mean)']
tests   = [68.2, 68.7, 69.1, 69.9]
colors_sys = [C_CRF, '#8FBFDB', C_NN, C_BIG]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(range(len(systems)), tests, color=colors_sys,
               edgecolor='white', linewidth=0.8, height=0.65)

for i, (bar, val) in enumerate(zip(bars, tests)):
    fw = 'bold' if i >= 2 else 'normal'
    if i == 0:
        txt = f'{val:.1f}%'
    else:
        d = val - 68.2
        txt = f'{val:.1f}%  ({d:+.1f})'
    ax.text(val + 0.08, bar.get_y() + bar.get_height()/2,
            txt, va='center', fontsize=11, fontweight=fw)

ax.set_yticks(range(len(systems)))
ax.set_yticklabels(systems, fontsize=10)
ax.set_xlabel('Test macro-F1 (%)')
ax.set_title('Final Test Results — Cross-System')
ax.set_xlim(66.5, 71.5)
ax.invert_yaxis()
ax.axvline(x=68.2, color=GREY, linestyle=':', alpha=0.5)

# ∆ inlined with the value above

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_cross_system.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_cross_system.png'), bbox_inches='tight', dpi=150)
plt.close()
print("9. Cross-system saved.")

# =============================================================================
# 10. Round-1 combo results (final01-05) — stacking wins
# =============================================================================
combo_labels = ['final01\npos+ln', 'final02\npos+gelu', 'final03\npos+ln+gelu',
                'final04\npos+pref+ln+gelu', 'final05\n+lstm2']
combo_scores = [66.4, 65.7, 64.7, 67.4, 66.7]
deltas       = [v - 63.3 for v in combo_scores]   # vs exp02 (best single axis = +pos 63.3)

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(combo_labels))
colors_combo = [C_NN if v == max(combo_scores) else '#8FBFDB' for v in combo_scores]
bars = ax.bar(x, combo_scores, color=colors_combo, edgecolor='white', linewidth=0.8)

for bar, val, d in zip(bars, combo_scores, deltas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}\n(+{d:.1f})',
            ha='center', va='bottom', fontsize=9,
            color='#27AE60', fontweight='bold')

ax.axhline(y=63.3, color=GREY, linestyle=':', alpha=0.6)
ax.text(4.45, 63.5, 'best single axis\n(exp02 +pos)', fontsize=8, color=GREY, ha='right')

ax.set_xticks(x)
ax.set_xticklabels(combo_labels, fontsize=9)
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Round-1 Combo Runs — Stacking the Single-Axis Winners')
ax.set_ylim(60, 70)

# Mark the winner
winner_idx = combo_scores.index(max(combo_scores))
ax.annotate('round-1 champion', xy=(winner_idx, combo_scores[winner_idx]),
            xytext=(winner_idx + 0.3, 69.2), fontsize=9, fontweight='bold',
            color=C_NN,
            arrowprops=dict(arrowstyle='->', color=C_NN, lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig12_combos.pdf'), bbox_inches='tight', dpi=150)
plt.savefig(os.path.join(OUT, 'fig12_combos.png'), bbox_inches='tight', dpi=150)
plt.close()
print("10. Combos saved.")

print("\nAll 1.2 plots generated!")
