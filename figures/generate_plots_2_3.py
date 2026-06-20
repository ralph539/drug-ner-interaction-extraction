"""Figures for System 2.3 — DDI with Large Language Models (LoRA FT).

All numbers traceable to code/2.3.DDI-LLM/NOTES.md and the .stats /
confmat / error_strata files in code/2.3.DDI-LLM/results/.

Figures (PDF + PNG in report/):
    fig23_fs_sweep         few-shot k sweep (Llama / prompts01)
    fig23_fs_vs_ft         few-shot -> fine-tune headline jump
    fig23_ft_configs       FT config grid (rank x model x prompt, devel+test)
    fig23_prompt_crossmodel the opposite-effect prompt finding (Llama vs Qwen)
    fig23_confmat          confusion matrix heat-map (champion, test)
    fig23_epoch_ablation   Qwen prompts02 3ep vs 10ep (collapse != overfit)
    fig23_length_filter    length-filter post-processing (negative result)
    fig23_strata           error stratification by length & source
"""
import numpy as np
from generate_plots_2_common import (
    plt, save, CLASSES, CLASS_COLORS, palette,
    C_LLAMA, C_QWEN, C_FS, C_FT, C_BEST, C_GOOD, C_BAD, GREY)

# =============================================================================
# 1. fig23_fs_sweep — few-shot k sweep (llama32B3 / prompts01 / quant, devel)
# =============================================================================
shots = [0, 3, 5, 10, 15]
MF1   = [9.5, 21.4, 21.0, 23.2, 21.8]   # macro-F1
mF1   = [8.2, 18.7, 18.7, 17.4, 22.0]   # micro-F1

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(shots, MF1, 'o-', color=C_LLAMA, linewidth=2.2, markersize=8, label='macro-F1')
ax.plot(shots, mF1, 's--', color=C_QWEN, linewidth=2.0, markersize=7, label='micro-F1')
for k, m in zip(shots, MF1):
    ax.annotate(f'{m:.1f}', xy=(k, m), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, color=C_LLAMA, fontweight='bold')
ax.set_xticks(shots)
ax.set_xlabel('Number of few-shot demonstrations  (k)')
ax.set_ylabel('Devel F1 (%)')
ax.set_title('Few-shot k sweep — Llama 3.2 3B / prompts01 / 4-bit quant\n'
             'Few-shot plateaus around M=23 — far below fine-tuning',
             fontsize=11)
ax.set_ylim(0, 30)
ax.legend(loc='lower right', fontsize=9)
save(fig, 'fig23_fs_sweep')


# =============================================================================
# 2. fig23_fs_vs_ft — few-shot -> fine-tune headline jump (devel macro-F1)
# =============================================================================
configs = ['FS k=10\n(best FS)', 'FT r=8\nllama', 'FT r=32\nllama p01', 'FT r=32\nllama p02']
vals    = [23.2, 28.9, 33.8, 41.6]
colors  = palette(3) + [C_BEST]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
bars = ax.bar(configs, vals, color=colors, edgecolor='white')
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.4, f'{v:.1f}', ha='center', fontsize=10,
            fontweight='bold' if v == max(vals) else 'normal',
            color=C_BEST if v == max(vals) else 'black')
ax.annotate('+18.4 pp\nFS → best FT',
            xy=(3, 41.6), xytext=(1.4, 38), ha='center', fontsize=10,
            color=C_GOOD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_GOOD, alpha=0.7))
ax.set_ylabel('Devel macro-F1 (%)')
ax.set_title('Few-shot → fine-tune — each step of the LoRA campaign (Llama)')
ax.set_ylim(0, 48)
save(fig, 'fig23_fs_vs_ft')


# =============================================================================
# 3. fig23_ft_configs — FT config grid (devel + test macro-F1)
# =============================================================================
cfg = ['llama r=8', 'qwen r=8', 'llama r=32\np01', 'qwen r=32\np01',
       'llama r=32\np02', 'qwen r=32\np02']
dev = [28.9, 33.7, 33.8, 32.5, 41.6, 28.8]
tst = [30.5, 27.6, 35.2, 29.1, 39.8, 26.0]
barcol = [C_LLAMA, C_QWEN, C_LLAMA, C_QWEN, C_BEST, C_QWEN]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(cfg)); w = 0.38
ax.bar(x-w/2, dev, w, color=barcol, alpha=0.55, edgecolor='white', label='devel')
ax.bar(x+w/2, tst, w, color=barcol, edgecolor='white', label='test')
for i in range(len(cfg)):
    ax.text(i-w/2, dev[i]+0.4, f'{dev[i]:.1f}', ha='center', fontsize=8.5)
    ax.text(i+w/2, tst[i]+0.4, f'{tst[i]:.1f}', ha='center', fontsize=8.5,
            fontweight='bold' if tst[i] == max(tst) else 'normal',
            color=C_BEST if tst[i] == max(tst) else 'black')
# champion highlight
champ = tst.index(max(tst))
ax.axvspan(champ-0.5, champ+0.5, color=C_BEST, alpha=0.08, zorder=0)
ax.set_xticks(x); ax.set_xticklabels(cfg, fontsize=9.5)
ax.set_ylabel('macro-F1 (%)')
ax.set_title('All fine-tune configurations — rank, model family and prompt\n'
             'Champion: Llama r=32 / prompts02 (test M=39.8). prompts02 helps Llama but hurts Qwen',
             fontsize=11)
ax.set_ylim(0, 48)
ax.legend(loc='upper right', fontsize=9)
save(fig, 'fig23_ft_configs')


# =============================================================================
# 4. fig23_prompt_crossmodel — opposite-effect prompt finding (test macro-F1)
# =============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6),
                               gridspec_kw={'width_ratios': [1.1, 1]})

# Panel A — prompts01 -> prompts02 for both models (slope)
axL.plot([0, 1], [35.2, 39.8], 'o-', color=C_LLAMA, linewidth=2.6, markersize=11,
         label='Llama r=32  (Δ=+4.6)')
axL.plot([0, 1], [29.1, 25.9], 's-', color=C_QWEN, linewidth=2.6, markersize=11,
         label='Qwen r=32  (Δ=−3.2)')
for xx, yy, c in [(0, 35.2, C_LLAMA), (1, 39.8, C_LLAMA), (0, 29.1, C_QWEN), (1, 25.9, C_QWEN)]:
    axL.annotate(f'{yy:.1f}', xy=(xx, yy), xytext=(0, 10 if c == C_LLAMA else -16),
                 textcoords='offset points', ha='center', fontsize=9.5, color=c, fontweight='bold')
axL.set_xticks([0, 1]); axL.set_xticklabels(['prompts01', 'prompts02\n(typo fix + null emphasis)'])
axL.set_xlim(-0.3, 1.3); axL.set_ylim(20, 45)
axL.set_ylabel('Test macro-F1 (%)')
axL.set_title('Same prompt, opposite sign across model families')
axL.legend(loc='upper center', fontsize=9)

# Panel B — null over-prediction rate (the mechanism)
models = ['Llama\np01', 'Llama\np02', 'Qwen\np01', 'Qwen\np02']
overpred = [36.6, 28.8, 49.6, 64.0]
ocol = [C_LLAMA, C_LLAMA, C_QWEN, C_QWEN]
xb = np.arange(len(models))
bars = axR.bar(xb, overpred, color=ocol, edgecolor='white')
for b, a in zip(bars, [0.55, 1.0, 0.55, 1.0]):
    b.set_alpha(a)
for b, v in zip(bars, overpred):
    axR.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')
axR.annotate('prompt ↓ over-pred', xy=(0.5, 33), xytext=(0.5, 18), ha='center',
             fontsize=8.5, color=C_GOOD, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_GOOD))
axR.annotate('prompt ↑ over-pred', xy=(2.5, 58), xytext=(2.5, 72), ha='center',
             fontsize=8.5, color=C_BAD, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_BAD))
axR.set_xticks(xb); axR.set_xticklabels(models, fontsize=9)
axR.set_ylabel('% of null pairs predicted positive')
axR.set_ylim(0, 80)
axR.set_title('Mechanism: prompts02 moves over-prediction\nin opposite directions')

fig.suptitle('Prompt brittleness across model families — the stronger null instruction '
             'helps Llama, backfires on Qwen', fontsize=12, y=1.03)
save(fig, 'fig23_prompt_crossmodel')


# =============================================================================
# 5. fig23_confmat — confusion matrix (champion Llama r=32 prompts02, test)
# =============================================================================
labels = ['advise', 'effect', 'int', 'mechanism', 'null']
cm = np.array([
    [163,   8,   0,   5,   33],
    [ 14, 247,   0,   6,   26],
    [  0,   6,  15,   1,   18],
    [  1,  69,   0, 199,   75],
    [331, 682,  35, 380, 3524],
])
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7.6, 6))
im = ax.imshow(cm_norm, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                color='white' if cm_norm[i, j] > 0.5 else 'black',
                fontsize=10, fontweight='bold' if i == j else 'normal')
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('PREDICTED'); ax.set_ylabel('GOLD')
ax.set_title('Confusion matrix — champion Llama r=32 / prompts02 (test)\n'
             '1428 / 4952 null pairs (28.8%) over-predicted as positive',
             fontsize=11)
for i in range(5):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                               edgecolor=C_BEST, linewidth=2.0))
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('row-normalised fraction')
save(fig, 'fig23_confmat')


# =============================================================================
# 6. fig23_epoch_ablation — Qwen prompts02 3ep vs 10ep (collapse != overfit)
# =============================================================================
metrics = ['devel\nmacro-F1', 'test\nmacro-F1', 'null over-pred\n(% of nulls)']
ep3  = [30.9, 25.9, 62.0]
ep10 = [28.8, 26.0, 73.2]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(metrics)); w = 0.36
ax.bar(x-w/2, ep3,  w, color=C_QWEN, alpha=0.6, edgecolor='white', label='3 epochs')
ax.bar(x+w/2, ep10, w, color=C_QWEN, edgecolor='white', label='10 epochs')
for i in range(len(metrics)):
    ax.text(i-w/2, ep3[i]+1, f'{ep3[i]:.1f}', ha='center', fontsize=9)
    ax.text(i+w/2, ep10[i]+1, f'{ep10[i]:.1f}', ha='center', fontsize=9)
ax.annotate('test F1 identical\n(25.9 ≈ 26.0)\n→ not overfitting',
            xy=(1, 26), xytext=(1.0, 45), ha='center', fontsize=9,
            color=C_GOOD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_GOOD, alpha=0.7))
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel('value')
ax.set_title('Qwen prompts02 epoch ablation — the over-prediction collapse is\n'
             'present at epoch 3; more epochs only raise over-prediction, not F1',
             fontsize=11)
ax.set_ylim(0, 85)
ax.legend(loc='upper left', fontsize=9)
save(fig, 'fig23_epoch_ablation')


# =============================================================================
# 7. fig23_length_filter — length-filter post-processing (negative result)
# =============================================================================
filt = ['no filter', 'drop >50w', 'drop >40w', 'drop >30w']
p02  = [39.8, 39.6, 39.0, 38.5]
p01  = [35.2, 35.2, 34.8, 34.7]

fig, ax = plt.subplots(figsize=(8.5, 4.5))
x = np.arange(len(filt))
ax.plot(x, p02, 'o-', color=C_BEST, linewidth=2.2, markersize=9, label='Llama r=32 prompts02')
ax.plot(x, p01, 's--', color=C_LLAMA, linewidth=2.0, markersize=8, label='Llama r=32 prompts01')
for i in range(len(filt)):
    ax.annotate(f'{p02[i]:.1f}', xy=(i, p02[i]), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=8.5, color=C_BEST, fontweight='bold')
    ax.annotate(f'{p01[i]:.1f}', xy=(i, p01[i]), xytext=(0, -16), textcoords='offset points',
                ha='center', fontsize=8.5, color=C_LLAMA)
ax.set_xticks(x); ax.set_xticklabels(filt)
ax.set_ylabel('Test macro-F1 (%)')
ax.set_title('Length-filter post-processing — a negative result\n'
             'Forcing null on long sentences never helps: the model already self-suppresses there',
             fontsize=11)
ax.set_ylim(32, 43)
ax.legend(loc='lower left', fontsize=9)
save(fig, 'fig23_length_filter')


# =============================================================================
# 8. fig23_strata — error stratification (length & source, champion, test)
# =============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

ln = ['short\n(≤10)', 'medium\n(11-25)', 'long\n(26-50)', 'very long\n(>50)']
ln_M = [45.1, 44.2, 30.8, 12.3]
xL = np.arange(len(ln))
barsL = axL.bar(xL, ln_M, color=palette(3, start=4) + [C_BAD], edgecolor='white')
for b, v in zip(barsL, ln_M):
    axL.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.1f}', ha='center', fontsize=9,
             fontweight='bold' if v == min(ln_M) else 'normal',
             color=C_BAD if v == min(ln_M) else 'black')
axL.annotate('collapse on\nlong sentences',
             xy=(3, 12.3), xytext=(2.0, 30), ha='center', fontsize=9,
             color=C_BAD, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_BAD, alpha=0.7))
axL.set_xticks(xL); axL.set_xticklabels(ln)
axL.set_ylabel('Test macro-F1 (%)'); axL.set_ylim(0, 55)
axL.set_title('By sentence length')

src = ['DrugBank', 'MedLine']
src_M = [39.8, 44.0]
xS = np.arange(len(src))
barsS = axR.bar(xS, src_M, color=palette(2, start=1), edgecolor='white')
for b, v in zip(barsS, src_M):
    axR.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.1f}', ha='center', fontsize=9)
axR.set_xticks(xS); axR.set_xticklabels(src)
axR.set_ylabel('Test macro-F1 (%)'); axR.set_ylim(0, 55)
axR.set_title('By document source')

fig.suptitle('System 2.3 error stratification (champion, test) — long sentences are the binding constraint',
             fontsize=12, y=1.02)
save(fig, 'fig23_strata')


print('All fig23_*.pdf/.png written.')
