"""Shared palette / helpers for the System 2.x (DDI) report figures.

Imported by generate_plots_2_1.py / _2_2.py / _2_3.py / _2_cross.py.
Keeps the colour scheme and styling identical across every Part-2 figure,
mirroring the conventions established for Part 1 (generate_plots_1_*.py).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── System colours (Part-2 DDI) ──────────────────────────────────────
C_BASE  = '#888888'    # grey       — 2.0 rule-based baseline
C_ML    = '#2E86AB'    # blue       — 2.1 Machine Learning
C_NN    = '#009E73'    # green      — 2.2 Neural Network
C_LLAMA = '#D55E00'    # orange     — 2.3 Llama 3.2 3B
C_QWEN  = '#5E3C99'    # purple     — 2.3 Qwen 2.5 3B
C_BEST  = '#CC0066'    # hot-pink   — champion / best highlight
C_GOOD  = '#007A33'    # green      — positive delta
C_BAD   = '#D55E00'    # red-orange — negative delta / collapse
C_FS    = '#F18F01'    # amber      — few-shot
C_FT    = '#2E86AB'    # blue       — fine-tune
C_BG    = '#F5F5F0'
GREY    = '#888888'

# ── Per-class colours (fixed so figures cross-reference) ──────────────
# DDI positive classes: advise, effect, int, mechanism
CLASS_COLORS = {
    'advise'    : '#E69F00',   # amber
    'effect'    : '#2E86AB',   # blue
    'int'       : '#CC0066',   # hot-pink  (rarest class — visually distinct)
    'mechanism' : '#009E73',   # green
}
CLASSES = ['advise', 'effect', 'int', 'mechanism']

# ── Vibrant categorical palette for single-series bar charts ──────────
# Used so every bar in a sweep gets its own distinct, consistent colour
# instead of a monochrome wall. Colour-blind-aware, high saturation.
PALETTE = [
    '#2E86AB',  # blue
    '#E69F00',  # amber
    '#009E73',  # green
    '#CC0066',  # hot-pink
    '#5E3C99',  # purple
    '#D55E00',  # orange
    '#2A8C8C',  # teal
    '#B22222',  # brick-red
    '#0072B2',  # deep-blue
    '#F18F01',  # tangerine
    '#7B5E3B',  # brown
    '#56B4E9',  # sky
]


def palette(n, start=0):
    """Return n distinct colours cycling the categorical palette."""
    return [PALETTE[(start + i) % len(PALETTE)] for i in range(n)]

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': C_BG,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(OUT, f'{name}.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
