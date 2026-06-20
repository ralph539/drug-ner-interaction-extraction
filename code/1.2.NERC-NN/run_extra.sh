#!/usr/bin/env bash
# [MOD-1.2] Extra experiments batch — round 2.
# Covers the gaps flagged in README_MODIFICATIONS section 5:
#   A. Data sweeps:  suf_len, pref_len, batch_size, max_len
#   B. Pretrained word embeddings (spaCy en_core_web_md, 300d)
#   C. Seed stability on best devel config (final04)
#   D. Longer training on best config
#   E. Best-config architecture variations
#
# All runs share train.pck / devel.pck and EPOCHS=8 / BS=32 defaults unless
# overridden. Each run writes:  results/<name>.log  and  results/devel-<name>.stats
# A one-line [EXP-DONE] summary with macro-F1 lands on stdout and in
# results/EXTRA.log so we can grep at the end.
set -u
cd "$(dirname "$0")"
source ../../.venv/bin/activate

EPOCHS=${EPOCHS:-8}
BS=${BS:-32}
RESULTS_DIR=results
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/EXTRA.log"
: > "$LOG"

run_exp () {
    local name="$1"; shift
    local extra="$*"
    # if caller already passed batch_size=/epochs= in extra, honour that;
    # otherwise inject the defaults
    [[ "$extra" == *"batch_size="* ]] || extra="batch_size=$BS $extra"
    [[ "$extra" == *"epochs="* ]]     || extra="epochs=$EPOCHS $extra"
    echo "[EXP-START] $name $(date +%H:%M:%S) :: $extra" | tee -a "$LOG"
    python3 bin/run.py train predict name="$name" $extra \
        > "$RESULTS_DIR/$name.log" 2>&1
    local status=$?
    if [ $status -eq 0 ] && [ -f "$RESULTS_DIR/devel-$name.stats" ]; then
        local f1=$(awk '/^M.avg/ {print $(NF)}' "$RESULTS_DIR/devel-$name.stats")
        echo "[EXP-DONE ] $name $(date +%H:%M:%S) :: macro-F1=$f1" | tee -a "$LOG"
    else
        echo "[EXP-FAIL ] $name status=$status  see $RESULTS_DIR/$name.log" | tee -a "$LOG"
    fi
}

# =============================================================================
# A. DATA SWEEPS  (each varies ONE preprocessing knob, everything else default)
# =============================================================================
run_exp exp11_suf3        suf_len=3
run_exp exp12_suf7        suf_len=7
run_exp exp13_pref2       use_pref=1 pref_len=2
run_exp exp14_pref4       use_pref=1 pref_len=4
run_exp exp15_bs16        batch_size=16
run_exp exp16_bs64        batch_size=64
run_exp exp17_maxlen100   max_len=100
run_exp exp18_maxlen200   max_len=200

# =============================================================================
# B. PRETRAINED WORD EMBEDDINGS  (spaCy en_core_web_md, 300d GloVe-derived)
# =============================================================================
# B1. Baseline architecture + pretrained (fine-tune)
run_exp exp19_pretrained           pretrained=1
# B2. Frozen pretrained embeddings
run_exp exp20_pretrained_frozen    pretrained=1 pretrained_freeze=1
# B3. Pretrained + the winning input feature (PoS)
run_exp exp21_pretrained_pos       pretrained=1 use_pos=1
# B4. Pretrained on top of final04 full combo
run_exp exp22_pretrained_final04   pretrained=1 use_pos=1 use_pref=1 use_layernorm=1 activation=gelu emb_dropout=0.2

# =============================================================================
# C. SEED STABILITY on final04  (same config, 3 different seeds)
# =============================================================================
FINAL04="use_pos=1 use_pref=1 use_layernorm=1 activation=gelu emb_dropout=0.2"
run_exp seed01_final04_s111  $FINAL04 seed=111
run_exp seed02_final04_s777  $FINAL04 seed=777
run_exp seed03_final04_s42   $FINAL04 seed=42

# =============================================================================
# D. LONGER TRAINING on best configs
# =============================================================================
run_exp long01_final04_e15   $FINAL04 epochs=15
run_exp long02_final04_e20   $FINAL04 epochs=20
run_exp long03_final05_e15   $FINAL04 lstm_layers=2 lstm_dropout=0.3 epochs=15

# =============================================================================
# E. BEST-CONFIG ARCHITECTURE VARIATIONS
# =============================================================================
run_exp best01_final04_big       $FINAL04 lstm_hidden=300 fc_hidden=300
run_exp best02_final04_drop03    $FINAL04 emb_dropout=0.3
run_exp best03_final04_wd1e5     $FINAL04 weight_decay=1e-5
run_exp best04_final04_lstm2drop5 $FINAL04 lstm_layers=2 lstm_dropout=0.5
run_exp best05_final04_tanh      use_pos=1 use_pref=1 use_layernorm=1 activation=tanh emb_dropout=0.2

echo "[EXTRA-DONE] $(date +%H:%M:%S)" | tee -a "$LOG"
