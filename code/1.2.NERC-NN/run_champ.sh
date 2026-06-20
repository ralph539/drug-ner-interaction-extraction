#!/usr/bin/env bash
# [MOD-1.2] Round-4: targeted tests on the champion config
# (`final2_03_e20_pref2_suf3`, devel 70.0 / test 68.8).
#
#   A. Seed stability — 3 new seeds of the exact champion config.
#      Tells us whether 70.0 is real or lucky.
#   B. One bigger-LSTM variant (hidden=300, fc=300) — never tried with
#      longer training + pref/suf tweaks.
#   C. ±2 epoch sweep (18, 22) around the 20-epoch sweet spot.
set -u
cd "$(dirname "$0")"
source ../../.venv/bin/activate

RESULTS_DIR=results
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/CHAMP.log"
: > "$LOG"

run_exp () {
    local name="$1"; shift
    local extra="$*"
    [[ "$extra" == *"batch_size="* ]] || extra="batch_size=32 $extra"
    [[ "$extra" == *"epochs="* ]]     || extra="epochs=20 $extra"
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

# champion = final2_03_e20_pref2_suf3
CHAMP="use_pos=1 use_pref=1 use_layernorm=1 activation=gelu emb_dropout=0.2 pref_len=2 suf_len=3"

# A. seed stability (epochs=20)
run_exp champ_seed111  $CHAMP seed=111
run_exp champ_seed777  $CHAMP seed=777
run_exp champ_seed42   $CHAMP seed=42

# B. bigger LSTM on champion
run_exp champ_big      $CHAMP lstm_hidden=300 fc_hidden=300

# C. ±2 epoch sweep around 20
run_exp champ_e18      $CHAMP epochs=18
run_exp champ_e22      $CHAMP epochs=22

echo "[CHAMP-DONE] $(date +%H:%M:%S)" | tee -a "$LOG"
