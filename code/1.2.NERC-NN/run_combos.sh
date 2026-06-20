#!/usr/bin/env bash
# [MOD-1.2] Combo runner: picks the best choice per axis and tries a few
# combinations. Launched after the baseline sweep.
set -u
cd "$(dirname "$0")"
source ../../.venv/bin/activate

EPOCHS=${EPOCHS:-8}
BS=${BS:-32}
RESULTS_DIR=results
mkdir -p "$RESULTS_DIR"

run_exp () {
    local name="$1"; shift
    local extra="$*"
    echo "[EXP-START] $name $(date +%H:%M:%S) :: $extra"
    python3 bin/run.py train predict \
        name="$name" epochs="$EPOCHS" batch_size="$BS" $extra \
        > "$RESULTS_DIR/$name.log" 2>&1
    local status=$?
    if [ $status -eq 0 ] && [ -f "$RESULTS_DIR/devel-$name.stats" ]; then
        local f1=$(awk '/^M.avg/ {print $(NF)}' "$RESULTS_DIR/devel-$name.stats")
        echo "[EXP-DONE ] $name $(date +%H:%M:%S) :: macro-F1=$f1"
    else
        echo "[EXP-FAIL ] $name status=$status  see $RESULTS_DIR/$name.log"
    fi
}

# --- single-axis winners (for reference) ------------------------------------
#   input features : use_pos=1              → exp02 63.3
#   architecture   : use_layernorm=1        → exp07 60.5
#   hyperparams    : gelu + emb_dropout=0.2 → exp10 62.7

# --- combinations -----------------------------------------------------------
run_exp final01_pos_ln        use_pos=1 use_layernorm=1
run_exp final02_pos_gelu      use_pos=1 activation=gelu emb_dropout=0.2
run_exp final03_pos_ln_gelu   use_pos=1 use_layernorm=1 activation=gelu emb_dropout=0.2
run_exp final04_pos_pref_ln_gelu use_pos=1 use_pref=1 use_layernorm=1 activation=gelu emb_dropout=0.2
run_exp final05_pos_lstm2_ln_gelu use_pos=1 use_layernorm=1 activation=gelu emb_dropout=0.2 lstm_layers=2 lstm_dropout=0.3

echo "[COMBO-DONE] $(date +%H:%M:%S)"
