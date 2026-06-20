#!/usr/bin/env bash
# [MOD-1.2] Experiment batch runner for System 1.2.
# Runs each configuration with train+predict on devel, logging start/end markers
# so the Monitor wrapper can emit progress events.
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

# ------------- experiment list -------------
run_exp exp01_baseline
run_exp exp02_pos          use_pos=1
run_exp exp03_pref         use_pref=1
run_exp exp04_lemma        use_lemma=1
run_exp exp05_all_inputs   use_pref=1 use_lemma=1 use_pos=1
run_exp exp06_lstm2        lstm_layers=2 lstm_dropout=0.3
run_exp exp07_layernorm    use_layernorm=1
run_exp exp08_adamw        optimizer=adamw learning_rate=0.0005 weight_decay=0.01
run_exp exp09_big          lstm_hidden=300 fc_hidden=300
run_exp exp10_gelu         activation=gelu emb_dropout=0.2

echo "[BATCH-DONE] $(date +%H:%M:%S)"
