#!/usr/bin/env bash
# [MOD-1.2] Round-5: confirm whether champ_big's 70.6% is real signal
# or seed-2345 luck. Re-run it with seeds 777 and 42 (known "average"
# and "low" from round-4) so we get 3 seeds total.
set -u
cd "$(dirname "$0")"
source ../../.venv/bin/activate

RESULTS_DIR=results
LOG="$RESULTS_DIR/BIG_SEEDS.log"
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

BIG="use_pos=1 use_pref=1 use_layernorm=1 activation=gelu emb_dropout=0.2 pref_len=2 suf_len=3 lstm_hidden=300 fc_hidden=300"

run_exp champ_big_s777  $BIG seed=777
run_exp champ_big_s42   $BIG seed=42

echo "[BIG_SEEDS-DONE] $(date +%H:%M:%S)" | tee -a "$LOG"
