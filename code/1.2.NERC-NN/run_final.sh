#!/usr/bin/env bash
# [MOD-1.2] Round-3: combine the winners from run_extra.sh to try to
# push past long02_final04_e20 (69.0% devel). New knowledge exploited:
#   - suf_len=3 helps (exp11: +3.4 vs default 5)
#   - pref_len=2 beats pref_len=3 (exp13 > exp03)
#   - more epochs help (8→15→20: 67.4→68.3→69.0)
set -u
cd "$(dirname "$0")"
source ../../.venv/bin/activate

RESULTS_DIR=results
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/FINAL2.log"
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

# Shared base = final04 @ 20 epochs (current best devel)
BASE="use_pos=1 use_pref=1 use_layernorm=1 activation=gelu emb_dropout=0.2"

# Combine winners one at a time and then all together
run_exp final2_01_e20_pref2      $BASE pref_len=2
run_exp final2_02_e20_suf3       $BASE suf_len=3
run_exp final2_03_e20_pref2_suf3 $BASE pref_len=2 suf_len=3
run_exp final2_04_e25            $BASE epochs=25
run_exp final2_05_e25_pref2_suf3 $BASE pref_len=2 suf_len=3 epochs=25

echo "[FINAL2-DONE] $(date +%H:%M:%S)" | tee -a "$LOG"
