#!/bin/bash

# ==========================================
# Fast mod8 tests: CRF + SVM linear + MEM
# (skipping SVM rbf gamma tuning -- too slow)
# ==========================================

FEAT_FILE="bin/extract_features.py"
TEST_RESULTS_DIR="test_results_mod8_fine"
LOG_FILE="test_results_mod8_fine.csv"

mkdir -p "$TEST_RESULTS_DIR"

# Features should already be extracted from previous run,
# but set flags just in case
sed -i "s/ENABLE_ADDITION_7 = .*/ENABLE_ADDITION_7 = True/" "$FEAT_FILE"
sed -i "s/ENABLE_ADDITION_8 = .*/ENABLE_ADDITION_8 = True/" "$FEAT_FILE"
sed -i "s/ENABLE_ADDITION_9 = .*/ENABLE_ADDITION_9 = False/" "$FEAT_FILE"

# Check if features exist, re-extract if needed
if [ ! -f "preprocessed/test.feat" ]; then
    python bin/run.py extract
    python bin/run.py extract test
fi

RANK=$(tail -n +2 "$LOG_FILE" | wc -l)

run_test() {
    local model=$1
    local label=$2
    shift 2
    local params=("$@")

    RANK=$((RANK + 1))
    echo "[$RANK] $model | ${params[*]}"

    python bin/run.py train $model "${params[@]}"
    python bin/run.py predict $model test "${params[@]}"

    local dest="$TEST_RESULTS_DIR/${RANK}_${model}_${label}"
    mkdir -p "$dest"
    if [ -d "results" ]; then
        cp results/test-${model}.* "$dest/" 2>/dev/null
    fi

    local score="N/A"
    if [ -f "$dest/test-${model}.stats" ]; then
        score=$(grep "^M\.avg[[:space:]]" "$dest/test-${model}.stats" | awk '{print $NF}')
    fi
    echo "$RANK,$model,${params[*]},$score" >> "$LOG_FILE"
    echo "  -> Test F1: $score"
}

# SVM linear
for C in 0.05 0.1 0.2 0.5 1.0; do
    run_test SVM "C_${C}_linear" C=$C kernel=linear
done

# CRF top configs
run_test CRF "c1_0.01_c2_0.1_iter_50"  c1=0.01 c2=0.1 max_iterations=50
run_test CRF "c1_0.1_c2_0.1_iter_50"   c1=0.1 c2=0.1 max_iterations=50
run_test CRF "c1_0.1_c2_0.5_iter_50"   c1=0.1 c2=0.5 max_iterations=50
run_test CRF "c1_0.5_c2_0.1_iter_50"   c1=0.5 c2=0.1 max_iterations=50
run_test CRF "c1_0.01_c2_1.0_iter_200" c1=0.01 c2=1.0 max_iterations=200

# MEM
run_test MEM "C_0.1" C=0.1
run_test MEM "C_1.0" C=1.0

# Restore to mod9_wo8
sed -i "s/ENABLE_ADDITION_7 = .*/ENABLE_ADDITION_7 = True/" "$FEAT_FILE"
sed -i "s/ENABLE_ADDITION_8 = .*/ENABLE_ADDITION_8 = False/" "$FEAT_FILE"
sed -i "s/ENABLE_ADDITION_9 = .*/ENABLE_ADDITION_9 = True/" "$FEAT_FILE"

echo ""
echo "DONE! All results:"
cat "$LOG_FILE"
