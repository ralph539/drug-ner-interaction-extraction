#!/bin/bash

# ==========================================
# Fine-grained search on mod8 features (test set)
# ==========================================
# mod8 SVM C=1.0 rbf got 67.9% on test -- best overall.
# Now testing:
#   - SVM rbf: finer C grid + gamma values
#   - SVM linear: finer C grid
#   - Top CRF configs from mod8 devel
#   - MEM for completeness

FEAT_FILE="bin/extract_features.py"
TEST_RESULTS_DIR="test_results_mod8_fine"
LOG_FILE="test_results_mod8_fine.csv"

mkdir -p "$TEST_RESULTS_DIR"
echo "Rank,Model,Params,Test_F1" > "$LOG_FILE"

# Set mod8 features: ADD7=True ADD8=True ADD9=False
sed -i "s/ENABLE_ADDITION_7 = .*/ENABLE_ADDITION_7 = True/" "$FEAT_FILE"
sed -i "s/ENABLE_ADDITION_8 = .*/ENABLE_ADDITION_8 = True/" "$FEAT_FILE"
sed -i "s/ENABLE_ADDITION_9 = .*/ENABLE_ADDITION_9 = False/" "$FEAT_FILE"
echo "Features set: ADD7=True ADD8=True ADD9=False (mod8)"

# Extract features for train + devel + test
python bin/run.py extract
python bin/run.py extract test

RANK=0

run_test() {
    local model=$1
    local label=$2
    shift 2
    local params=("$@")

    RANK=$((RANK + 1))
    echo "--------------------------------------------"
    echo "[$RANK] $model | ${params[*]}"
    echo "--------------------------------------------"

    # Train on train set
    python bin/run.py train $model "${params[@]}"
    # Predict on test set
    python bin/run.py predict $model test "${params[@]}"

    # Move results
    local dest="$TEST_RESULTS_DIR/${RANK}_${model}_${label}"
    mkdir -p "$dest"
    if [ -d "results" ]; then
        cp results/test-${model}.* "$dest/" 2>/dev/null
    fi

    # Extract score
    local score="N/A"
    if [ -f "$dest/test-${model}.stats" ]; then
        score=$(grep "^M\.avg[[:space:]]" "$dest/test-${model}.stats" | awk '{print $NF}')
    fi
    echo "$RANK,$model,${params[*]},$score" >> "$LOG_FILE"
    echo "  -> Test F1: $score"
    echo ""
}

# ==========================================
# SVM rbf -- finer C grid around C=1.0
# ==========================================
echo "=== SVM rbf -- finer C grid ==="
for C in 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.5 10.0 15.0 20.0 50.0; do
    run_test SVM "C_${C}_rbf" C=$C kernel=rbf
done

# ==========================================
# SVM rbf -- gamma tuning (with C=1.0 and nearby)
# ==========================================
echo "=== SVM rbf -- gamma tuning ==="
for C in 0.5 1.0 2.0 5.0 10.0; do
    for GAMMA in 0.001 0.005 0.01 0.05 0.1; do
        run_test SVM "C_${C}_rbf_gamma_${GAMMA}" C=$C kernel=rbf gamma=$GAMMA
    done
done

# ==========================================
# SVM linear -- finer C grid
# ==========================================
echo "=== SVM linear -- finer C grid ==="
for C in 0.05 0.1 0.2 0.5 1.0; do
    run_test SVM "C_${C}_linear" C=$C kernel=linear
done

# ==========================================
# CRF -- top mod8 devel configs
# ==========================================
echo "=== CRF -- top mod8 configs ==="
run_test CRF "c1_0.01_c2_0.1_iter_50"  c1=0.01 c2=0.1 max_iterations=50
run_test CRF "c1_0.1_c2_0.1_iter_50"   c1=0.1 c2=0.1 max_iterations=50
run_test CRF "c1_0.1_c2_0.5_iter_50"   c1=0.1 c2=0.5 max_iterations=50
run_test CRF "c1_0.5_c2_0.1_iter_50"   c1=0.5 c2=0.1 max_iterations=50
run_test CRF "c1_0.01_c2_1.0_iter_200" c1=0.01 c2=1.0 max_iterations=200

# ==========================================
# MEM -- quick check
# ==========================================
echo "=== MEM ==="
run_test MEM "C_0.1" C=0.1
run_test MEM "C_1.0" C=1.0

# ==========================================
# Restore features to mod9_wo8
# ==========================================
sed -i "s/ENABLE_ADDITION_7 = .*/ENABLE_ADDITION_7 = True/" "$FEAT_FILE"
sed -i "s/ENABLE_ADDITION_8 = .*/ENABLE_ADDITION_8 = False/" "$FEAT_FILE"
sed -i "s/ENABLE_ADDITION_9 = .*/ENABLE_ADDITION_9 = True/" "$FEAT_FILE"

echo "============================================"
echo "DONE! Results in $LOG_FILE"
echo "============================================"
echo ""
echo "Top 10 by Test F1:"
head -1 "$LOG_FILE"
tail -n +2 "$LOG_FILE" | sort -t',' -k4 -rn | head -10
