#!/bin/bash

# ==========================================
# Evaluate best hyperparameter configs on TEST set
# ==========================================
# Selected models based on devel set performance across
# three feature configurations: mod6, mod8, mod9_wo8
#
# Feature flag mapping:
#   mod6     → ADD7=False ADD8=False ADD9=False
#   mod8     → ADD7=True  ADD8=True  ADD9=False
#   mod9_wo8 → ADD7=True  ADD8=False ADD9=True

FEAT_FILE="bin/extract_features.py"
TEST_RESULTS_DIR="test_results"
LOG_FILE="test_results_best_models.csv"

mkdir -p "$TEST_RESULTS_DIR"
echo "Rank,FeatureSet,Model,Params,Devel_F1,Test_F1" > "$LOG_FILE"

# ==========================================
# Helper: set feature flags via sed
# ==========================================
set_features() {
    local add7=$1 add8=$2 add9=$3
    sed -i "s/ENABLE_ADDITION_7 = .*/ENABLE_ADDITION_7 = $add7/" "$FEAT_FILE"
    sed -i "s/ENABLE_ADDITION_8 = .*/ENABLE_ADDITION_8 = $add8/" "$FEAT_FILE"
    sed -i "s/ENABLE_ADDITION_9 = .*/ENABLE_ADDITION_9 = $add9/" "$FEAT_FILE"
    echo "  Features set: ADD7=$add7 ADD8=$add8 ADD9=$add9"
}

# ==========================================
# Helper: train, predict on test, extract score
# ==========================================
run_test() {
    local rank=$1
    local feat_set=$2
    local model=$3
    local devel_f1=$4
    local label=$5
    shift 5
    local params=("$@")

    echo "--------------------------------------------"
    echo "[$rank] $feat_set | $model | ${params[*]} | devel=$devel_f1"
    echo "--------------------------------------------"

    # Train
    python bin/run.py train $model "${params[@]}"

    # Predict on test
    python bin/run.py predict $model test "${params[@]}"

    # Move results
    local dest="$TEST_RESULTS_DIR/${rank}_${feat_set}_${model}_${label}"
    mkdir -p "$dest"
    if [ -d "results" ]; then
        cp results/test-${model}.* "$dest/" 2>/dev/null
    fi

    # Extract score
    local score="N/A"
    if [ -f "$dest/test-${model}.stats" ]; then
        score=$(grep "^M\.avg[[:space:]]" "$dest/test-${model}.stats" | awk '{print $NF}')
    fi
    echo "$rank,$feat_set,$model,${params[*]},$devel_f1,$score" >> "$LOG_FILE"
    echo "  → Test F1: $score"
    echo ""
}

# ==========================================
# MOD6: ADD7=False ADD8=False ADD9=False
# ==========================================
echo "============================================"
echo "Setting up MOD6 features..."
echo "============================================"
set_features "False" "False" "False"
python bin/run.py extract test

# 1. CRF c1=0.01, c2=0.1, iter=50 → devel 68.5%
run_test 1 mod6 CRF "68.5%" "c1_0.01_c2_0.1_iter_50" \
    c1=0.01 c2=0.1 max_iterations=50

# 2. CRF c1=0.1, c2=0.1, iter=50 → devel 68.4%
run_test 2 mod6 CRF "68.4%" "c1_0.1_c2_0.1_iter_50" \
    c1=0.1 c2=0.1 max_iterations=50

# 3. CRF c1=0.01, c2=0.1, iter=500 → devel 68.2%
run_test 3 mod6 CRF "68.2%" "c1_0.01_c2_0.1_iter_500" \
    c1=0.01 c2=0.1 max_iterations=500

# 4. CRF c1=0.01, c2=1.0, iter=50 → devel 68.2%
run_test 4 mod6 CRF "68.2%" "c1_0.01_c2_1.0_iter_50" \
    c1=0.01 c2=1.0 max_iterations=50

# 5. CRF c1=0.1, c2=0.5, iter=100 → devel 68.2%
run_test 5 mod6 CRF "68.2%" "c1_0.1_c2_0.5_iter_100" \
    c1=0.1 c2=0.5 max_iterations=100

# 6. SVM C=0.1, linear → devel 67.5%
run_test 6 mod6 SVM "67.5%" "C_0.1_linear" \
    C=0.1 kernel=linear

# ==========================================
# MOD9_WO8: ADD7=True ADD8=False ADD9=True
# ==========================================
echo "============================================"
echo "Setting up MOD9_WO8 features..."
echo "============================================"
set_features "True" "False" "True"
python bin/run.py extract test

# 7. CRF c1=0.1, c2=0.5, iter=200 → devel 68.3%
run_test 7 mod9_wo8 CRF "68.3%" "c1_0.1_c2_0.5_iter_200" \
    c1=0.1 c2=0.5 max_iterations=200

# 8. CRF c1=0.01, c2=0.1, iter=100 → devel 68.2%
run_test 8 mod9_wo8 CRF "68.2%" "c1_0.01_c2_0.1_iter_100" \
    c1=0.01 c2=0.1 max_iterations=100

# 9. CRF c1=0.01, c2=0.5, iter=50 → devel 68.2%
run_test 9 mod9_wo8 CRF "68.2%" "c1_0.01_c2_0.5_iter_50" \
    c1=0.01 c2=0.5 max_iterations=50

# 10. SVM C=0.1, linear → devel 67.7%
run_test 10 mod9_wo8 SVM "67.7%" "C_0.1_linear" \
    C=0.1 kernel=linear

# ==========================================
# MOD8: ADD7=True ADD8=True ADD9=False
# ==========================================
echo "============================================"
echo "Setting up MOD8 features..."
echo "============================================"
set_features "True" "True" "False"
python bin/run.py extract test

# 11. SVM C=1.0, rbf → devel 67.7%
run_test 11 mod8 SVM "67.7%" "C_1.0_rbf" \
    C=1.0 kernel=rbf

# ==========================================
# Restore features to mod9_wo8 (current default)
# ==========================================
set_features "True" "False" "True"

echo "============================================"
echo "DONE! Results in $LOG_FILE"
echo "============================================"
cat "$LOG_FILE"
