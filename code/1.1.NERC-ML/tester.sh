#!/bin/bash

# ==========================================
# 1. Setup Directories and Log File
# ==========================================
MOD="mod9_wo8"
BASE_RESULTS="results"
EXPERIMENT_DIR="experiments_$MOD"
LOG_FILE="hyperparameter_results_$MOD.csv"

# Create the experiments directory if it doesn't exist
mkdir -p "$EXPERIMENT_DIR"

# Initialize the log file with headers
echo "Model,Param1,Param2,Param3,Micro_Avg_F1" > "$LOG_FILE"

# ==========================================
# 2. Define Parameter Search Grids
# ==========================================
# CRF Parameters (pycrfsuite)
C1_VALUES=(0.01 0.1 0.5 1.0)
C2_VALUES=(0.1 0.5 1.0 5.0)
CRF_ITER_VALUES=(50 100 200 500)

# MEM Parameters (Logistic Regression)
MEM_C_VALUES=(0.01 0.1 1.0 10.0 100.0)
# MEM_ITER_VALUES=(100 500 1500)

# SVM Parameters (SVC)
SVM_C_VALUES=(0.1 1.0 10.0 100.0)
SVM_KERNELS=("linear" "rbf" "poly")

MODELS=("CRF" "MEM" "SVM")

python bin/run.py extract

# ==========================================
# 3. Main Tuning Loop
# ==========================================
for MODEL in "${MODELS[@]}"; do 
    echo "====================================="
    echo "Starting tuning for model: $MODEL"
    echo "====================================="

    case $MODEL in
        "CRF")
            for C1 in "${C1_VALUES[@]}"; do
                for C2 in "${C2_VALUES[@]}"; do
                    for ITER in "${CRF_ITER_VALUES[@]}"; do
                        echo "Running CRF C1=$C1 C2=$C2 Iterations=$ITER"
                        
                        # Train and Predict
                        python bin/run.py train $MODEL c1=$C1 c2=$C2 max_iterations=$ITER
                        python bin/run.py predict $MODEL c1=$C1 c2=$C2 max_iterations=$ITER
                        
                        # Organize Folders
                        NEW_RESULTS="$EXPERIMENT_DIR/$MODEL/c1_${C1}_c2_${C2}_iter_${ITER}"
                        mkdir -p "$NEW_RESULTS"
                        
                        if [ -d "$BASE_RESULTS" ]; then
                            mv "$BASE_RESULTS"/* "$NEW_RESULTS"/
                        fi
                        
                        # Extract Score and Log
                        SCORE=$(grep "^M\.avg[[:space:]]" "$NEW_RESULTS/devel-${MODEL}.stats" | awk '{print $NF}')
                        echo "$MODEL,c1=$C1,c2=$C2,iter=$ITER,$SCORE" >> "$LOG_FILE"
                    done
                done
            done
            ;;
            
        "MEM")
            for C in "${MEM_C_VALUES[@]}"; do
                echo "Running MEM C=$C (using default max_iter of 1500)"
                
                # Notice we removed max_iter=$ITER from these lines
                python bin/run.py train $MODEL C=$C
                python bin/run.py predict $MODEL C=$C
                
                NEW_RESULTS="$EXPERIMENT_DIR/$MODEL/C_${C}"
                mkdir -p "$NEW_RESULTS"
                
                if [ -d "$BASE_RESULTS" ]; then
                    mv "$BASE_RESULTS"/* "$NEW_RESULTS"/
                fi
                
                SCORE=$(grep "^M\.avg[[:space:]]" "$NEW_RESULTS/devel-${MODEL}.stats" | awk '{print $NF}')
                echo "$MODEL,C=$C,N/A,N/A,$SCORE" >> "$LOG_FILE"
            done
            ;;
            
        "SVM")
            for C in "${SVM_C_VALUES[@]}"; do
                for KERNEL in "${SVM_KERNELS[@]}"; do
                    echo "Running SVM C=$C Kernel=$KERNEL"
                    
                    # Train and Predict
                    python bin/run.py train $MODEL C=$C kernel=$KERNEL
                    python bin/run.py predict $MODEL C=$C kernel=$KERNEL
                    
                    # Organize Folders
                    NEW_RESULTS="$EXPERIMENT_DIR/$MODEL/C_${C}_kernel_${KERNEL}"
                    mkdir -p "$NEW_RESULTS"
                    
                    if [ -d "$BASE_RESULTS" ]; then
                        mv "$BASE_RESULTS"/* "$NEW_RESULTS"/
                    fi
                    
                    # Extract Score and Log
                    SCORE=$(grep "^M\.avg[[:space:]]" "$NEW_RESULTS/devel-${MODEL}.stats" | awk '{print $NF}')
                    echo "$MODEL,C=$C,kernel=$KERNEL,N/A,$SCORE" >> "$LOG_FILE"
                done
            done
            ;;
    esac
done

echo "====================================="
echo "Tuning Complete! Check $LOG_FILE for results."
echo "====================================="