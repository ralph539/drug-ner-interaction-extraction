#!/bin/bash

# Arrays of values
C1_VALUES=(0.01  0.1  0.5  1.0)
C2_VALUES=(0.1  0.5  1.0  5.0)
ITER_VALUES=(50 100 200 500 1000)
MODELS=("MEM" "CRF" "SVM")

# Base results folder
BASE_RESULTS="results"
RESULTS="results_mod6"

# Loop over models, c1, c2, and iterations
for C1 in "${C1_VALUES[@]}"; do
    for C2 in "${C2_VALUES[@]}"; do
        for ITER in "${ITER_VALUES[@]}"; do
            for MODEL in "${MODELS[@]}"; do 
                echo "Running model=$MODEL c1=$C1 c2=$C2 iter=$ITER"
                
                # Run the training command
                python bin/run.py train $MODEL c1=$C1 c2=$C2 max_iterations=$ITER
                python bin/run.py predict $MODEL c1=$C1 c2=$C2 max_iterations=$ITER
                
            done
            # Build new results folder name
            NEW_RESULTS="${RESULTS}_c1_${C1}_c2_${C2}_iter_${ITER}"
            
            # Move/rename the results folder
            if [ -d "$BASE_RESULTS" ]; then
                mv "$BASE_RESULTS" "$NEW_RESULTS"
                echo "Moved results to $NEW_RESULTS"
            else
                echo "Warning: $BASE_RESULTS folder not found!"
            fi
        done
    done
done