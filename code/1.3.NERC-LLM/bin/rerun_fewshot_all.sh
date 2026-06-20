#!/bin/bash
# ============================================================
# Re-run all 12 few-shot configurations with fixed prompts.py
# (professor's 2026-04-23 notice about usrprompt-repetition bug).
# ------------------------------------------------------------
# Run from: /scratch/nas/1/ahlt1008/AHLT-project/code/1.3.NERC-LLM/bin/
# Usage:    bash rerun_fewshot_all.sh
#
# This script:
#   1. Backs up existing FS-*.stats / .out / .json to .pre-prompts-fix
#   2. Submits 12 sbatch jobs for every few-shot configuration
#      that appears in the report.
# ============================================================

set -e
cd "$(dirname "$0")"

RESULTS_DIR=../results

echo "=== Backing up pre-fix FS results ==="
for f in "$RESULTS_DIR"/FS-*.stats "$RESULTS_DIR"/FS-*.out "$RESULTS_DIR"/FS-*.json; do
    [ -f "$f" ] || continue
    [ -f "${f}.pre-prompts-fix" ] && { echo "  skip (already backed up): $f"; continue; }
    cp "$f" "${f}.pre-prompts-fix"
    echo "  backed up: $(basename "$f")"
done

echo ""
echo "=== Submitting 12 few-shot jobs with fixed prompts.py ==="

# Arg order for fewshot.sh: MODEL PROMPTS SHOTS TRAIN TEST QUANT [BAL]

# --- Llama, prompts01 sweep on devel ---
sbatch fewshot.sh llama32B3 prompts01.json  0 train devel -quant
sbatch fewshot.sh llama32B3 prompts01.json  3 train devel -quant
sbatch fewshot.sh llama32B3 prompts01.json  5 train devel -quant
sbatch fewshot.sh llama32B3 prompts01.json 10 train devel -quant
sbatch fewshot.sh llama32B3 prompts01.json 15 train devel -quant

# --- Llama, prompts01 15-shot, balanced sampler ---
sbatch fewshot.sh llama32B3 prompts01.json 15 train devel -quant -balanced

# --- Llama, prompts01 15-shot on test (final FS test eval) ---
sbatch fewshot.sh llama32B3 prompts01.json 15 train test -quant

# --- Llama, prompts02 at 5 and 15 shots ---
sbatch fewshot.sh llama32B3 prompts02.json  5 train devel -quant
sbatch fewshot.sh llama32B3 prompts02.json 15 train devel -quant

# --- Llama, prompts03 at 5 and 15 shots ---
sbatch fewshot.sh llama32B3 prompts03.json  5 train devel -quant
sbatch fewshot.sh llama32B3 prompts03.json 15 train devel -quant

# --- Qwen, prompts01 5-shot (cross-model check) ---
sbatch fewshot.sh qwen25B3  prompts01.json  5 train devel -quant

echo ""
echo "=== All 12 jobs submitted. ==="
echo "Monitor with:  squeue -u \$USER"
echo "Results in:    $RESULTS_DIR/FS-*"
