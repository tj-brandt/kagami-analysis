#!/bin/bash

# ==============================================================================
# Master Analysis Script for the Kagami Project
#
# This script runs all reproducible analyses in a logical order.
#
# USAGE:
# 1. Make sure you have installed all dependencies from requirements.txt
#    pip install -r requirements.txt
#
# 2. Make this script executable from your terminal:
#    chmod +x run_all.sh
#
# 3. Run the script from the project's root directory:
#    ./run_all.sh
# ==============================================================================

# Exit immediately if any command fails, to prevent partial or incorrect results.
set -e

# --- SETUP ---
# Navigate into the src directory so all python scripts can find their data files
# using relative paths like ../data/
cd src

echo "--- Starting Kagami Project Analysis Pipeline ---"
echo "This will regenerate all tables and figures in the ../reports/ directory."
echo ""


# --- 1. SENSITIVITY & POWER ANALYSIS ---
echo "[1/6] Running Power & Sensitivity Analysis..."
python power_sensitivity.py
echo "Done."
echo ""


# --- 2. CONFIRMATORY ANALYSIS (Preregistered Hypotheses & Figure 3) ---
echo "[2/6] Running Confirmatory Analysis (H1-H3 & Figure 3)..."
python confirmatory_analysis.py
echo "Done."
echo ""


# --- 3. CONTENT ANALYSIS (Figures 5, 6, 7) ---
echo "[3/6] Running Conversational Content Analysis (Figures 5, 6, 7)..."
python content_analysis.py
echo "Done."
echo ""


# --- 4. EXPLORATORY ANALYSIS (Figures 8, 9) ---
echo "[4/6] Running Exploratory Analysis (Figures 8, 9)..."
python exploratory_analysis.py
echo "Done."
echo ""


# --- 5. LINGUISTIC AUDIT (Supplementary Tables) ---
echo "[5/6] Running Linguistic Audit..."
python linguistic_audit.py
echo "Done."
echo ""


# --- 6. ROBUSTNESS CHECKS (Supplementary Tables) ---
echo "[6/6] Running Robustness Checks..."
python robustness_checks.py
echo "Done."
echo ""


# --- COMPLETION ---
# Navigate back to the root directory
cd ..
echo "--- Analysis Pipeline Complete! ---"
echo "All tables and figures have been successfully regenerated in the /reports/ directory."