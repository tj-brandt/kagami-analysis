# /src/power_sensitivity.py

"""
Power and Sensitivity Analysis
==============================

This script conducts a post-hoc sensitivity analysis to determine the minimum
effect size the study was adequately powered to detect, given the final sample
size, alpha level, and desired statistical power.

This analysis helps in interpreting non-significant results by establishing the
magnitude of an effect that could have been plausibly missed.
"""

import pandas as pd
import os
import numpy as np
import logging
from statsmodels.stats.power import FTestAnovaPower

# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)

# --- Analysis Parameters ---
# These reflect the final state of the 3x2 factorial experiment.
N_PARTICIPANTS = 162
N_GROUPS = 6  # From the 3 (Avatar) x 2 (LSM) design
ALPHA = 0.05
POWER = 0.80

# --- Output Paths ---
REPORTS_DIR = '../reports'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'power_sensitivity_log.txt')

# --- Setup Logging and Directories ---
os.makedirs(TABLES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

logging.info("--- Starting power_sensitivity.py ---")


# --- 2. SENSITIVITY ANALYSIS ---
def run_sensitivity_analysis():
    """
    Calculates and reports the minimum detectable effect size (as Cohen's f
    and partial eta-squared) for the study's ANOVA design.
    """
    logging.info("\nRunning post-hoc sensitivity analysis for ANOVA...")
    logging.info(f"Parameters: N={N_PARTICIPANTS}, groups={N_GROUPS}, alpha={ALPHA}, power={POWER}")

    # Initialize the power analysis object for ANOVA
    power_analyzer = FTestAnovaPower()

    # Use solve_power to find the effect size (f) given other parameters
    detectable_f = power_analyzer.solve_power(
        k_groups=N_GROUPS,
        nobs=N_PARTICIPANTS,
        alpha=ALPHA,
        power=POWER
    )

    # Convert Cohen's f to partial eta-squared (η²p)
    # For a one-way ANOVA design context (which FTestAnovaPower assumes),
    # this formula provides a close approximation.
    detectable_eta_squared_p = detectable_f**2 / (1 + detectable_f**2)

    # --- 3. REPORTING ---
    logging.info("\n--- Results ---")
    result_string = (
        f"With N={N_PARTICIPANTS}, the study had {POWER*100:.0f}% power to detect an overall "
        f"effect size of f ≈ {detectable_f:.3f}, which corresponds to a "
        f"partial eta-squared (η²p) of ≈ {detectable_eta_squared_p:.3f}."
    )
    logging.info(result_string)
    print(result_string) # Also print to console for immediate feedback

    # Save results to a structured table
    results_df = pd.DataFrame({
        'n_participants': [N_PARTICIPANTS],
        'n_groups': [N_GROUPS],
        'alpha': [ALPHA],
        'power': [POWER],
        'detectable_cohen_f': [detectable_f],
        'detectable_eta_squared_p': [detectable_eta_squared_p]
    })
    
    output_path = os.path.join(TABLES_DIR, 'table_power_sensitivity_analysis.csv')
    results_df.to_csv(output_path, index=False)
    logging.info(f"Sensitivity analysis results saved to: {output_path}")


# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    run_sensitivity_analysis()
    logging.info("\n--- power_sensitivity.py Complete ---")