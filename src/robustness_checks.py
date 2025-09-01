# /src/robustness_checks.py

"""
Robustness and Supplementary Analyses
=====================================

This script conducts several supplementary and robustness checks to validate and
further explore the primary findings of the thesis. It uses only the publicly
available, de-identified datasets.

The analyses include:
1.  A Bayes Factor analysis for the non-significant effect on objective LSM.
2.  A formal mediation analysis to test the mechanisms of the Adaptation Paradox.
3.  An ANCOVA to check if the main avatar effect holds when controlling for user engagement.
4.  FDR correction for the multiple ANOVAs run on the conversational content bins.
"""

import pandas as pd
import pingouin as pg
import numpy as np
import os
import logging
# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)

# Define paths to public data files
INPUT_DATASET_FILE = '../data/analysis_dataset_deidentified.csv'
CHAT_METRICS_FILE = '../data/chat_metrics_derived.csv'
CONTENT_BINS_FILE = '../data/content_analysis_bins_deidentified.csv'

# Define output directories
REPORTS_DIR = '../reports'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'robustness_checks_log.txt')

# Setup logging and directories
os.makedirs(TABLES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

logging.info("--- Starting robustness_checks.py ---")


# --- 2. DATA PREPARATION ---
def load_and_prepare_data():
    """Loads and merges all necessary public datasets for the checks."""
    try:
        df_main = pd.read_csv(INPUT_DATASET_FILE)
        df_metrics = pd.read_csv(CHAT_METRICS_FILE)
        df_bins = pd.read_csv(CONTENT_BINS_FILE)
        logging.info("Successfully loaded all required public datasets.")
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required public data file not found: {e}.")
        exit()

    df = pd.merge(df_main, df_metrics, on='participant_id', how='left')
    df = pd.merge(df, df_bins[['participant_id', 'bin_label']], on='participant_id', how='left')
    
    df['Rapport'] = df[['CDV1_13', 'CDV1_14', 'CDV1_15', 'CDV1_16']].mean(axis=1)
    df['Trust'] = df[['CDV1_8', 'CDV1_9', 'CDV1_10', 'CDV1_11', 'CDV1_12']].mean(axis=1)
    df['Loneliness'] = df[['WB1_1', 'WB1_2']].mean(axis=1)
    df['Skepticism'] = df[['AP1_4', 'AP1_5']].mean(axis=1)
    
    return df

# --- 3. ROBUSTNESS CHECK FUNCTIONS ---

def check_bayes_factor_lsm(df: pd.DataFrame):
    """Bayes Factor check for the non-significant effect on objective LSM."""
    logging.info("\n--- ROBUSTNESS CHECK 1: BAYES FACTOR FOR OBJECTIVE LSM ---")
    if 'objective_lsm' not in df.columns:
        logging.warning("Column 'objective_lsm' not found. Skipping Bayes Factor check.")
        return
        
    static = df[df['lsm_type_raw'] == 'static']['objective_lsm'].dropna()
    adaptive = df[df['lsm_type_raw'] == 'adaptive']['objective_lsm'].dropna()
    ttest = pg.ttest(static, adaptive, correction=True)
    bf = pg.bayesfactor_ttest(t=ttest['T'].iloc[0], nx=len(static), ny=len(adaptive))
    logging.info(f"Bayes Factor (BF10) for Objective LSM t-test: {bf:.3f}. Values < 1 support the null.")
    ttest['BF10'] = bf
    ttest.to_csv(os.path.join(TABLES_DIR, 'table_bayes_lsm_ttest.csv'), index=False)

def check_mediation_paradox(df: pd.DataFrame):
    """Formal mediation analysis to test mechanisms of the Adaptation Paradox."""
    logging.info("\n--- ROBUSTNESS CHECK 2: FORMAL MEDIATION ANALYSIS (PARADOX) ---")
    df_med = df[['lsm_type_raw', 'MC4_1', 'WPS', 'QMark']].copy()
    df_med['is_static'] = (df_med['lsm_type_raw'] == 'static').astype(int)
    
    results = pg.mediation_analysis(data=df_med.dropna(), x='is_static', m=['WPS', 'QMark'], y='MC4_1', seed=42)
    logging.info(f"Mediation Analysis Results (Paradox):\n{results.round(3)}\n")
    results.to_csv(os.path.join(TABLES_DIR, 'table_mediation_paradox.csv'), index=False)

def check_dose_covariate(df: pd.DataFrame):
    """ANCOVA: Checks if the Avatar -> Rapport effect holds when controlling for message count."""
    logging.info("\n--- ROBUSTNESS CHECK 3: ANCOVA WITH MESSAGE COUNT COVARIATE ---")
    if 'WC' not in df.columns:
        logging.warning("'WC' (Word Count) column not found. Skipping dose covariate check.")
        return
        
    ancova = pg.ancova(data=df, dv='Rapport', between='avatar_type_raw', covar='WC')
    logging.info("ANCOVA results for Avatar -> Rapport, controlling for Word Count:")
    print(ancova.round(4))
    ancova.to_csv(os.path.join(TABLES_DIR, 'table_ancova_rapport_with_dose.csv'), index=False)

def apply_fdr_to_bin_analyses(df: pd.DataFrame):
    """Runs ANOVAs for multiple DVs by bin and applies FDR correction (matches Thesis Table 4)."""
    logging.info("\n--- ROBUSTNESS CHECK 4: FDR CORRECTION FOR DV-BY-BIN ANALYSES ---")
    
    dvs_to_test = ['Rapport', 'Trust', 'Loneliness', 'Skepticism']
    results = []
    for dv in dvs_to_test:
        aov = pg.anova(data=df.dropna(subset=[dv, 'bin_label']), dv=dv, between='bin_label')
        results.append({'DV': dv, 'p-val': aov['p-unc'].iloc[0], 'F': aov['F'].iloc[0]})
    
    df_results = pd.DataFrame(results)
    
    reject, qvals = pg.multicomp(df_results['p-val'].values, method='fdr_bh')
    
    df_results['FDR-q'] = qvals
    df_results['Significant (FDR)'] = reject
    
    logging.info("ANOVA Results for All DVs by Content Bin (FDR Corrected):")
    print(df_results.round(4))
    df_results.to_csv(os.path.join(TABLES_DIR, 'table4_dv_by_bin_fdr.csv'), index=False)


# --- 4. MAIN EXECUTION BLOCK ---
def main():
    """Orchestrates the loading of data and execution of all robustness checks."""
    df = load_and_prepare_data()
    
    check_bayes_factor_lsm(df)
    check_mediation_paradox(df)
    check_dose_covariate(df)
    apply_fdr_to_bin_analyses(df)
    
    logging.info("\n--- robustness_checks.py Complete ---")

if __name__ == "__main__":
    main()