# /src/robustness_checks.py

import pandas as pd
import pingouin as pg
import numpy as np
import os
import logging
import statsmodels.api as sm
from statsmodels.formula.api import ols


# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)

INPUT_DATASET_FILE = '../data/analysis_dataset_deidentified.csv'
CHAT_METRICS_FILE = '../data/chat_metrics_derived.csv'

REPORTS_DIR = '../reports/'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'robustness_checks_log.txt')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Setup
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH): os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

logging.info("--- Starting robustness_checks.py ---")


# --- 2. DATA PREPARATION ---
try:
    df_main = pd.read_csv(INPUT_DATASET_FILE)
    df_metrics = pd.read_csv(CHAT_METRICS_FILE)
    logging.info("Successfully loaded public main and chat metrics datasets.")
except FileNotFoundError as e:
    logging.error(f"FATAL: Required public data file not found. {e}. Please run data_preprocessing.py first.")
    exit()

# Merge the two public data sources
df_main['lsm_type_clean'] = df_main['lsm_type_raw'].str.lower().str.strip()
df_merged = pd.merge(df_main, df_metrics, on='participant_id', how='left')


# --- 3. CHECK 1: BAYES FACTOR FOR OBJECTIVE LSM ---
logging.info("\n--- ROBUSTNESS CHECK 1: BAYES FACTOR FOR OBJECTIVE LSM ---")
if 'objective_lsm' in df_merged.columns:
    static_lsm = df_merged[df_merged['lsm_type_clean'] == 'static']['objective_lsm'].dropna()
    adaptive_lsm = df_merged[df_merged['lsm_type_clean'] == 'adaptive']['objective_lsm'].dropna()
    lsm_ttest = pg.ttest(static_lsm, adaptive_lsm, correction=True)
    t_val, n_static, n_adaptive = lsm_ttest['T'].iloc[0], len(static_lsm), len(adaptive_lsm)
    bf = pg.bayesfactor_ttest(t=t_val, nx=n_static, ny=n_adaptive)

    logging.info(f"T-Test for Objective LSM:\n{lsm_ttest.round(3)}\n")
    logging.info(f"Bayes Factor (BF10) for this test: {bf:.3f}")
    logging.info(f"CONCLUSION: The Bayes Factor (BF10 = {bf:.3f}) provides evidence for the null hypothesis (no difference).")

    # Save output table
    lsm_ttest_bf = lsm_ttest.copy()
    lsm_ttest_bf['BF10'] = bf
    lsm_ttest_bf['BF01'] = 1 / bf
    lsm_ttest_bf.to_csv(os.path.join(TABLES_DIR, 'table_bayes_lsm_ttest.csv'), index=False)
    logging.info(f"Bayesian t-test for objective LSM saved to '{TABLES_DIR}'.")
else:
    logging.warning("Column 'objective_lsm' not found. Skipping Bayes Factor check.")


# --- 4. CHECK 2: SEMANTIC SYNCHRONY (NOT REPRODUCIBLE) ---
logging.warning("\n" + "="*60)
logging.warning("SKIPPING ROBUSTNESS CHECK 2: SEMANTIC SYNCHRONY.")
logging.warning(
    "This analysis requires calculating cosine similarity on sentence embeddings\n"
    "from raw per-turn conversation text. To protect participant privacy,\n"
    "this raw text is not available in the public dataset.\n"
    "The original results and figure ('fig4_semantic_sync_raincloud.png') are\n"
    "preserved in the '/reports' directory for reference."
)
logging.warning("="*60)


# --- 5. CHECK 3: FORMAL MEDIATION ANALYSIS ---
logging.info("\n--- ROBUSTNESS CHECK 3: FORMAL MEDIATION ANALYSIS ---")
logging.info("Testing if structural features (WPS, QMark) mediate the effect of condition on perceived adaptiveness.")

# Prepare the data for mediation from our merged public dataframe
df_mediation = df_merged[['participant_id', 'lsm_type_clean', 'MC4_1', 'WPS', 'QMark']].copy()
df_mediation['is_static'] = (df_mediation['lsm_type_clean'] == 'static').astype(int)
df_mediation.dropna(inplace=True)

if not df_mediation.empty:
    mediation_results = pg.mediation_analysis(data=df_mediation, x='is_static', m=['WPS', 'QMark'], y='MC4_1', seed=42, n_boot=2000)
    logging.info(f"Mediation Analysis Results:\n{mediation_results.round(3)}\n")

    indirect_paths = mediation_results[mediation_results['path'].str.contains('Indirect')]
    is_any_indirect_path_significant = (indirect_paths['pval'] < 0.05).any()

    if is_any_indirect_path_significant:
        logging.info("CONCLUSION: At least one indirect effect is significant, providing some evidence for mediation.")
    else:
        logging.warning("CONCLUSION: No significant indirect effects were found. We cannot conclude that WPS and QMark fully mediate the relationship.")

    # Save the mediation table
    mediation_results.to_csv(os.path.join(TABLES_DIR, 'table_mediation.csv'), index=False)
    logging.info(f"Mediation analysis results saved to '{TABLES_DIR}'.")
else:
    logging.warning("Data for mediation analysis is empty after dropping NaNs. Skipping.")


logging.info("\n--- robustness_checks.py Complete ---")