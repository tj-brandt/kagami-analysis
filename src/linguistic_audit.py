# /src/linguistic_audit.py

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

# Define output directories
REPORTS_DIR = '../reports'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'linguistic_audit_log.txt')

# Setup logging and directories
os.makedirs(TABLES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

logging.info("--- Starting linguistic_audit.py ---")


# --- 2. DATA LOADING & PREPARATION ---
def load_and_prepare_data():
    """Loads and merges the main dataset with the derived chat metrics."""
    try:
        df_main = pd.read_csv(INPUT_DATASET_FILE)
        df_metrics = pd.read_csv(CHAT_METRICS_FILE)
        logging.info("Successfully loaded main dataset and derived chat metrics.")
    except FileNotFoundError as e:
        logging.error(f"FATAL: Required data file not found. {e}.")
        exit()

    # Merge to get condition information for each participant's metrics
    df_main['lsm_type_clean'] = df_main['lsm_type_raw'].str.lower().str.strip()
    df_audit = pd.merge(df_metrics, df_main[['participant_id', 'lsm_type_clean']], on='participant_id', how='left')
    return df_audit


# --- 3. ANALYSIS FUNCTIONS ---

def audit_affective_language(df):
    """
    Compares aggregated affective and social language metrics between conditions.
    This corresponds to the "Affect & Authenticity" audit in the thesis.
    """
    logging.info("\n--- AUDIT 1: AFFECTIVE & SOCIAL LANGUAGE ---")
    
    affect_dims = {
        'Positive Tone': 'tone_pos', 
        'Authenticity': 'Authentic', 
        'Social Language': 'Social'
    }
    
    audit_results = []
    for dim_name, col in affect_dims.items():
        static = df[df['lsm_type_clean'] == 'static'][col]
        adaptive = df[df['lsm_type_clean'] == 'adaptive'][col]
        ttest = pg.ttest(static, adaptive, correction=True).assign(Dimension=dim_name)
        audit_results.append(ttest)
        
    summary_table = pd.concat(audit_results, ignore_index=True)
    logging.info("Affective Language Audit Results:")
    print(summary_table.round(4))
    summary_table.to_csv(os.path.join(TABLES_DIR, 'table_audit_affective_language.csv'), index=False)

def audit_structural_language(df):
    """
    Compares aggregated structural language metrics (e.g., verbosity, question-asking)
    between conditions. This corresponds to the "Structural Roles" audit.
    """
    logging.info("\n--- AUDIT 2: STRUCTURAL LANGUAGE ROLES ---")

    structural_dims = {
        'Verbosity (WPS)': 'WPS',
        'Question Asking (%)': 'QMark',
        'Tentativeness': 'differ',
        'Assertiveness (Clout)': 'Clout',
        'Self-Focus (i)': 'i',
        'Other-Focus (you)': 'you',
        'Cognitive Depth (cogproc)': 'cogproc'
    }

    audit_results = []
    for dim_name, col in structural_dims.items():
        if col in df.columns:
            static = df[df['lsm_type_clean'] == 'static'][col]
            adaptive = df[df['lsm_type_clean'] == 'adaptive'][col]
            ttest = pg.ttest(static, adaptive, correction=True).assign(Dimension=dim_name)
            audit_results.append(ttest)
        else:
            logging.warning(f"Column '{col}' for '{dim_name}' not found in chat metrics. Skipping.")
            
    if audit_results:
        summary_table = pd.concat(audit_results, ignore_index=True)
        logging.info("Structural Language Audit Results:")
        print(summary_table.round(4))
        summary_table.to_csv(os.path.join(TABLES_DIR, 'table_audit_structural_language.csv'), index=False)


# --- 4. MAIN EXECUTION BLOCK ---
def main():
    """Main function to orchestrate the linguistic audit pipeline."""
    df_audit = load_and_prepare_data()
    
    audit_affective_language(df_audit)
    audit_structural_language(df_audit)
    
    # NOTE: Non-reproducible analyses (like affective consistency) 
    # have been removed to create
    # a clean, focused, and fully reproducible script.
    
    logging.info("\n--- linguistic_audit.py Complete ---")

if __name__ == "__main__":
    main()