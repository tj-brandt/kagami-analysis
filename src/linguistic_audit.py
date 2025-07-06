# /src/linguistic_audit.py

import pandas as pd
import pingouin as pg
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)

INPUT_DATASET_FILE = '../data/analysis_dataset_deidentified.csv'
CHAT_METRICS_FILE = '../data/chat_metrics_derived.csv' 

REPORTS_DIR = '../reports/'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'linguistic_audit_log.txt')

logging.info("--- Starting linguistic_audit.py ---")


# --- 2. LOAD DATA ---
try:
    df_main = pd.read_csv(INPUT_DATASET_FILE)
    df_metrics = pd.read_csv(CHAT_METRICS_FILE) 
    logging.info(f"Loaded main and derived chat metrics datasets.")
except FileNotFoundError as e:
    logging.error(f"FATAL: Required file not found. {e}. Ensure data_preprocessing.py has been run.")
    exit()

# Merge the main dataframe (for condition info) with the aggregated metrics
df_main['lsm_type_clean'] = df_main['lsm_type_raw'].str.lower().str.strip()
df_audit = pd.merge(df_metrics, df_main[['participant_id', 'lsm_type_clean']], on='participant_id', how='left')


# --- 3. AUDIT 1: AFFECT, AUTHENTICITY, SOCIAL LANGUAGE (AGGREGATED) ---
logging.info("\n--- AUDIT 1: AFFECT & AUTHENTICITY ---")
affect_dims = {'Positive Tone': 'tone_pos', 'Authenticity': 'Authentic', 'Social Language': 'Social'}

audit1_results = []
pvals1 = []

for dim_name, liwc_col in affect_dims.items():
    static = df_audit[df_audit['lsm_type_clean'] == 'static'][liwc_col]
    adaptive = df_audit[df_audit['lsm_type_clean'] == 'adaptive'][liwc_col]
    ttest = pg.ttest(static, adaptive, correction=True)
    ttest['Dimension'] = dim_name
    audit1_results.append(ttest)
    pvals1.append(ttest['p-val'].iloc[0])

sig1, qvals1 = pg.multicomp(pvals1, method='fdr_bh')
for df, q in zip(audit1_results, qvals1):
    df['q'] = q

pd.concat(audit1_results).to_csv(os.path.join(TABLES_DIR, 'table_audit1_affect.csv'), index=False)
logging.info(f"Saved Affect & Authenticity audit table to '{TABLES_DIR}'.")


# --- 4. AUDIT 2: AFFECTIVE CONSISTENCY ---
# This requires per-turn data, which we cannot share publicly. 
# This specific analysis cannot be reproduced with the public data.
logging.warning("\n--- SKIPPING AUDIT 2: AFFECTIVE CONSISTENCY ---")
logging.warning("This analysis requires per-turn data which is not available in the public dataset to protect participant privacy.")
# tone_var = df_audit.groupby(['participant_id', 'lsm_type_clean'])['tone_pos'].std().reset_index()


# --- 5. AUDIT 3: STRUCTURAL LANGUAGE ROLES ---
logging.info("\n--- AUDIT 3: STRUCTURAL ROLES ---")
structural_dims = {
    'Verbosity (WPS)': 'WPS',
    'Question Asking (%)': 'QMark',
    'Tentativeness': 'differ',
    'Assertiveness (Clout)': 'Clout',
    'Self-Focus (i)': 'i',
    'Other-Focus (you)': 'you',
    'Cognitive Depth (cogproc)': 'cogproc'
}

participant_means = df_audit 
audit3_results = []
pvals3 = []

for dim_name, liwc_col in structural_dims.items():
    if liwc_col in participant_means.columns:
        static = participant_means[participant_means['lsm_type_clean'] == 'static'][liwc_col]
        adaptive = participant_means[participant_means['lsm_type_clean'] == 'adaptive'][liwc_col]
        ttest = pg.ttest(static, adaptive, correction=True)
        ttest['Dimension'] = dim_name
        audit3_results.append(ttest)
        pvals3.append(ttest['p-val'].iloc[0])
    else:
        logging.warning(f"Column '{liwc_col}' for dimension '{dim_name}' not found in chat metrics. Skipping.")

sig3, qvals3 = pg.multicomp(pvals3, method='fdr_bh')
for df, q in zip(audit3_results, qvals3):
    df['q'] = q

if audit3_results:
    pd.concat(audit3_results).to_csv(os.path.join(TABLES_DIR, 'table_audit3_structural.csv'), index=False)
    logging.info(f"Saved Structural Roles audit table to '{TABLES_DIR}'.")

# --- 6. FIGURE: CONTRIBUTOR VS. INTERVIEWER SCATTER ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=participant_means, x='WPS', y='QMark',
    hue='lsm_type_clean', style='lsm_type_clean',
    s=100, alpha=0.8
)
plt.title('Figure 2: Conversational Roles (Contributor vs. Interviewer)')
plt.xlabel('Verbosity (Words per Sentence)')
plt.ylabel('Question Asking (%)')
plt.legend(title='Condition', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_role_contrast_scatter.png'))
plt.close()
logging.info("Saved role contrast figure.")

logging.info("\n--- linguistic_audit.py Complete ---")