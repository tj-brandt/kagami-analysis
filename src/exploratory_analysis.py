# /src/exploratory_analysis.py

import pandas as pd
import pingouin as pg
import numpy as np
import os
import logging
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)
INPUT_DATASET_FILE = '../data/analysis_dataset_deidentified.csv'
CODED_PROMPTS_FILE = '../data/generated_prompts_coded_deidentified.csv'
REPORTS_DIR = '../reports/'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'exploratory_analysis_log.txt')

# Setup
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH): os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

logging.info("--- Starting exploratory_analysis.py ---")

# --- 2. DATA PREPARATION ---
try:
    df = pd.read_csv(INPUT_DATASET_FILE)
    logging.info(f"Loaded main dataset with {len(df)} participants.")
    df_prompts = pd.read_csv(CODED_PROMPTS_FILE)
    logging.info(f"Loaded coded avatar prompts for {len(df_prompts)} participants.")
except FileNotFoundError as e:
    logging.error(f"FATAL: Required file not found. {e}.")
    exit()

# Normalize IDs for a clean merge
df['participant_id'] = df['participant_id'].astype(str).str.strip()
df_prompts['participant_id'] = df_prompts['participant_id'].astype(str).str.strip()

theme_map = {
    'Idealized Self': 'Self',
    'Helper/Companion': 'Self',
    'Animal': 'Other',
    'Fantasy/Other': 'Other',
    'Other': 'Other',
    'Abstract Traits': 'Other'
}
df_prompts['theme_collapsed'] = df_prompts['theme_1'].replace(theme_map)

# Merge the coded themes into the main dataframe
df = pd.merge(df, df_prompts[['participant_id', 'theme_collapsed']], on='participant_id', how='left')

# Create composite DV_Rapport score
df['DV_Rapport'] = df[['CDV1_13', 'CDV1_14', 'CDV1_15', 'CDV1_16']].mean(axis=1)


# --- 3. ANALYSIS 1: AVATAR THEME EFFECT ON RAPPORT ---
logging.info("\n--- EXPLORATORY ANALYSIS: EFFECT OF AVATAR THEME ON RAPPORT ---")
df_generated = df[df['avatar_type_raw'] == 'generated'].dropna(subset=['theme_collapsed', 'DV_Rapport'])
logging.info(f"Analyzing {len(df_generated)} participants from the 'generated' condition with valid theme codes.")

model_theme = ols('DV_Rapport ~ C(theme_collapsed)', data=df_generated).fit()
anova_table = sm.stats.anova_lm(model_theme, typ=2)

logging.info("--- Collapsed Theme OLS Summary ---")
logging.info(f"\n{model_theme.summary()}")
logging.info("\n--- Collapsed Theme ANOVA Table ---")
logging.info(f"\n{anova_table}\n")

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_generated, x='theme_collapsed', y='DV_Rapport')
plt.title('Rapport Scores by Collapsed Avatar Theme')
plt.savefig(os.path.join(FIGURES_DIR, 'fig_exploratory_theme_rapport.png'))
plt.close()
logging.info(f"Saved avatar theme boxplot to '{FIGURES_DIR}'.")


# --- 4. ANALYSIS 2: LONELINESS MODERATION ANALYSIS ---
logging.info("\n--- EXPLORATORY ANALYSIS: LONELINESS AS A MODERATOR ---")
# We will compare 'generated' vs. 'premade'
df_mod = df[df['avatar_type_raw'].isin(['generated', 'premade'])].copy()
df_mod['is_generated'] = (df_mod['avatar_type_raw'] == 'generated').astype(int)

# Create and center the loneliness moderator
df_mod['loneliness'] = df_mod[['WB1_1', 'WB1_2']].mean(axis=1)
# Drop rows where loneliness couldn't be computed
df_mod.dropna(subset=['loneliness'], inplace=True)
df_mod['loneliness_c'] = df_mod['loneliness'] - df_mod['loneliness'].mean()

logging.info("Running OLS Regression for Moderation: Rapport ~ Generated * Loneliness...")
moderation_model = ols('DV_Rapport ~ is_generated * loneliness_c', data=df_mod).fit()
logging.info(f"Moderation Model Summary:\n{moderation_model.summary()}\n")

# Check for significance
interaction_p_val = moderation_model.pvalues.get('is_generated:loneliness_c', 1.0)
logging.info(f"Interaction p-value: {interaction_p_val:.4f}")
if interaction_p_val < 0.05:
    logging.info("CONCLUSION: The interaction is significant. Loneliness moderates the effect of avatar type on rapport.")
else:
    logging.warning("CONCLUSION: The interaction is NOT significant. There is no evidence that loneliness moderates the effect.")

# Visualization
sns.lmplot(data=df_mod, x='loneliness_c', y='DV_Rapport', hue='is_generated', height=6, aspect=1.2)
plt.title('Interaction of Avatar Type and Loneliness on Rapport')
plt.xlabel('Loneliness (Centered)')
plt.ylabel('Rapport Score')
plt.savefig(os.path.join(FIGURES_DIR, 'fig_exploratory_loneliness_moderation.png'))
plt.close()
logging.info(f"Saved loneliness moderation plot to '{FIGURES_DIR}'.")

logging.info("\n--- exploratory_analysis.py Complete ---")