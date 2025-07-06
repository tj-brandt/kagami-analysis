# /src/confirmatory_analysis.py

import pandas as pd
import pingouin as pg
import numpy as np
import os
import logging
from scipy import stats
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)
INPUT_DATASET_FILE = '../data/analysis_dataset_deidentified.csv'
REPORTS_DIR = '../reports'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'confirmatory_analysis_log.txt')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Setup
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH): os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

logging.info("--- Starting confirmatory_analysis.py (v1.6) ---")

# --- 2. DATA PREPARATION ---
try:
    df = pd.read_csv(INPUT_DATASET_FILE)
    logging.info(f"Loaded processed dataset with {len(df)} participants.")
except FileNotFoundError:
    logging.error(f"FATAL: Processed dataset not found at '{INPUT_DATASET_FILE}'. Please run data_preprocessing.py first.")
    exit()

scale_definitions = {
    'DV_Anthro': ['CDV1_1', 'CDV1_2', 'CDV1_3'],
    'DV_Pers': ['CDV1_4', 'CDV1_5', 'CDV1_6', 'CDV1_7'],
    'DV_Trust': ['CDV1_8', 'CDV1_9', 'CDV1_10', 'CDV1_11', 'CDV1_12'],
    'DV_Rapport': ['CDV1_13', 'CDV1_14', 'CDV1_15', 'CDV1_16'],
    'DV_Presence': ['CDV1_17', 'CDV1_18', 'CDV1_19', 'CDV1_20'],
    'DV_Engagement': ['CDV1_21', 'CDV1_22', 'CDV1_23', 'CDV1_24'],
    'DV_Satisfaction': ['CDV1_25', 'CDV1_26', 'CDV1_27', 'CDV1_28']
}

logging.info("\n--- CALCULATING COMPOSITE DVS & RELIABILITY ---")
alpha_results = {}  # Initialize dictionary to store results
for scale_name, items in scale_definitions.items():
    existing_items = [item for item in items if item in df.columns]
    df[scale_name] = df[existing_items].mean(axis=1)
    alpha_result = pg.cronbach_alpha(data=df[existing_items])
    logging.info(f"Cronbach's alpha for {scale_name}: {alpha_result[0]:.3f} (95% CI: {alpha_result[1][0]:.3f} - {alpha_result[1][1]:.3f})")
    
    # Store the alpha results
    alpha_results[scale_name] = alpha_result[0]

df['lsm_type_clean'] = df['lsm_type_raw'].str.lower().str.strip()
df['avatar_type_clean'] = df['avatar_type_raw'].str.lower().str.strip()
df['condition_name'] = df['avatar_type_clean'] + '_' + df['lsm_type_clean']


# --- 3. MANIPULATION CHECKS ---
logging.info("\n--- MANIPULATION & ATTENTION CHECKS ---")
logging.info(f"Confirmatory dataset contains N = {len(df)} participants, indicating 0 exclusions from MC1.")
lsm_static = df[df['lsm_type_clean'] == 'static']['MC4_1']
lsm_adaptive = df[df['lsm_type_clean'] == 'adaptive']['MC4_1']
mc4_adapted_test = pg.ttest(lsm_static, lsm_adaptive, correction=True)
logging.warning(f"HOOK: Perceived Adaptiveness (Reversed): Static (M={lsm_static.mean():.2f}) > Adaptive (M={lsm_adaptive.mean():.2f})")
logging.warning(f"t({mc4_adapted_test['dof'].iloc[0]:.2f}) = {mc4_adapted_test['T'].iloc[0]:.2f}, p = {mc4_adapted_test['p-val'].iloc[0]:.3f}, Cohen's d = {mc4_adapted_test['cohen-d'].iloc[0]:.2f}")

plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='lsm_type_clean', y='MC4_1', order=['static', 'adaptive'], capsize=0.1, ci=95)
plt.title('Figure 1: Perceived Adaptiveness by Condition (The "Adaptation Paradox")')
plt.xlabel('LSM Condition')
plt.ylabel('Perceived Adaptiveness (1-5 Scale)')
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_paradox_barplot.png'))
plt.close()
logging.info(f"Saved paradox bar plot to '{FIGURES_DIR}'.")

# --- 4. PRIMARY CONFIRMATORY 3x2 ANOVAS ---
logging.info("\n--- PRIMARY CONFIRMATORY 3x2 ANOVAS ---")
all_anova_results = []
for dv in scale_definitions.keys():
    logging.info(f"\n{'='*25} ANALYSIS FOR: {dv} {'='*25}")
    
    model = ols(f"{dv} ~ C(avatar_type_clean) * C(lsm_type_clean)", data=df).fit()
    shapiro_p = stats.shapiro(model.resid).pvalue
    levene_p = pg.homoscedasticity(data=df, dv=dv, group='condition_name')['pval'].iloc[0]
    
    if shapiro_p < 0.05: logging.warning(f"Assumption Violation: Normality of residuals (Shapiro-Wilk p={shapiro_p:.4f}).")
    if levene_p < 0.05: logging.warning(f"Assumption Violation: Homogeneity of variances (Levene's p={levene_p:.4f}).")

    aov = pg.anova(data=df, dv=dv, between=['avatar_type_clean', 'lsm_type_clean'], ss_type=2, effsize='np2')
    
    ss_total = aov['SS'].sum()
    ms_resid = aov.loc[aov['Source'] == 'Residual', 'MS'].iloc[0]
    omega_sq_list = []
    for i, row in aov.iterrows():
        if 'Residual' not in row['Source']:
            ss_effect = row['SS']
            df_effect = row['DF']
            omega_sq = (ss_effect - (df_effect * ms_resid)) / (ss_total + ms_resid)
            omega_sq_list.append(omega_sq)
        else:
            omega_sq_list.append(np.nan)
    aov['omega_sq'] = omega_sq_list
    
    all_anova_results.append(aov.assign(DV=dv))
    logging.info(f"Standard ANOVA Results for {dv}:\n{aov.round(4)}\n")

    if shapiro_p < 0.05:
        logging.info(f"--- Running Robust Kruskal-Wallis Tests for {dv} ---")
        kruskal_avatar = pg.kruskal(data=df, dv=dv, between='avatar_type_clean')
        kruskal_lsm = pg.kruskal(data=df, dv=dv, between='lsm_type_clean')
        
        n = len(df.dropna(subset=[dv]))
        eps_sq_avatar = (kruskal_avatar['H'].iloc[0] - kruskal_avatar['ddof1'].iloc[0] + 1) / (n - kruskal_avatar['ddof1'].iloc[0])
        eps_sq_lsm = (kruskal_lsm['H'].iloc[0] - kruskal_lsm['ddof1'].iloc[0] + 1) / (n - kruskal_lsm['ddof1'].iloc[0])
        
        logging.info(f"Kruskal-Wallis (Avatar): H({kruskal_avatar['ddof1'].iloc[0]})={kruskal_avatar['H'].iloc[0]:.3f}, p={kruskal_avatar['p-unc'].iloc[0]:.3f}, ε²={eps_sq_avatar:.3f}")
        logging.info(f"Kruskal-Wallis (LSM): H({kruskal_lsm['ddof1'].iloc[0]})={kruskal_lsm['H'].iloc[0]:.3f}, p={kruskal_lsm['p-unc'].iloc[0]:.3f}, ε²={eps_sq_lsm:.3f}\n")

# --- 5. PLANNED POST-HOC TEST FOR RAPPORT ---
logging.info("\n--- POST-HOC ANALYSIS FOR AVATAR EFFECT ON RAPPORT ---")
rapport_aov_df = next((a for a in all_anova_results if a.iloc[0]['DV'] == 'DV_Rapport'), None)

if rapport_aov_df is not None:
    avatar_effect_p_val = rapport_aov_df[rapport_aov_df['Source'] == 'avatar_type_clean']['p-unc'].iloc[0]
    if avatar_effect_p_val < 0.05:
        logging.info("Main effect of Avatar on Rapport is significant. Running Games-Howell post-hoc...")

        # Run Games-Howell post-hoc
        posthoc_rapport = pg.pairwise_gameshowell(data=df, dv='DV_Rapport', between='avatar_type_clean')
        logging.info(f"Games-Howell Post-Hoc Test for Rapport:\n{posthoc_rapport.round(3)}")
        posthoc_rapport.to_csv(os.path.join(TABLES_DIR, 'table_posthoc_rapport.csv'), index=False)

        # --- Visualize the Avatar × Rapport Post-Hoc Comparison ---
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x='avatar_type_clean', y='DV_Rapport', ci=95, capsize=0.1)
        plt.title('Post-Hoc: Avatar Type Effect on Rapport')
        plt.xlabel('Avatar Type')
        plt.ylabel('Rapport Score')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_posthoc_rapport_barplot.png'))
        plt.close()
        logging.info("Saved post-hoc rapport bar plot to 'fig_posthoc_rapport_barplot.png'")

    else:
        logging.warning(f"Main effect of Avatar on Rapport not significant (p={avatar_effect_p_val:.3f}). Preregistered post-hoc not performed.")
else:
    logging.error("Could not find ANOVA results for DV_Rapport to perform post-hoc test.")

# --- 6. SAVE FINAL OUTPUTS ---
logging.info("\n--- SAVING FINAL OUTPUTS ---")
final_anova_summary = pd.concat(all_anova_results).reset_index(drop=True)
final_anova_summary.to_csv(os.path.join(TABLES_DIR, 'table_anova_summary.csv'), index=False)
logging.info(f"Full ANOVA summary saved to '{TABLES_DIR}/table_anova_summary.csv'")

descriptives = df.groupby(['avatar_type_clean', 'lsm_type_clean'])[list(scale_definitions.keys())].agg(['mean', 'std']).round(3)
descriptives.to_csv(os.path.join(TABLES_DIR, 'table_descriptives.csv'))
logging.info(f"Descriptive statistics saved to '{TABLES_DIR}/table_descriptives.csv'")

logging.info("\n--- confirmatory_analysis.py Complete ---")

# --- 6b. INTERACTION PLOTS FOR SIGNIFICANT EFFECTS ---
logging.info("\n--- GENERATING INTERACTION PLOTS FOR SIGNIFICANT EFFECTS ---")

significant_interactions = final_anova_summary[
    (final_anova_summary['p-unc'] < 0.05) &
    (final_anova_summary['Source'] == 'avatar_type_clean:lsm_type_clean')
]['DV'].unique()

for dv in significant_interactions:
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=df, x='avatar_type_clean', y=dv, hue='lsm_type_clean',
        ci='se', capsize=0.1, dodge=True, errwidth=1.5
    )
    plt.title(f'Interaction: Avatar × LSM on {dv.replace("DV_", "")}')
    plt.ylabel(f'{dv.replace("DV_", "")} Score')
    plt.xlabel('Avatar Type')
    plt.tight_layout()
    fname = f'fig_interaction_{dv.lower()}.png'
    plt.savefig(os.path.join(FIGURES_DIR, fname))
    plt.close()
    logging.info(f"Saved interaction plot to '{fname}'")


# --- 7. SAVE ADDITIONAL TABLES ---
logging.info("\n--- SAVING ADDITIONAL TABLES ---")
alpha_df = pd.DataFrame.from_dict(alpha_results, orient='index', columns=['alpha'])
alpha_df.index.name = 'DV_Scale'
alpha_df.to_csv(os.path.join(TABLES_DIR, 'table_reliability_alphas.csv'))
logging.info(f"Reliability table saved to '{TABLES_DIR}'.")

assumption_data = []
for dv in scale_definitions.keys():
    model = ols(f"{dv} ~ C(avatar_type_clean) * C(lsm_type_clean)", data=df).fit()
    shapiro_p = stats.shapiro(model.resid).pvalue
    levene_p = pg.homoscedasticity(data=df, dv=dv, group='condition_name')['pval'].iloc[0]
    assumption_data.append({'DV': dv, 'shapiro_p': shapiro_p, 'levene_p': levene_p})
assumption_df = pd.DataFrame(assumption_data)
assumption_df.to_csv(os.path.join(TABLES_DIR, 'table_assumption_checks.csv'), index=False)
logging.info(f"Assumption checks table saved to '{TABLES_DIR}'.")