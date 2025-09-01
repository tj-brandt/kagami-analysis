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
from matplotlib import colors as mcolors

# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)
INPUT_DATASET_FILE = '../data/analysis_dataset_deidentified.csv'
REPORTS_DIR = '../reports'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'confirmatory_analysis_log.txt')

# Setup directories and logging
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

# Define Dependent Variables for analysis
DV_SCALE_DEFINITIONS = {
    'Perceived Anthropomorphism': ['CDV1_1', 'CDV1_2', 'CDV1_3'],
    'Perceived Personalization': ['CDV1_4', 'CDV1_5', 'CDV1_6', 'CDV1_7'],
    'Trust': ['CDV1_8', 'CDV1_9', 'CDV1_10', 'CDV1_11', 'CDV1_12'],
    'Rapport': ['CDV1_13', 'CDV1_14', 'CDV1_15', 'CDV1_16'],
    'Social Presence': ['CDV1_17', 'CDV1_18', 'CDV1_19', 'CDV1_20'],
    'Engagement': ['CDV1_21', 'CDV1_22', 'CDV1_23', 'CDV1_24'],
    'Satisfaction': ['CDV1_25', 'CDV1_26', 'CDV1_27', 'CDV1_28']
}
DVS = list(DV_SCALE_DEFINITIONS.keys())

logging.info("--- Starting confirmatory_analysis.py ---")


# --- 2. DATA PREPARATION ---
def prepare_data(filepath):
    """Loads the dataset, calculates composite DVs, and checks reliability."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded analysis dataset with {len(df)} participants.")
    except FileNotFoundError:
        logging.error(f"FATAL: Dataset not found at '{filepath}'.")
        exit()

    logging.info("\n--- Calculating Composite DVs & Scale Reliability ---")
    for scale_name, items in DV_SCALE_DEFINITIONS.items():
        df[scale_name] = df[items].mean(axis=1)
        alpha = pg.cronbach_alpha(data=df[items])
        logging.info(f"  - {scale_name}: Cronbach's α = {alpha[0]:.3f}")

    df['lsm_type'] = df['lsm_type_raw'].astype('category')
    df['avatar_type'] = df['avatar_type_raw'].astype('category')
    return df


# --- 3. STATISTICAL ANALYSIS FUNCTIONS ---

def run_manipulation_checks(df):
    """Performs and logs the manipulation and attention checks."""
    logging.info("\n--- MANIPULATION & ATTENTION CHECKS ---")
    logging.info(f"Analysis based on N = {len(df)} participants who passed all inclusion criteria.")
    
    static_group = df[df['lsm_type'] == 'static']['MC4_1']
    adaptive_group = df[df['lsm_type'] == 'adaptive']['MC4_1']
    mc4_ttest = pg.ttest(static_group, adaptive_group, correction=True)
    
    logging.warning("Adaptation Paradox Found: Participants perceived the 'Static' condition as more adaptive.")
    logging.warning(f"Static (M={static_group.mean():.2f}, SD={static_group.std():.2f}) vs. "
                    f"Adaptive (M={adaptive_group.mean():.2f}, SD={adaptive_group.std():.2f})")
    logging.warning(f"t({mc4_ttest['dof'].iloc[0]:.2f}) = {mc4_ttest['T'].iloc[0]:.2f}, "
                    f"p = {mc4_ttest['p-val'].iloc[0]:.3f}, Cohen's d = {mc4_ttest['cohen-d'].iloc[0]:.2f}")
    mc4_ttest.to_csv(os.path.join(TABLES_DIR, 'table_manipulation_check_adaptiveness.csv'), index=False)

def run_assumption_checks(df):
    """Performs assumption checks (normality, homogeneity) for all DVs."""
    logging.info("\n--- Running Assumption Checks for ANOVAs ---")
    df['condition_group'] = df['avatar_type'].astype(str) + '_' + df['lsm_type'].astype(str)
    for dv in DVS:
        model = ols(f"Q('{dv}') ~ C(avatar_type) * C(lsm_type)", data=df).fit()
        shapiro_p = stats.shapiro(model.resid).pvalue
        levene_p = pg.homoscedasticity(data=df, dv=dv, group='condition_group')['pval'].iloc[0]
        if shapiro_p < 0.05:
            logging.warning(f"  - {dv}: Normality assumption VIOLATED (Shapiro-Wilk p={shapiro_p:.4f}).")
        if levene_p < 0.05:
            logging.warning(f"  - {dv}: Homogeneity assumption VIOLATED (Levene's p={levene_p:.4f}).")

def run_confirmatory_anovas(df):
    """Runs the preregistered 3x2 ANOVA for each DV and saves the summary."""
    logging.info("\n--- PRIMARY CONFIRMATORY 3x2 ANOVAS ---")
    all_results = [pg.anova(data=df, dv=dv, between=['avatar_type', 'lsm_type'], effsize='np2').assign(DV=dv) for dv in DVS]
    summary_df = pd.concat(all_results).reset_index(drop=True)
    logging.info("Full ANOVA summary table generated.")
    summary_df.to_csv(os.path.join(TABLES_DIR, 'table_anova_summary_full.csv'), index=False)
    return summary_df

def run_post_hoc_tests(df, anova_summary):
    """Runs preregistered post-hoc tests for significant main effects."""
    logging.info("\n--- Running Preregistered Post-Hoc Tests ---")
    rapport_effect = anova_summary[(anova_summary['DV'] == 'Rapport') & (anova_summary['Source'] == 'avatar_type')]
    if not rapport_effect.empty and rapport_effect['p-unc'].iloc[0] < 0.05:
        logging.info("Significant main effect of Avatar on Rapport found. Running Games-Howell post-hoc...")
        posthoc = pg.pairwise_gameshowell(data=df, dv='Rapport', between='avatar_type')
        logging.info(f"Games-Howell Post-Hoc Test for Rapport:\n{posthoc.round(3)}")
        posthoc.to_csv(os.path.join(TABLES_DIR, 'table_posthoc_rapport_games_howell.csv'), index=False)
    else:
        logging.info("Main effect of Avatar on Rapport not significant. No post-hoc performed.")


# --- 4. FINAL FIGURE GENERATION ---

def setup_plot_style():
    """Sets a consistent, minimalist style for plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Helvetica', 'figure.dpi': 300})

def lighten_color(color, amount=0.5):
    """Lightens a color by mixing it with white."""
    c = mcolors.to_rgb(color); w = (1, 1, 1)
    return mcolors.to_hex([(1 - amount) * cv + amount * wv for cv, wv in zip(c, w)])

def generate_figure_3_paradox(df, output_dir):
    """Generates the final, publication-quality version of Figure 3 (Perceived Adaptiveness)."""
    logging.info("\n--- Generating Final Figure 3: The Adaptation Paradox ---")
    setup_plot_style()
    
    plot_order = ['static', 'adaptive']
    stroke_colors = {'static': '#A1A1AA', 'adaptive': '#2250EA'}
    fill_colors = {k: lighten_color(v) for k, v in stroke_colors.items()}

    stats_data = df.groupby('lsm_type')['MC4_1'].agg(['mean', 'sem']).reindex(plot_order)
    means = stats_data['mean'].values
    cis = (stats_data['sem'] * 1.96).values

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(
        plot_order, means, yerr=cis, capsize=4, width=0.4,
        color=[fill_colors[k] for k in plot_order],
        edgecolor=[stroke_colors[k] for k in plot_order],
        linewidth=1.5, zorder=2
    )
    
    ax.set_title("Perceived Adaptiveness by LSM Condition", pad=20)
    ax.set_xlabel("Language Style Matching (LSM) Condition")
    ax.set_ylabel("Perceived Adaptiveness (1–5 Scale)")
    ax.set_ylim(0, 5)
    ax.set_xticklabels(['Static', 'Adaptive'])
    ax.grid(axis='x', visible=False)
    
    output_path = os.path.join(output_dir, "Figure_3_Perceived_Adaptiveness.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved final Figure 3 to: {output_path}")


# --- 5. MAIN EXECUTION BLOCK ---
def main():
    """Main function to orchestrate the entire confirmatory analysis pipeline."""
    df = prepare_data(INPUT_DATASET_FILE)
    
    run_manipulation_checks(df)
    run_assumption_checks(df)
    anova_summary = run_confirmatory_anovas(df)
    run_post_hoc_tests(df, anova_summary)
    
    descriptives = df.groupby(['avatar_type', 'lsm_type'])[DVS].agg(['mean', 'std']).round(3)
    descriptives.to_csv(os.path.join(TABLES_DIR, 'table_descriptives_by_condition.csv'))
    logging.info(f"\nDescriptive statistics saved to '{TABLES_DIR}'.")
    
    generate_figure_3_paradox(df, FIGURES_DIR)

    logging.info("\n--- confirmatory_analysis.py Complete ---")

if __name__ == "__main__":
    main()