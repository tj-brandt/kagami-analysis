# /src/content_analysis.py

import os
import sys
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from matplotlib import colors as mcolors
from math import sqrt

# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)

# --- File Paths ---
INPUT_DATASET_FILE = '../data/analysis_dataset_deidentified.csv'
BINS_FILE = '../data/content_analysis_bins_deidentified.csv'
REPORTS_DIR = '../reports'
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'content_analysis_log.txt')

# --- Setup Directories & Logging ---
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH): os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

# --- Thematic & Style Constants ---
BIN_ORDER = ["PERSONAL", "SUPERFICIAL", "META_AI", "LOGISTICS", "UNLABELED"]
AVATAR_ORDER_VIS = ["Generated", "Premade", "None"]

logging.info("--- Starting content_analysis.py ---")


# --- 2. DATA LOADING & PREPARATION ---
def load_and_prepare_data():
    """Loads and merges datasets, and computes composite DVs."""
    try:
        df_base = pd.read_csv(INPUT_DATASET_FILE)
        df_bins = pd.read_csv(BINS_FILE)
        logging.info(f"Loaded base dataset (N={len(df_base)}) and bins (N={len(df_bins)}).")
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found. {e}")
        sys.exit(1)

    df = pd.merge(df_base, df_bins, on='participant_id', how='left')
    df['bin_label'].fillna('UNLABELED', inplace=True)
    df['avatar_vis'] = df['avatar_type_raw'].str.capitalize()
    
    df['Rapport'] = df[['CDV1_13', 'CDV1_14', 'CDV1_15', 'CDV1_16']].mean(axis=1)
    
    logging.info(f"Successfully merged data. Final N={len(df)}.")
    return df


# --- 3. STATISTICAL ANALYSIS ---
def run_statistical_analyses(df):
    """Performs Chi-square and ANOVA tests related to content bins."""
    logging.info("\n--- Running Statistical Analyses ---")
    
    ct = pd.crosstab(df['avatar_vis'], df['bin_label'])
    ct = ct.reindex(index=AVATAR_ORDER_VIS, columns=BIN_ORDER)
    chi2, p, dof, _ = chi2_contingency(ct)
    logging.info(f"Avatar × Bin Chi-square Test: χ²({dof}) = {chi2:.2f}, p = {p:.4f}")
    ct.to_csv(os.path.join(TABLES_DIR, 'table_contingency_avatar_bin.csv'))
    
    aov = pg.anova(data=df, dv='Rapport', between='bin_label', detailed=True)
    logging.info(f"\nANOVA Results (Rapport by Bin):\n{aov.round(4)}\n")
    aov.to_csv(os.path.join(TABLES_DIR, 'table_anova_rapport_by_bin.csv'))
    return ct


# --- 4. FINAL FIGURE GENERATION ---

def setup_plot_style():
    """Sets a consistent, minimalist style for plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Helvetica', 'figure.dpi': 300})

def wilson_ci(k, n, z=1.96):
    """Calculates the Wilson confidence interval for a proportion."""
    if n == 0: return (0, 0)
    p = k / n; denom = 1 + z**2 / n; center = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def generate_figure_5_prevalence(df, output_dir):
    """Generates final Figure 5: Bin Prevalence."""
    logging.info("Generating final version of Figure 5: Bin Prevalence...")
    setup_plot_style()
    
    total = len(df)
    counts = df['bin_label'].value_counts()
    df_plot = counts.reindex(BIN_ORDER, fill_value=0).reset_index()
    df_plot.columns = ['bin', 'count']
    df_plot['bin_vis'] = df_plot['bin'].replace({"UNLABELED": "Mixed/Other"})
    df_plot['percent'] = (df_plot['count'] / total) * 100
    
    cis = df_plot['count'].apply(lambda k: wilson_ci(k, total))
    lower_ci_bound = cis.apply(lambda t: t[0] * 100)
    upper_ci_bound = cis.apply(lambda t: t[1] * 100)
    y_err = [
        (df_plot['percent'] - lower_ci_bound).values,
        (upper_ci_bound - df_plot['percent']).values
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df_plot['bin_vis'], df_plot['percent'], yerr=y_err, color="#BFDBFE", edgecolor="#2563EB", capsize=5, zorder=2)
    ax.set_title("Conversation Bin Prevalence (95% Wilson CI)", pad=20)
    ax.set_xlabel("Conversation Content Bin")
    ax.set_ylabel("Participants (%)")
    ax.grid(axis='x', visible=False)
    
    output_path = os.path.join(output_dir, "Figure_5_Bin_Prevalence.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved final Figure 5 to: {output_path}")

def generate_figure_6_residuals(contingency_table, output_dir):
    """Generates final Figure 6: Standardized Residuals Heatmap."""
    logging.info("Generating final version of Figure 6: Standardized Residuals...")
    setup_plot_style()
    
    _, _, _, expected = chi2_contingency(contingency_table)
    residuals = (contingency_table - expected) / np.sqrt(expected)
    residuals_vis = residuals.rename(columns={'UNLABELED': 'Mixed/Other'})
    vmax = np.abs(residuals.values).max()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(residuals_vis, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-vmax, vmax=vmax,
                linewidths=.5, cbar_kws={'label': 'Standardized Residual'}, ax=ax)
    ax.set_title("Standardized Residuals: Avatar Type × Conversation Bin", pad=20)
    ax.set_ylabel("Avatar Type")
    ax.set_xlabel("Conversation Content Bin")
    
    output_path = os.path.join(output_dir, "Figure_6_Residuals_Heatmap.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved final Figure 6 to: {output_path}")

def generate_figure_7_rapport_by_bin(df, output_dir):
    """Generates final Figure 7: Rapport by Bin."""
    logging.info("Generating final version of Figure 7: Rapport by Bin...")
    setup_plot_style()

    df_vis = df.copy()
    df_vis['bin_vis'] = df_vis['bin_label'].replace({"UNLABELED": "Mixed/Other"})
    vis_order = [b.replace("UNLABELED", "Mixed/Other") for b in BIN_ORDER]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.pointplot(data=df_vis, x='Rapport', y='bin_vis', order=vis_order,
                  color="#2563EB", errorbar=('ci', 95), join=False, capsize=0.1, ax=ax)
    ax.set_title("Rapport by Conversation Bin", pad=20)
    ax.set_xlabel("Mean Rapport Score (± 95% CI)")
    ax.set_ylabel("Conversation Content Bin")
    ax.grid(axis='y', visible=False)
    
    output_path = os.path.join(output_dir, "Figure_7_Rapport_by_Bin.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved final Figure 7 to: {output_path}")


# --- 5. MAIN EXECUTION BLOCK ---
def main():
    """Main function to orchestrate the content analysis pipeline."""
    df = load_and_prepare_data()
    contingency_table = run_statistical_analyses(df)
    
    logging.info("\n--- Generating Final Figures ---")
    generate_figure_5_prevalence(df, FIG_DIR)
    generate_figure_6_residuals(contingency_table, FIG_DIR)
    generate_figure_7_rapport_by_bin(df, FIG_DIR)

    logging.info("\n--- content_analysis.py Complete ---")

if __name__ == "__main__":
    main()