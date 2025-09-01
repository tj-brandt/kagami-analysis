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
from matplotlib import colors as mcolors

# --- 1. CONFIGURATION & SETUP ---
np.random.seed(42)

# Define paths to public data files
INPUT_DATASET_FILE = '../data/analysis_dataset_deidentified.csv'
CODED_PROMPTS_FILE = '../data/generated_prompts_coded_deidentified.csv'

# Define output directories
REPORTS_DIR = '../reports'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
LOG_FILE_PATH = os.path.join(REPORTS_DIR, 'exploratory_analysis_log.txt')

# Setup logging and directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
if os.path.exists(LOG_FILE_PATH): os.remove(LOG_FILE_PATH)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

logging.info("--- Starting exploratory_analysis.py ---")


# --- 2. DATA PREPARATION ---
def load_and_prepare_data():
    """Loads and prepares all data needed for exploratory analyses."""
    try:
        df = pd.read_csv(INPUT_DATASET_FILE)
        df_prompts = pd.read_csv(CODED_PROMPTS_FILE)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found. {e}.")
        exit()

    # Merge coded prompt themes
    theme_map = {'Idealized Self': 'Self', 'Helper/Companion': 'Self', 'Animal': 'Other', 'Fantasy/Other': 'Other', 'Other': 'Other', 'Abstract Traits': 'Other'}
    df_prompts['theme_collapsed'] = df_prompts['theme_1'].replace(theme_map)
    df = pd.merge(df, df_prompts[['participant_id', 'theme_collapsed']], on='participant_id', how='left')

    # Calculate composite DVs
    df['Rapport'] = df[['CDV1_13', 'CDV1_14', 'CDV1_15', 'CDV1_16']].mean(axis=1)
    df['Loneliness'] = df[['WB1_1', 'WB1_2']].mean(axis=1)
    
    logging.info(f"Successfully loaded and prepared data for {len(df)} participants.")
    return df


# --- 3. PLOTTING & ANALYSIS FUNCTIONS ---

def setup_plot_style():
    """Sets a consistent, minimalist style for plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Helvetica', 'figure.dpi': 300})

def analyze_and_plot_avatar_theme(df, tables_dir, figures_dir):
    """Performs t-test and generates final Figure 8 (Rapport by Avatar Theme)."""
    logging.info("\n--- ANALYSIS 1: EFFECT OF AVATAR THEME ON RAPPORT (THESIS FIG 8) ---")
    df_generated = df[df['avatar_type_raw'] == 'generated'].dropna(subset=['theme_collapsed', 'Rapport'])
    
    # Statistical Test
    group_self = df_generated[df_generated['theme_collapsed'] == 'Self']['Rapport']
    group_other = df_generated[df_generated['theme_collapsed'] == 'Other']['Rapport']
    ttest = pg.ttest(group_self, group_other, correction=True)
    logging.info(f"T-test (Self vs. Other on Rapport):\n{ttest.round(3)}\n")
    ttest.to_csv(os.path.join(tables_dir, 'table_exploratory_theme_ttest.csv'), index=False)
    
    # Figure Generation
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df_generated, x='theme_collapsed', y='Rapport', order=['Other', 'Self'], width=0.4,
                boxprops={'facecolor': '#FFFFFF', 'edgecolor': 'black'}, medianprops={'color': 'black'},
                whiskerprops={'color': 'black'}, capprops={'color': 'black'}, fliersize=0, ax=ax)
    sns.stripplot(data=df_generated, x='theme_collapsed', y='Rapport', order=['Other', 'Self'], color="#4B5563",
                  alpha=0.7, jitter=0.15, ax=ax)
    ax.set_title("Rapport Scores by Generated Avatar Theme", pad=20)
    ax.set_xlabel("Generated Avatar Theme")
    ax.set_ylabel("Rapport Score (1–5 Scale)")
    ax.grid(axis='x', visible=False)
    ax.set_ylim(2.2, 5.2)
    output_path = os.path.join(figures_dir, "Figure_8_Rapport_by_Theme.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved final Figure 8 to: {output_path}")

def analyze_and_plot_loneliness_moderation(df, tables_dir, figures_dir):
    """Performs moderation analysis and generates final Figure 9."""
    logging.info("\n--- ANALYSIS 2: LONELINESS AS A MODERATOR (THESIS FIG 9) ---")
    df_mod = df[df['avatar_type_raw'].isin(['generated', 'premade'])].copy()
    df_mod.dropna(subset=['Loneliness', 'Rapport'], inplace=True)
    df_mod['is_generated'] = (df_mod['avatar_type_raw'] == 'generated').astype(int)
    df_mod['loneliness_c'] = df_mod['Loneliness'] - df_mod['Loneliness'].mean()
    
    # Statistical Model
    model = ols('Rapport ~ is_generated * loneliness_c', data=df_mod).fit()
    logging.info(f"Moderation Model Summary:\n{model.summary()}\n")
    summary_table = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    summary_table.to_csv(os.path.join(tables_dir, 'table_moderation_loneliness.csv'))
    
    # Figure Generation
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    palette = {'Generated': '#3B82F6', 'Premade': '#374151'}
    df_mod['Avatar Type'] = df_mod['avatar_type_raw'].str.capitalize()
    
    sns.regplot(data=df_mod[df_mod['Avatar Type'] == 'Premade'], x='loneliness_c', y='Rapport', color=palette['Premade'], ax=ax,
                scatter_kws={'alpha': 0.6, 'edgecolor': 'w'}, line_kws={'linewidth': 2.5}, label='Premade')
    sns.regplot(data=df_mod[df_mod['Avatar Type'] == 'Generated'], x='loneliness_c', y='Rapport', color=palette['Generated'], ax=ax,
                scatter_kws={'alpha': 0.6, 'edgecolor': 'w'}, line_kws={'linewidth': 2.5}, label='Generated')

    ax.set_title("Interaction of Avatar Type and Loneliness on Rapport", pad=20)
    ax.set_xlabel("Loneliness (Centered)")
    ax.set_ylabel("Rapport Score (1-5 Scale)")
    ax.legend(title="Avatar Type", frameon=False)
    ax.set_ylim(1.0, 5.5)
    ax.grid(axis='x', visible=False)
    
    output_path = os.path.join(figures_dir, "Figure_9_Loneliness_Moderation.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved final Figure 9 to: {output_path}")


# --- 4. MAIN EXECUTION BLOCK ---
def main():
    """Main function to orchestrate the exploratory analysis pipeline."""
    df = load_and_prepare_data()
    
    analyze_and_plot_avatar_theme(df, TABLES_DIR, FIGURES_DIR)
    analyze_and_plot_loneliness_moderation(df, TABLES_DIR, FIGURES_DIR)
    
    logging.info("\n--- exploratory_analysis.py Complete ---")

if __name__ == "__main__":
    main()