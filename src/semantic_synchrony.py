# /src/semantic_synchrony.py

"""
Semantic Synchrony Analysis (Private Workflow Reconstruction for Figure 10)
==========================================================================

!!! IMPORTANT NOTE FOR PUBLIC REPOSITORY USERS !!!

This script is a complete reconstruction of the private analysis used to
generate Figure 10 in the thesis ("Semantic Synchrony by Language Style Matching (LSM) Condition").

*** THIS SCRIPT CANNOT BE RUN ON THE PUBLIC REPOSITORY ***

It requires access to a private dataset containing the full, raw, turn-by-turn
conversational text for each participant. This data has been withheld to
protect participant privacy due to the risk of re-identification.

This file serves as a transparent and executable record of the exact methodology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging
import pingouin as pg
from sentence_transformers import SentenceTransformer, util
from matplotlib import colors as mcolors
from pathlib import Path

# --- 1. CONFIGURATION ---
np.random.seed(42)

# NOTE: These paths point to private data sources and are for documentation.
ANALYSIS_DATASET_FILE = Path('../data/processed/analysis_dataset.csv')
LIWC_TURN_BY_TURN_FILE = Path('../data/external/turnbyturn_liwc.csv') # This private file contains the raw text

# --- Output Paths ---
REPORTS_DIR = Path('../reports')
TABLES_DIR = REPORTS_DIR / 'tables'
FIGURES_DIR = REPORTS_DIR / 'figures'
LOG_FILE_PATH = REPORTS_DIR / 'semantic_synchrony_log.txt'

# --- Setup Logging and Directories ---
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
if LOG_FILE_PATH.exists(): LOG_FILE_PATH.unlink()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()])

# --- 2. AESTHETIC & STYLE CONFIGURATION ---

def setup_plot_style():
    """Sets a consistent, minimalist style for plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Helvetica', 'figure.dpi': 300})

def lighten_color(color, amount=0.7):
    """Lightens a color by mixing it with white."""
    c = mcolors.to_rgb(color); w = (1, 1, 1)
    return mcolors.to_hex([(1 - amount) * cv + amount * wv for cv, wv in zip(c, w)])

def darken_color(color, amount=0.2):
    """Darkens a color by mixing it with black."""
    c = mcolors.to_rgb(color)
    return mcolors.to_hex([(1 - amount) * cv for cv in c])

# --- 3. DATA PROCESSING & ANALYSIS ---

def calculate_and_analyze_synchrony(meta_path, turns_path):
    """
    Loads private turn-by-turn text data, calculates semantic synchrony for each
    participant, merges with metadata, and performs the statistical test.
    """
    logging.info("--- Calculating Semantic Synchrony from Private Data ---")
    try:
        df_meta = pd.read_csv(meta_path)
        df_turns = pd.read_csv(turns_path)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required private data file was not found. This script cannot be run. {e}")
        sys.exit(1)

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    similarities = []
    for pid, group in df_turns.groupby('participant_id'):
        user_texts = group[group['ColumnID'] == 'user_text']['Text'].tolist()
        bot_texts = group[group['ColumnID'] == 'bot_text']['Text'].tolist()
        min_len = min(len(user_texts), len(bot_texts))
        
        if min_len > 0:
            user_embeds = model.encode(user_texts[:min_len], show_progress_bar=False)
            bot_embeds = model.encode(bot_texts[:min_len], show_progress_bar=False)
            avg_sim = util.cos_sim(user_embeds, bot_embeds).diag().mean().item()
            similarities.append({'participant_id': pid, 'semantic_similarity': avg_sim})

    df_sim = pd.DataFrame(similarities)
    df_sync = pd.merge(df_meta, df_sim, on='participant_id', how='left').dropna(subset=['semantic_similarity'])
    
    # --- Statistical Test ---
    logging.info("\n--- T-Test for Semantic Synchrony (Static vs. Adaptive) ---")
    static_sim = df_sync[df_sync['lsm_type_raw'] == 'static']['semantic_similarity']
    adaptive_sim = df_sync[df_sync['lsm_type_raw'] == 'adaptive']['semantic_similarity']
    ttest = pg.ttest(static_sim, adaptive_sim, correction=True)
    logging.info(f"T-Test Results:\n{ttest.round(4)}")
    ttest.to_csv(TABLES_DIR / "table_semantic_synchrony_ttest.csv", index=False)
    
    return df_sync

# --- 4. PLOTTING FUNCTION ---

def create_figure_10_raincloud(df: pd.DataFrame, output_dir: Path):
    """Creates and saves the final, publication-quality raincloud plot for Figure 10."""
    logging.info("\n--- Generating Final Version of Figure 10: Semantic Synchrony Raincloud ---")
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    order = ['adaptive', 'static']

    STROKE = {'adaptive': '#2343AF', 'static': '#B1C0DD'}
    FILL = {k: lighten_color(v) for k, v in STROKE.items()}
    DOTS = {k: darken_color(v) for k, v in STROKE.items()}
    
    sns.violinplot(data=df, x="lsm_type_raw", y="semantic_similarity", order=order, palette=FILL, inner=None, cut=0, zorder=1, ax=ax, hue="lsm_type_raw", legend=False)
    for i, cat in enumerate(order):
        ax.collections[i].set_edgecolor(STROKE[cat])
        ax.collections[i].set_linewidth(1.5)

    sns.boxplot(data=df, x="lsm_type_raw", y="semantic_similarity", order=order, width=0.2, zorder=2, ax=ax, hue="lsm_type_raw", legend=False,
                boxprops={'facecolor': 'white', 'linewidth': 1.5}, medianprops={'color': 'black', 'linewidth': 2},
                whiskerprops={'linewidth': 1.5}, capprops={'linewidth': 1.5}, flierprops={'marker': ''})
    for i, cat in enumerate(order):
        ax.patches[i].set_edgecolor(STROKE[cat])
        for j in range(i * 6, i * 6 + 5):
            ax.lines[j].set_color(STROKE[cat])

    sns.stripplot(data=df, x="lsm_type_raw", y="semantic_similarity", order=order, hue="lsm_type_raw", palette=DOTS,
                  legend=False, size=3.5, jitter=0.15, alpha=0.8, zorder=3, ax=ax)

    ax.set_title("Semantic Synchrony by LSM Condition", pad=20)
    ax.set_xlabel("Language Style Matching (LSM) Condition")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_xticklabels([label.get_text().capitalize() for label in ax.get_xticklabels()])
    ax.grid(axis='x', visible=False)
    
    output_path = output_dir / "Figure_10_Semantic_Synchrony.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Methodology for Figure 10 is documented. Final plot would be saved to:\n  -> {output_path}")

# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("This script contains the exact private code used to generate Figure 10.")
    print("Due to privacy risks, it cannot be run using the public data.")
    print("To run, you would need the private datasets and uncomment the lines below.")
    #
    # try:
    #     synchrony_df = calculate_and_analyze_synchrony(ANALYSIS_DATASET_FILE, LIWC_TURN_BY_TURN_FILE)
    #     create_figure_10_raincloud(synchrony_df, FIGURES_DIR)
    # except Exception as e:
    #     print(f"\nExecution failed as expected on a public repo: {e}")