# /src/lsm_trajectory.py

"""
LSM Trajectory Modeling (Private Workflow Reconstruction for Figure 4)
======================================================================

!!! IMPORTANT NOTE FOR PUBLIC REPOSITORY USERS !!!

This script is a complete reconstruction of the private analysis used to
generate Figure 4 in the thesis ("Objective LSM Trajectory by Condition").

*** THIS SCRIPT CANNOT BE RUN ON THE PUBLIC REPOSITORY ***

It requires access to the private `turnbyturn_liwc.csv` file, which contains
per-turn LIWC data that has been withheld to protect participant privacy due
to the risk of linguistic fingerprinting and re-identification.

This file serves as a transparent and executable record of the exact methodology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.interpolate import make_interp_spline
from pathlib import Path

# --- 1. CONFIGURATION ---

# NOTE: These paths point to private data sources and are for documentation.
ANALYSIS_DATASET_FILE = Path('../data/processed/analysis_dataset.csv')
LIWC_TURN_BY_TURN_FILE = Path('../data/external/turnbyturn_liwc.csv')
FIGURES_DIR = Path('../reports/figures')

# --- Analysis Constants ---
LIWC_LSM_CATEGORIES = [
    "auxverb", "article", "pronoun", "ppron", "adverb",
    "prep", "conj", "negate", "quantity"
]

# --- 2. AESTHETIC & STYLE CONFIGURATION ---

def setup_plot_style():
    """Sets a consistent, minimalist style for plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': 'Helvetica',
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
        'figure.dpi': 300
    })

# --- 3. DATA LOADING & PREPARATION ---

def load_and_prepare_lsm_data(meta_path, turns_path):
    """Loads private turn-by-turn data and calculates LSM scores."""
    try:
        df_meta = pd.read_csv(meta_path)
        df_turns = pd.read_csv(turns_path)
    except FileNotFoundError as e:
        print(f"FATAL: A required private data file was not found. This script cannot be run. {e}")
        sys.exit(1)

    condition_map = df_meta.set_index('participant_id')['lsm_type_raw'].to_dict()
    
    lsm_results = []
    for pid, group in df_turns.groupby('participant_id'):
        user = group[group['ColumnID'] == 'user_text'].set_index('turn_number')
        bot = group[group['ColumnID'] == 'bot_text'].set_index('turn_number')
        aligned_turns = user.index.intersection(bot.index)

        for turn in aligned_turns:
            diffs = [1 - (abs(user.loc[turn, c] - bot.loc[turn, c]) / 
                         (user.loc[turn, c] + bot.loc[turn, c] + 1e-6))
                     for c in LIWC_LSM_CATEGORIES]
            lsm_results.append({'participant_id': pid, 'turn_number': turn, 'lsm_score': np.nanmean(diffs)})

    df_lsm = pd.DataFrame(lsm_results)
    df_lsm['condition'] = df_lsm['participant_id'].map(condition_map)
    df_lsm = df_lsm.dropna(subset=['lsm_score', 'condition'])
    return df_lsm

# --- 4. PLOTTING FUNCTION ---

def create_lsm_trajectory_chart(df: pd.DataFrame, output_dir: Path):
    """Creates and saves the final line chart for LSM trajectory with smooth error bands."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {'adaptive': "#003CFF", 'static': '#374151'}

    for condition, style in zip(['static', 'adaptive'], ['--', '-']):
        agg_df = df[df['condition'] == condition].groupby('turn_number')['lsm_score'].agg(['mean', 'count', 'std']).reset_index()
        sem = agg_df['std'].fillna(0) / np.sqrt(agg_df['count'])
        agg_df['ci_low'] = agg_df['mean'] - 1.96 * sem
        agg_df['ci_high'] = agg_df['mean'] + 1.96 * sem
        agg_df.dropna(subset=['ci_low', 'ci_high'], inplace=True)
        agg_df = agg_df.sort_values(by='turn_number')

        x, y_mean, y_low, y_high = agg_df['turn_number'], agg_df['mean'], agg_df['ci_low'], agg_df['ci_high']
        color = palette[condition]
        
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline_mean = make_interp_spline(x, y_mean, k=3)(x_smooth)
            spline_low = make_interp_spline(x, y_low, k=3)(x_smooth)
            spline_high = make_interp_spline(x, y_high, k=3)(x_smooth)
            
            ax.plot(x_smooth, spline_mean, color=color, linestyle=style, label=condition.capitalize(), zorder=3)
            ax.fill_between(x_smooth, spline_low, spline_high, color=color, alpha=0.15, linewidth=0, zorder=2)
        else:
            ax.plot(x, y_mean, color=color, linestyle=style, label=condition.capitalize())

    ax.set_title("Objective Language Style Matching (LSM) Trajectory", pad=20)
    ax.set_xlabel("Conversational Turn Number")
    ax.set_ylabel("Mean LSM Score (0–1)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0)
    ax.legend(title="Condition", frameon=False, loc='upper right')
    ax.grid(axis='x', visible=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Figure_4_LSM_Trajectory.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"\nMethodology for Figure 4 is documented. Final plot would be saved to:\n  -> {output_path}")

# --- 5. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    print("This script contains the exact private code used to generate Figure 4.")
    print("Due to privacy risks, it cannot be run using the public data.")
    print("To run, you would need the private datasets and uncomment the lines below.")
    #
    # try:
    #     lsm_data = load_and_prepare_lsm_data(ANALYSIS_DATASET_FILE, LIWC_TURN_BY_TURN_FILE)
    #     create_lsm_trajectory_chart(lsm_data, FIGURES_DIR)
    # except Exception as e:
    #     print(f"\nExecution failed as expected on a public repo: {e}")