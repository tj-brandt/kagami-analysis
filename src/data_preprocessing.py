# ==============================================================================
# data_preprocessing.py
#
# IMPORTANT NOTE FOR PUBLIC REPOSITORY USERS:
#
# This script is provided for documentation and transparency purposes only.
# It details the exact process used to generate the final, de-identified public
# datasets from the original raw, private data.
#
# *** THIS SCRIPT CANNOT BE RUN ON THE PUBLIC REPOSITORY ***
#
# It requires access to sensitive raw data files (e.g., from /data/raw/ and
# /data/processed/) that have been withheld to protect participant privacy,
# in accordance with the IRB protocol.
#
# The output of this script (the de-identified public datasets) is already
# included in the /data/ directory of this repository. The other analysis
# scripts (e.g., confirmatory_analysis.py) are designed to run on those
# public files.
# ==============================================================================

import pandas as pd
import json
import os
import logging
import numpy as np

# --- 1. CONFIGURATION ---
np.random.seed(42)

# >>>>> LOCAL/PRIVATE FILE PATHS (These must exist on your local computer) <<<<<
LOG_DIR = '../data/raw/logs/complete/'
QUALTRICS_FILE = '../data/raw/qualtrics.csv'
PROLIFIC_FILE = '../data/raw/prolific.csv'

# >>>>> INTERMEDIATE FILE PATHS (Assumes other scripts create these locally) <<<<<
INTERMEDIATE_DIR = '../data/processed'
TURNBYTURN_FILE = os.path.join(INTERMEDIATE_DIR, 'turnbyturn.csv')
LSM_UB_FILE = os.path.join(INTERMEDIATE_DIR, 'lsm_ub.csv')
GENERATED_PROMPTS_FILE = os.path.join(INTERMEDIATE_DIR, 'generated_prompts_coded.csv')

# >>>>> FINAL PUBLIC OUTPUT PATHS (These will be created by this script) <<<<<
PUBLIC_OUTPUT_DIR = '../data'
PUBLIC_ANALYSIS_FILE = os.path.join(PUBLIC_OUTPUT_DIR, 'analysis_dataset_deidentified.csv')
PUBLIC_CHAT_METRICS_FILE = os.path.join(PUBLIC_OUTPUT_DIR, 'chat_metrics_derived.csv')
PUBLIC_PROMPTS_FILE = os.path.join(PUBLIC_OUTPUT_DIR, 'generated_prompts_coded_deidentified.csv')

# Setup logging and directories
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(PUBLIC_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. LOGS-FIRST PARSING (Build the Source of Truth) ---
logging.info(f"Parsing JSONL logs from: {LOG_DIR}")
log_data = []

log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.jsonl')]

for filename in log_files:
    participant_id = filename.split('_')[1]
    
    with open(os.path.join(LOG_DIR, filename), 'r') as f:
        try:
            first_line = next(f)
            log_entry = json.loads(first_line)
            
            condition_info = log_entry.get('backend_confirmed_condition_obj', {})
            avatar_type = condition_info.get('avatarType', 'none')
            lsm_type = 'adaptive' if condition_info.get('lsm', False) else 'static'

            log_data.append({
                'participant_id': participant_id,
                'avatar_type_raw': avatar_type,
                'lsm_type_raw': lsm_type
            })
        except (StopIteration, json.JSONDecodeError) as e:
            logging.warning(f"Could not parse metadata from log file: {filename}. Skipping. Error: {e}")

logs_df = pd.DataFrame(log_data)
completed_participant_ids = logs_df['participant_id'].unique().tolist()
logging.info(f"Identified {len(completed_participant_ids)} unique completed participants from logs.")


# --- 3. LOAD AND FILTER QUALTRICS DATA ---
logging.info(f"Loading and filtering Qualtrics data...")
qualtrics_df = pd.read_csv(QUALTRICS_FILE, header=0, skiprows=[1, 2])
qualtrics_df.rename(columns={'PROLIFIC_PID': 'participant_id'}, inplace=True)

qualtrics_filtered_df = qualtrics_df[qualtrics_df['participant_id'].isin(completed_participant_ids)].copy()
logging.info(f"Filtered Qualtrics data to match {len(qualtrics_filtered_df)} completed participants.")

# --- 4. SCALE CONVERSION (Likert Scales) ---
logging.info("Converting Likert scale text to numeric values...")

map_agreement_5pt = {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}
map_frequency_5pt = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5}
map_bothered_4pt = {'Not at all': 1, 'Several days': 2, 'More than half the days': 3, 'Nearly every day': 4}
cols_agreement = ['AP1_1', 'AP1_2', 'AP1_3', 'AP1_4', 'AP1_5', 'AP1_6', 'AP1_7', 'AP1_8', 'AP1_9', 'AP1_10', 'AP1_11', 'MC2_1', 'MC2_2', 'MC2_3', 'MC2_4', 'MC3', 'MC4_1', 'MC4_2', 'MC4_3', 'MC5', 'CDV1_1', 'CDV1_2', 'CDV1_3', 'CDV1_4', 'CDV1_5', 'CDV1_6', 'CDV1_7', 'CDV1_8', 'CDV1_9', 'CDV1_10', 'CDV1_11', 'CDV1_12', 'CDV1_13', 'CDV1_14', 'CDV1_15', 'CDV1_16', 'CDV1_17', 'CDV1_18', 'CDV1_19', 'CDV1_20', 'CDV1_21', 'CDV1_22', 'CDV1_23', 'CDV1_24', 'CDV1_25', 'CDV1_26', 'CDV1_27', 'CDV1_28']
cols_frequency = ['WB1_1', 'WB1_2']
cols_bothered = ['WB2_1', 'WB2_2', 'WB2_3', 'WB2_4']

for col in cols_agreement:
    if col in qualtrics_filtered_df.columns:
        temp_map = map_agreement_5pt.copy()
        temp_map['Neither agree nor disagree'] = 3
        qualtrics_filtered_df[col] = qualtrics_filtered_df[col].replace(temp_map)

for col in cols_frequency:
    if col in qualtrics_filtered_df.columns:
        qualtrics_filtered_df[col] = qualtrics_filtered_df[col].replace(map_frequency_5pt)

for col in cols_bothered:
    if col in qualtrics_filtered_df.columns:
        qualtrics_filtered_df[col] = qualtrics_filtered_df[col].replace(map_bothered_4pt)

all_scale_cols = cols_agreement + cols_frequency + cols_bothered
for col in all_scale_cols:
     if col in qualtrics_filtered_df.columns:
        qualtrics_filtered_df[col] = pd.to_numeric(qualtrics_filtered_df[col], errors='coerce')

logging.info("Successfully converted all specified Likert scales to numeric.")


# --- 5. LOAD AND FILTER PROLIFIC DATA ---
logging.info(f"Loading and filtering Prolific data...")
prolific_df = pd.read_csv(PROLIFIC_FILE)
prolific_df.rename(columns={'Participant id': 'participant_id'}, inplace=True)

prolific_filtered_df = prolific_df[prolific_df['participant_id'].isin(completed_participant_ids)].copy()
prolific_cols_to_keep = ['participant_id', 'Age', 'Sex', 'Ethnicity simplified', 'Student status', 'Employment status']
prolific_final_df = prolific_filtered_df[prolific_cols_to_keep]
logging.info(f"Filtered Prolific data to match {len(prolific_final_df)} completed participants.")


# --- 6. MERGE ALL DATA SOURCES ---
logging.info("Merging log, Qualtrics, and Prolific data sources...")
final_df = logs_df.copy()

# Note: 'MC1' is now kept as a string column from Qualtrics.
qualtrics_cols_to_merge = ['participant_id'] + ['MC1'] + all_scale_cols + ['Demo1', 'Demo2', 'Demo3', 'Demo4', 'OE1', 'OE2', 'OE3', 'OE4']
existing_qualtrics_cols = [col for col in qualtrics_cols_to_merge if col in qualtrics_filtered_df.columns]
final_df = pd.merge(final_df, qualtrics_filtered_df[existing_qualtrics_cols], on='participant_id', how='left')
final_df = pd.merge(final_df, prolific_final_df, on='participant_id', how='left')

# --- 7. CREATE PUBLIC, DE-IDENTIFIED ANALYSIS FILE ---
logging.info("\n--- CREATING PUBLIC, DE-IDENTIFIED DATASETS ---")

# Use the `final_df` created in-memory by the previous steps
df_public_main = final_df.copy()

# Define columns to DROP because they contain PII, raw demographics, or free text
columns_to_drop = [
    'OE1', 'OE2', 'OE3', 'OE4',  # Open-ended free text
    'Demo1', # Original raw age from Qualtrics
    'Age',   # Original raw age from Prolific
    'Demo2', # Original gender from Qualtrics
    'Sex',   # Original sex from Prolific
    'Ethnicity simplified', 'Student status', 'Employment status' # Potentially identifying Prolific data
]
df_public_main.drop(columns=[col for col in columns_to_drop if col in df_public_main.columns], inplace=True)

# Perform K-Anonymity on Demographics
# Bin Age (using the 'Demo1' raw age column from the original `final_df`)
age_bins = [18, 24, 34, 44, 54, 64, 100]
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df_public_main['age_binned'] = pd.cut(final_df['Demo1'], bins=age_bins, labels=age_labels, right=False)

# Collapse Gender (using the 'Demo2' column from the original `final_df`)
df_public_main['gender_collapsed'] = final_df['Demo2'].replace({
    'Nonbinary, Genderqueer, Genderfluid': 'Non-binary/Other',
    'Other (please specify)': 'Non-binary/Other',
    'Prefer not to answer': 'Prefer not to answer',
    'Woman, Female, Feminine': 'Woman',
    'Man, Male, Masculine': 'Man'
}).fillna('Unknown') # Handle any potential missing values

# Collapse Race/Ethnicity (using 'Demo3' from `final_df`)
df_public_main['race_ethnicity_collapsed'] = final_df['Demo3'].apply(
    lambda x: 'Two or more races' if isinstance(x, str) and len(x.split(',')) > 1 else x
)
df_public_main.drop(columns=['Demo3'], inplace=True) # Drop the original multi-select column

# Save the de-identified main file
df_public_main.to_csv(PUBLIC_ANALYSIS_FILE, index=False)
logging.info(f"Saved DE-IDENTIFIED main analysis file to: {PUBLIC_ANALYSIS_FILE}")


# --- 8. CREATE PUBLIC, DERIVED CHAT METRICS FILE ---
logging.info("Creating the public, derived chat metrics dataset...")

# Load the intermediate file that has LIWC scores but also raw text
df_turns_raw = pd.read_csv(TURNBYTURN_FILE)
# Drop the sensitive text column
df_turns_safe = df_turns_raw.drop(columns=['Text', 'ColumnID', 'condition'], errors='ignore')

# Aggregate the safe, numeric data per participant
numeric_cols = df_turns_safe.select_dtypes(include=np.number).columns
df_chat_metrics_agg = df_turns_safe.groupby('participant_id')[numeric_cols].mean().reset_index()

# Load and merge objective LSM score
df_lsm_raw = pd.read_csv(LSM_UB_FILE)
df_lsm_agg = df_lsm_raw[df_lsm_raw['Person'] == 'user'].groupby('participant_id')['LSM'].mean().reset_index().rename(columns={'LSM': 'objective_lsm'})

# Merge all derived metrics
df_public_chat_metrics = pd.merge(df_chat_metrics_agg, df_lsm_agg, on='participant_id', how='left')

df_public_chat_metrics.to_csv(PUBLIC_CHAT_METRICS_FILE, index=False)
logging.info(f"Saved SAFE derived chat metrics file to: {PUBLIC_CHAT_METRICS_FILE}")


# --- 9. CREATE PUBLIC, DE-IDENTIFIED PROMPTS FILE ---
logging.info("Creating the public, de-identified coded prompts file...")

df_prompts_raw = pd.read_csv(GENERATED_PROMPTS_FILE)
# Drop the raw avatar prompt text, keeping only the coded themes
df_prompts_safe = df_prompts_raw.drop(columns=['avatar_prompt', 'notes'], errors='ignore')

df_prompts_safe.to_csv(PUBLIC_PROMPTS_FILE, index=False)
logging.info(f"Saved SAFE coded prompts file to: {PUBLIC_PROMPTS_FILE}")

logging.info("--- All public data files have been generated successfully. ---")