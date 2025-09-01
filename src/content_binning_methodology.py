# /src/content_binning_methodology.py

"""
Content Analysis Binning Methodology (Private Workflow Reconstruction)
====================================================================

This script is a complete reconstruction of the private analysis pipeline used
to generate the conversational content bins for the Kagami project.

!!! NOTE: THIS SCRIPT WILL NOT RUN ON THE PUBLIC REPOSITORY !!!

It requires access to sensitive raw data that has been withheld to protect
participant privacy, including:
- Full, raw user chat logs for each participant.
- Full LIWC-22 dictionary outputs generated from those chat logs.

This file serves as a transparent and executable record of the methodology. The
final de-identified output of this process—a file mapping each participant_id
to a final bin_label—is available in the public repository at:
`data/content_analysis_bins_deidentified.csv`
"""

import pandas as pd
import numpy as np
import logging
import sys
import re
from pathlib import Path

# --- Machine Learning & NLP Imports ---
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan

# --- 1. CONFIGURATION ---

# NOTE: The file paths below point to private data sources and are for
# documentation purposes only. They will not exist in the public repository.
RAW_USER_TEXTS_FILE = Path('path/to/private/user_texts.csv')
RAW_LIWC_RESULTS_FILE = Path('path/to/private/liwc_results.csv')
ANALYSIS_DATASET_FILE = Path('path/to/private/analysis_dataset.csv')
OUTPUT_BINS_FILE = Path('path/to/processed/content_bins.csv')

# --- Binning Methodology Constants ---
BIN_KEYWORDS = {
    'PERSONAL': [r"\b(relationship|partner|husband|wife|therap(y|ist)|anxiety|depress|family|childhood|feelings?)\b"],
    'META_AI': [r"\b(ai|chatgpt|gpt|llm|chatbot|model|prompt|hallucinat)\b"],
    'LOGISTICS': [r"^\s*(hi|hello|hey)\b", r"\b(how are you)\b", r"\b(bye|goodbye|see you)\b"]
}
BIN_SCORING_WEIGHTS = {
    'personal': {'i_rate': 0.9, 'affect_rate': 0.8, 'family_rate': 0.6, 'friend_rate': 0.4, 'keyword': 1.0},
    'superficial': {'social_rate': 0.8, 'cogproc_rate': -0.3, 'topic_prob': 1.2},
    'meta_ai': {'keyword': 1.5},
    'logistics': {'keyword': 1.2}
}
BIN_MIN_SCORE = 0.50
BIN_MIN_MARGIN = 0.20

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


# --- 2. DATA LOADING & FEATURE EXTRACTION ---

def load_and_merge_private_data():
    """Loads all necessary raw data sources."""
    try:
        df_base = pd.read_csv(ANALYSIS_DATASET_FILE)
        df_texts = pd.read_csv(RAW_USER_TEXTS_FILE)
        df_liwc = pd.read_csv(RAW_LIWC_RESULTS_FILE)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required private data file was not found. This script cannot be run. {e}")
        sys.exit(1)
    
    df_liwc.columns = df_liwc.columns.str.lower()
    df_liwc.rename(columns={'source (a)': 'participant_id'}, inplace=True)
    
    df = pd.merge(df_base[['participant_id']], df_texts, on='participant_id', how='left')
    df = pd.merge(df, df_liwc, on='participant_id', how='left')
    df['user_text'] = df['user_text'].fillna('').astype(str)
    return df

def run_bertopic_modeling(docs):
    """Fits a BERTopic model with the exact parameters used in the thesis."""
    logging.info("Fitting BERTopic model on conversational data...")
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=3)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42, min_dist=0.0)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean', cluster_selection_method='eom')
    model = BERTopic(embedding_model=embedder, vectorizer_model=vectorizer, umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=True)
    _, probs = model.fit_transform(docs)
    return model, probs


# --- 3. SCORING AND BIN ASSIGNMENT LOGIC ---

def create_conversational_bins(df: pd.DataFrame, model: BERTopic, probs: np.ndarray) -> pd.DataFrame:
    """Applies the full heuristic ruleset to classify conversations into thematic bins."""
    logging.info("Applying heuristic ruleset to create conversational bins...")
    
    # 1. Add topic probabilities to dataframe
    prob_df = pd.DataFrame(probs, index=df.index, columns=[f"prob_topic_{t}" for t in model.get_topics().keys() if t != -1])
    df = pd.concat([df, prob_df], axis=1)

    # 2. Identify superficial topics based on their keywords
    superficial_terms = {"music", "movie", "game", "food", "book", "tv", "watch", "play"}
    topic_info = model.get_topic_info()
    superficial_ids = [row['Topic'] for _, row in topic_info.iterrows() if row['Topic'] != -1 and any(term in row['Name'] for term in superficial_terms)]
    sup_cols = [f"prob_topic_{tid}" for tid in superficial_ids if f"prob_topic_{tid}" in df.columns]
    df["superficial_topic_prob"] = df[sup_cols].sum(axis=1) if sup_cols else 0.0

    # 3. Calculate z-scored LIWC rates
    wc = df.get("wc", 1).replace(0, 1) # Use word count from LIWC, avoid division by zero
    rate_cols = []
    for col in ['i', 'affect', 'social', 'cogproc', 'family', 'friend']:
        if col in df.columns:
            df[f"{col}_rate"] = 100 * df[col] / wc
            rate_cols.append(f"{col}_rate")
    if rate_cols:
        df[rate_cols] = StandardScaler().fit_transform(df[rate_cols].fillna(0.0))

    # 4. Flag presence of keywords for each bin
    for bin_name, patterns in BIN_KEYWORDS.items():
        kw_col = f"{bin_name.lower().replace('meta_ai', 'meta_ai')}_kw"
        df[kw_col] = df['user_text'].apply(lambda x: int(any(re.search(p, x, re.IGNORECASE) for p in patterns)))
        
    # 5. Calculate composite scores for each bin using predefined weights
    w = BIN_SCORING_WEIGHTS
    df['PERSONAL_SCORE']    = (w['personal']['i_rate'] * df.get('i_rate', 0) + w['personal']['affect_rate'] * df.get('affect_rate', 0) + w['personal']['family_rate'] * df.get('family_rate', 0) + w['personal']['friend_rate'] * df.get('friend_rate', 0) + w['personal']['keyword'] * df.get('personal_kw', 0))
    df['SUPERFICIAL_SCORE'] = (w['superficial']['social_rate'] * df.get('social_rate', 0) + w['superficial']['cogproc_rate'] * df.get('cogproc_rate', 0) + w['superficial']['topic_prob'] * df.get('superficial_topic_prob', 0))
    df['META_AI_SCORE']     = w['meta_ai']['keyword'] * df.get('meta_ai_kw', 0)
    df['LOGISTICS_SCORE']   = w['logistics']['keyword'] * df.get('logistics_kw', 0)

    # 6. Assign final bin label based on highest score and thresholds
    score_cols = ['PERSONAL_SCORE', 'SUPERFICIAL_SCORE', 'META_AI_SCORE', 'LOGISTICS_SCORE']
    scores = df[score_cols].values
    
    best_indices = np.argmax(scores, axis=1)
    best_scores = np.max(scores, axis=1)
    scores[np.arange(len(scores)), best_indices] = -np.inf # Temporarily remove max to find second max
    second_best_scores = np.max(scores, axis=1)
    
    margins = best_scores - second_best_scores
    conditions = (best_scores >= BIN_MIN_SCORE) & (margins >= BIN_MIN_MARGIN)
    
    bin_labels_map = {0: 'PERSONAL', 1: 'SUPERFICIAL', 2: 'META_AI', 3: 'LOGISTICS'}
    bin_labels = np.array([bin_labels_map[i] for i in best_indices])
    
    df['bin_label'] = np.where(conditions, bin_labels, 'UNLABELED')
    return df[['participant_id', 'bin_label']]


# --- 4. MAIN EXECUTION BLOCK ---

def main():
    """Main function to run the entire private binning pipeline."""
    logging.info("--- Starting Content Binning Methodology Script ---")
    
    df_full = load_and_merge_private_data()
    topic_model, topic_probabilities = run_bertopic_modeling(df_full['user_text'].tolist())
    df_binned = create_conversational_bins(df_full, topic_model, topic_probabilities)
    
    df_binned.to_csv(OUTPUT_BINS_FILE, index=False)
    logging.info(f"Successfully generated and saved content bins to: {OUTPUT_BINS_FILE}")
    logging.info("The de-identified version of this output is available in the public repository.")
    
    logging.info("\n--- Content Binning Methodology Script Complete ---")

if __name__ == "__main__":
    print("This script reconstructs the private content binning methodology.")
    print("It requires access to sensitive raw data and will not run on the public repository.")
    # To run this script, you would need the private data files and uncomment the following line:
    # main()