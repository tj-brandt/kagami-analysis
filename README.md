# Data and Analysis for "Multimodal Anthropomorphism in Companion Chatbots"
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15800990.svg)](https://doi.org/10.5281/zenodo.15800990)

This repository contains the de-identified data and analysis scripts for the Master's thesis project titled  
**"Multimodal Anthropomorphism in Companion Chatbots: Examining Avatar Choice and Adaptive Language Style with Kagami,"**  
conducted at the University of Minnesota (IRB #STUDY00025677).

## Preregistration

This study was preregistered on the Open Science Framework (OSF). The full, public preregistration—including the study design, hypotheses, and analysis plan—can be accessed at the following link:

- **OSF Preregistration:** https://osf.io/we24d

## Project Summary

This study investigated how visual and linguistic anthropomorphism affect perceptions of a companion chatbot. In a 3×2 between-subjects experiment, N = 162 participants recruited from Prolific chatted with the Kagami AI for 10 minutes and completed pre- and post-task surveys. Conditions varied on avatar type (Premade, Generated, None) and linguistic adaptivity (Adaptive vs. Static Language Style Matching).

## How to Reproduce All Analyses

This repository is structured for full reproducibility of all figures and tables.

**Prerequisites:**
*   You have Python 3 installed.
*   You have cloned this repository.

**Steps:**

1.  **Install Dependencies:**
    Navigate to the project's root directory in your terminal and install the required Python packages.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Master Script:**
    First, make the master script executable (you only need to do this once).

    ```bash
    chmod +x run_all.sh
    ```

    Then, execute the script to run all analyses and generate all outputs.

    ```bash
    ./run_all.sh
    ```

This will run all analysis scripts in sequence. When it's finished, all tables and figures in the `/reports` directory will be fully regenerated from the public data.

## Repository Structure
```.
├── data/
│   ├── analysis_dataset_deidentified.csv
│   ├── chat_metrics_derived.csv
│   ├── generated_prompts_coded_deidentified.csv
│   └── content_analysis_bins_deidentified.csv
├── reports/
│   ├── figures/
│   ├── tables/
│   └── (analysis_log.txt files...)
├── src/
│   ├── confirmatory_analysis.py
│   ├── content_analysis.py
│   ├── exploratory_analysis.py
│   ├── linguistic_audit.py
│   ├── robustness_checks.py
│   ├── power_sensitivity.py
│   ├── data_preprocessing.py             # (Documentation Only)
│   ├── content_binning_methodology.py    # (Documentation Only)
│   ├── lsm_trajectory.py                 # (Documentation Only)
│   └── semantic_synchrony.py             # (Documentation Only)
├── materials/
│   └── (IRB Protocol, Consent Forms, etc.)
├── .gitignore
├── LICENSE
└── README.md
```

## Data Availability & Ethical Considerations

The public datasets are located in the `/data/` directory and have been **rigorously de-identified** in accordance with the approved IRB protocol.

-   `analysis_dataset_deidentified.csv`: The main dataset with survey responses, de-identified demographics, and experimental conditions for N=162 participants.
-   `chat_metrics_derived.csv`: Aggregate linguistic features (e.g., LIWC scores, word counts) at the participant level.
-   `generated_prompts_coded_deidentified.csv`: Coded themes of user-generated avatar prompts, with all raw text removed.
-   `content_analysis_bins_deidentified.csv`: Final conversational content bin assignment for each participant.

### Important Note on Withheld Data

To ensure participant confidentiality, all raw text—including full chat logs and open-ended survey responses—and granular per-turn behavioral data have been **withheld** from this public repository.

## Analysis Scripts Overview

### Reproducible Analyses
The following scripts are executed by `run_all.sh` and use only the public data files.

*   `confirmatory_analysis.py`: Runs the preregistered 3x2 ANOVAs and generates **Figure 3**.
*   `content_analysis.py`: Analyzes the conversational content bins and generates **Figures 5, 6, and 7**.
*   `exploratory_analysis.py`: Runs exploratory moderation/mediation analyses and generates **Figures 8 and 9**.
*   `linguistic_audit.py`: Conducts supplementary t-tests comparing aggregate linguistic features between conditions.
*   `robustness_checks.py`: Performs supplementary analyses, including ANCOVA and FDR corrections.
*   `power_sensitivity.py`: Conducts a post-hoc sensitivity power analysis.

### Non-Reproducible Methodologies (Documentation Only)
These scripts contain the complete, original code from the private analysis pipeline. They document the methodology for analyses that required sensitive data but **will not execute**.

*   `data_preprocessing.py`: Documents how the raw, private data was cleaned and processed into the final public datasets.
*   `content_binning_methodology.py`: Documents the BERTopic and LIWC-based pipeline used to classify conversations into thematic bins.
*   `lsm_trajectory.py`: Contains the code that generated **Figure 4**.
*   `semantic_synchrony.py`: Contains the code that generated **Figure 10**.

## Study Materials

The `materials/` folder contains participant-facing documents, including:
- Consent forms
- Post-task surveys
- Experimental stimuli (avatar options, LSM prompt variants)

No identifying information or internal IRB documentation is included.

## License

All files in this repository are shared under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license.  
This applies to all code, data, and study materials contained herein.

**Note:** The Kagami chatbot application used in the experiment is maintained in a separate repository with its own license:  
🔗 https://github.com/tj-brandt/kagami

## Citation

> **Brandt, T.J.** (2025). *Kagami Study – Data Analysis Repository* (Version 1.0-prereg) [Data and Code]. Zenodo. https://doi.org/10.5281/zenodo.15800990