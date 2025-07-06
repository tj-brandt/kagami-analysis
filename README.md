# Data and Analysis for "Multimodal Anthropomorphism in Companion Chatbots"
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15800990.svg)](https://doi.org/10.5281/zenodo.15800990)

This repository contains the de-identified data and analysis scripts for the Master's thesis project titled  
**"Multimodal Anthropomorphism in Companion Chatbots: Examining Avatar Choice and Adaptive Language Style with Kagami,"**  
conducted at the University of Minnesota (IRB #STUDY00025677).

## Preregistration

This project was preregistered on the Open Science Framework (OSF). The registration is currently under embargo and will be made public upon submission of the first manuscript.

- **OSF Project Page:** https://osf.io/we24d/  
- **Preregistration (Embargoed):** [Link to be added upon public release]

## Project Summary

This study investigated how visual and linguistic anthropomorphism affect perceptions of a companion chatbot. In a 3×2 between-subjects experiment, N = 162 participants recruited from Prolific chatted with the Kagami AI for 10 minutes and completed pre- and post-task surveys. Conditions varied on avatar type (Premade, Generated, None) and linguistic adaptivity (Adaptive vs. Static Language Style Matching).

## How to Cite

If you use this data or code, please cite the following:

> Brandt, T.J. (2025). *Multimodal Anthropomorphism in Companion Chatbots: Examining Avatar Choice and Adaptive Language Style with Kagami* [Master's thesis, University of Minnesota]. [Link to final thesis document when available]

## Repository Structure
```
.
├── data/
│   ├── analysis_dataset_deidentified.csv
│   ├── chat_metrics_derived.csv
│   └── generated_prompts_coded_deidentified.csv
├── reports/
│   ├── figures/
│   ├── tables/
│   └── (analysis_log.txt files...)
├── src/
│   ├── confirmatory_analysis.py
│   ├── exploratory_analysis.py
│   ├── linguistic_audit.py
│   ├── robustness_checks.py
│   ├── lsm_trajectory.py          # Placeholder only – see README
│   ├── power_sensitivity.py
│   └── requirements.txt
├── materials/
│   ├── (IRB Protocol, Consent Forms, etc.)
├── .gitignore
├── LICENSE
└── README.md
```

## Data Availability & Ethical Considerations

The public datasets are located in the `/data/` directory and have been **rigorously de-identified** in accordance with the approved IRB protocol.

- `analysis_dataset_deidentified.csv`: Survey responses, demographic info, and experimental metadata (deidentified).
- `chat_metrics_derived.csv`: Aggregate linguistic features (e.g., LIWC function word scores, word counts) at the participant level.
- `generated_prompts_coded_deidentified.csv`: Avatar prompt themes, coded and stripped of all textual content.

### ❗ Important Note on Withheld Data

To ensure participant confidentiality:

- All raw text—including full chat logs, open-ended survey responses, and avatar-generation prompts—has been **withheld**.
- Granular **per-turn behavioral data** (e.g., LSM scores for each user-bot exchange) has also been withheld due to the risk of *linguistic fingerprinting* or indirect re-identification.

As a result:
- The LSM trajectory model (used for Figure 4 in the thesis) is **not publicly reproducible**.
- The file `src/lsm_trajectory.py` is provided as a **non-functional placeholder**. It documents the analysis rationale and modeling approach, but will not execute.
- No analysis in this repository uses data from the unreleased `turnbyturn.csv`.

## How to Reproduce the Analyses

1.  Clone this repository.
2.  Set up a Python virtual environment and install the required packages: `pip install -r src/requirements.txt`
3.  Run the analysis scripts located in the `/src` folder (e.g., `python src/confirmatory_analysis.py`).


**Note:** The `data_preprocessing.py` script was part of the original private workflow and is *not needed* to reproduce any results with the public datasets.

## Key Findings & Disclosures

The primary confirmatory results are presented in `confirmatory_analysis.py`. A critical result was a reversed manipulation check: participants perceived the *Static* bot as significantly more adaptive than the *Adaptive* bot (p = .003). This "implementation-perception gap" complicates interpretation of H2 and is a focal point in the discussion. Full details are available in the thesis and accompanying logs.

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