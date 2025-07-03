# Data and Analysis for "Multimodal Anthropomorphism in Companion Chatbots"

This repository contains the de-identified data and analysis scripts for the Master's thesis project titled "Multimodal Anthropomorphism in Companion Chatbots: Examining Avatar Choice and Adaptive Language Style with Kagami," conducted at the University of Minnesota (IRB #STUDY00025677).

## Preregistration

This project was preregistered on the Open Science Framework (OSF). The registration is currently under embargo and will be made public upon submission of the first manuscript.

**OSF Project Page:** https://osf.io/we24d/

**Preregistration (Embargoed):** [Link to be added upon public release]

## Project Summary

This study investigated the effects of user-controlled avatars (Premade, Generated, None) and adaptive language style matching (Adaptive, Static) on user perceptions of a companion chatbot. In a 3x2 between-subjects experiment, N=162 participants recruited from Prolific chatted with the "Kagami" AI for 10 minutes and completed pre- and post-surveys.

## How to Cite

If you use this data or code, please cite the following thesis:
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
│   ├── lsm_trajectory.py
│   ├── power_sensitivity.py
│   └── requirements.txt
├── materials/
│   ├── (IRB Protocol, Consent Forms, etc.)
├── .gitignore
├── LICENSE
└── README.md
```

## Data Availability & Ethical Considerations

The public datasets are located in the `/data` directory and have been rigorously de-identified to protect participant privacy, in accordance with the IRB-approved protocol.

-   `analysis_dataset_deidentified.csv`: Contains all quantitative survey responses and de-identified demographic data.
-   `chat_metrics_derived.csv`: Contains aggregated (mean) linguistic metrics (e.g., LIWC scores, word count) derived from the conversation logs for each participant.
-   `generated_prompts_coded_deidentified.csv`: Contains the coded themes of avatar prompts, with the raw text removed.

**Important Note on Withheld Data:** All raw text data, including full conversation logs, open-ended survey responses, and raw avatar-generation prompts, have been **withheld** from this public dataset. Furthermore, granular per-turn behavioral data (e.g., the sequence of LSM scores) has also been withheld due to the potential for "linguistic fingerprinting." This is a necessary measure to ensure participant confidentiality cannot be breached. The analyses in `lsm_trajectory.py` and parts of `linguistic_audit.py` that rely on this per-turn data are therefore not reproducible with the public data and have been disabled in the scripts.

## How to Reproduce the Analyses

1.  Clone this repository.
2.  Set up a Python virtual environment and install the required packages: `pip install -r src/requirements.txt`
3.  Run the analysis scripts located in the `/src` folder (e.g., `python src/confirmatory_analysis.py`).

*Note: The `data_preprocessing.py` script is part of the original private workflow and is not needed to run the analyses on the provided public datasets.*

## Key Findings & Disclosures

The primary confirmatory analyses are presented in `confirmatory_analysis.py`. A critical finding was a reversed manipulation check for the Language Style Matching (LSM) condition: participants perceived the "Static" bot as significantly more adaptive than the "Adaptive" bot (p = .003). This implementation-perception gap is a central point of discussion in the thesis and complicates the interpretation of H2. Full details are available in the logs and the final manuscript.

## License

This folder contains all participant-facing and design documentation related to the Kagami study, including the final IRB-approved protocol (HRP-580), consent form, survey instruments, and experimental stimuli. All files are shared under CC BY-NC-SA 4.0.

## Related Repositories

This repository contains all data, analysis scripts, and study materials for the paper:  
**"Multimodal Anthropomorphism in Companion Chatbots: Examining Avatar Choice and Adaptive Language Style with Kagami"**

The source code for the Kagami chatbot platform used in the experiment is available here:  
🔗 [tj-brandt/kagami](https://github.com/tj-brandt/kagami)

Please note: the app and this analysis repo use separate licenses appropriate to their content.