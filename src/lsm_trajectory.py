# /src/lsm_trajectory.py

"""
LSM Trajectory Modeling (Placeholder Script)
===========================================

This file documents the existence of a longitudinal analysis of turn-by-turn
Language Style Matching (LSM) trajectories conducted as part of the Kagami project.

⚠️ Due to participant privacy concerns, the dataset required to reproduce this
analysis (`turnbyturn.csv`) is NOT publicly released. It contains per-turn LIWC
vectors, message structure, and metadata that could pose re-identification risk.

As such, this file serves as a placeholder and record only.

What Was Done (Summary)
-----------------------
- Computed LSM per conversational turn from derived LIWC categories:
  [auxverb, article, pronoun, ppron, adverb, prep, conj, negate, quantity]
- Modeled the interaction between turn progression and LSM condition (static vs. adaptive)
  using a linear mixed effects model:
    `LSM ~ turn_number * condition + (1 | participant_id)`
- Visualized the average LSM trajectory with 95% CI using seaborn

Where to Find the Results
-------------------------
- See thesis Figure 4 and the Kagami project manuscript for visual and statistical summaries.
- Outputs were generated using a private script (`lsm_trajectory_model_private.py`),
  which is not included in this public repository.

Data Restrictions
-----------------
This analysis relies on a file not safe for public release: `turnbyturn.csv`
containing structured conversational metadata and derived LIWC per-turn vectors.

As a result, the code and raw analysis are excluded from this repository in accordance
with open science best practices and ethical guidelines.

Contact
-------
For questions about this analysis, please contact the lead author or open an issue
on the Kagami GitHub repository.

"""

print("This is a placeholder script for the turn-by-turn LSM trajectory analysis.")
print("Due to privacy risks, this analysis is not publicly reproducible.")