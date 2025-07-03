# /src/power_sensitivity.py

from statsmodels.stats.power import FTestAnovaPower
import pandas as pd
import os
import numpy as np
np.random.seed(42)
REPORTS_DIR = '../reports/'
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

power_analysis = FTestAnovaPower()
f_detectable = power_analysis.solve_power(k_groups=6, nobs=162, alpha=0.05, power=0.80)
eta_sq_detectable = f_detectable**2 / (1 + f_detectable**2)

print(f"Sensitivity Analysis: With N=162, we had 80% power to detect a medium effect of η² ≈ {eta_sq_detectable:.3f} (f ≈ {f_detectable:.3f}).")
pd.DataFrame({
    'eta_sq_detectable_at_80_power': [eta_sq_detectable]
}).to_csv(os.path.join(TABLES_DIR, 'table_power_sensitivity.csv'), index=False)