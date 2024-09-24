import numpy as np
import pandas as pd


glob = pd.read_excel("../data/incidence_dr_globocan.xlsx", sheet_name="Globocan")
glob = glob.iloc[3:16, :]

### Calibration Targets
# Target 1: SEER Incidence
# dr_inc = pd.read_excel("../data/incidence_crude.xlsx", sheet_name="1975-1990 Adj")
# dr_inc = dr_inc[dr_inc["Age"] >= 20].reset_index()  # single ages, 20-84 (65 ages)
# dr_inc = dr_inc[dr_inc["Age"] <= 84].reset_index()  # starting age 20, 65 ages
dr_inc = pd.read_excel(
    "../data/incidence_dr_globocan.xlsx", sheet_name="DR incidence factor"
)  # US rate by stage * DR factor (per age)
seer_inc = dr_inc.iloc[:65, :]
