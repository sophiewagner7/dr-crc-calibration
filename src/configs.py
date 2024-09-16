import pandas as pd
import numpy as np
import common_functions as func
from scipy.interpolate import interp1d

# Global Parameters
starting_age = 20
max_age = 100
N = 100000  # Size of sample populations
model_type = "logis_healthy_lr"  # linear, logis_all, logis_healthy_lr, logis_linear

# Global strategy parameters
screen_start = 45
screen_end = 75
surveil_end = 85

# State Structure
health_states = {
    "healthy": 0,
    "LR_polyp": 1,
    "HR_polyp": 2,
    "u_CRC_loc": 3,
    "u_CRC_reg": 4,
    "u_CRC_dis": 5,
    "d_CRC_loc": 6,
    "d_CRC_reg": 7,
    "d_CRC_dis": 8,
    "cancer_death": 9,
    "healthy_ACM": 10,
    "cancer_ACM": 11,
    "polyp_ACM": 12,  # death while in polyp state
    "uCRC_ACM": 13,  # death while in undiagnosed state
}

# Mapping ACM states
acm_states = [10, 12, 12, 13, 13, 13, 11, 11, 11]

# Transition points
points = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (4, 7), (5, 8)]

# Age indices for the model
ages = np.arange(0, 80, 1)

# Initial population state
starting_pop = np.zeros((len(health_states), 1))
starting_pop[0, 0] = N  # Everyone starts in healthy state

# Inputs
data_interval = 1  # 1-year or 5-year data

# All cause mortality
acm_1y = pd.read_excel("../data/acm_us.xlsx", sheet_name="ACM_1y")  # Age, Rate
acm_1y["Age Group"] = (acm_1y["Age"] // 5) * 5
acm_5y = acm_1y.groupby("Age Group")["Rate"].mean().reset_index()
acm_5y = acm_5y[acm_5y["Age Group"] >= 20].reset_index()  # age_layers 20-100 (16 items)
acm_5y = acm_5y[acm_5y["Age Group"] < 100].reset_index()  # age_layers 20-100 (16 items)
acm_rate_5y = func.probtoprob(acm_5y["Rate"]).to_numpy()
acm_1y = acm_1y[acm_1y["Age"] >= 20].reset_index()
acm_1y = acm_1y[acm_1y["Age"] < 100].reset_index(drop=True)
acm_rate_1y = func.probtoprob(acm_1y["Rate"]).to_numpy()
acm_rate = acm_rate_1y if data_interval == 1 else acm_rate_5y

# Cancer specific death
seer_surv = pd.read_excel(
    "../data/survival_km.xlsx", sheet_name="Survival"
).reset_index(
    drop=True
)  # In 5y age layers
seer_surv = seer_surv[seer_surv["Age"] < 100]
# csd_rate = seer_surv[['Local', 'Regional', 'Distant']].apply(lambda col: func.probtoprob(col)).to_numpy() # Convert to monthly probs
age_points = np.arange(20, 100, 5)  # Original age points (every 5 years)
new_age_points = np.arange(20, 100)  # New age points (every year)

csd_interp = {}
for col in ["Local", "Regional", "Distant"]:
    f = interp1d(age_points, seer_surv[col], kind="linear", fill_value="extrapolate")  # type: ignore
    csd_interp[col] = f(new_age_points)

csd_rate = pd.DataFrame(csd_interp).apply(lambda col: func.probtoprob(col)).to_numpy()


# Calibration Targets
# Target 1: SEER Incidence
seer_inc = pd.read_excel("../data/incidence_crude.xlsx", sheet_name="1975-1990 Adj")
seer_inc = seer_inc[seer_inc["Age"] >= 20].reset_index()  # single ages, 20-84 (65 ages)
seer_inc = seer_inc[seer_inc["Age"] <= 84].reset_index()  # starting age 20, 65 ages

# Target 2: Polyp prevalence
polyp_prev = pd.read_excel("../data/polyp_targets.xlsx", sheet_name="Sheet1")
polyp_targets = polyp_prev["Value"].to_numpy()  # uCRC, polyp, uCRC + polyp
