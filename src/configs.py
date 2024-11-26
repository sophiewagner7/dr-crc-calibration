import pandas as pd
import numpy as np
import common_functions as func
from scipy.interpolate import interp1d

# Global Parameters
starting_age = 20
max_age = 84
N = 100000  # Size of sample populations
model_version = "US"  # US, DR
model_tps = "all"  # non_progress, all
stage_DR = "HGPS"  # HGPS, SEER
data_interval = 5  # 1-year or 5-year data

OUTPUT_PATHS = {
    "interp": f"../out/{model_version}/interp/",
}

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

desired_transitions = [
    ("healthy", "LR_polyp"),
    ("LR_polyp", "HR_polyp"),
    ("HR_polyp", "u_CRC_loc"),
    ("u_CRC_loc", "u_CRC_reg"),
    ("u_CRC_reg", "u_CRC_dis"),
    ("u_CRC_loc", "d_CRC_loc"),
    ("u_CRC_reg", "d_CRC_reg"),
    ("u_CRC_dis", "d_CRC_dis"),
]

# Age indices for the model
ages_1y = np.arange(starting_age, max_age, 1)
ages_5y = np.arange(starting_age, max_age, 5)
age_layers_1y = np.arange(0, len(ages_1y), 1)
age_layers_5y = np.arange(0, len(ages_5y), 1)
age_layers = age_layers_5y if data_interval == 5 else age_layers_1y

# Initial population state
starting_pop = np.zeros((len(health_states), 1))
starting_pop[0, 0] = N  # Everyone starts in healthy state

# Inputs


# Helper Functions for Readability
def load_acm_us(data_interval, age_layers):
    """Load and process all-cause mortality for the US model."""
    acm_1y = pd.read_excel("../data/acm_us.xlsx", sheet_name="ACM_1y")
    acm_1y["Age Group"] = (acm_1y["Age"] // 5) * 5
    acm_5y = acm_1y.groupby("Age Group")["Rate"].mean().reset_index()
    acm_5y = acm_5y.query("20 <= `Age Group` < 100").reset_index(drop=True)
    acm_rate_5y = func.probtoprob(acm_5y["Rate"]).to_numpy()

    acm_1y = acm_1y.query("20 <= Age < 100").reset_index(drop=True)
    acm_rate_1y = func.probtoprob(acm_1y["Rate"]).to_numpy()

    acm_rate = acm_rate_1y if data_interval == 1 else acm_rate_5y
    return acm_rate[: len(age_layers)]


def load_csd(seer_surv, data_interval):
    """Load and interpolate cancer-specific death rates."""
    if data_interval == 1:
        age_points, new_age_points = np.arange(20, 100, 5), np.arange(20, 100)
        seer_surv_1y = {
            col: interp1d(
                age_points, seer_surv[col], kind="linear", fill_value="extrapolate"
            )(new_age_points)
            for col in ["Local", "Regional", "Distant"]
        }
        seer_surv = seer_surv_1y
    return pd.DataFrame(seer_surv).apply(lambda col: func.probtoprob(col)).to_numpy()


def load_seer_incidence(seer_inc):
    """Load SEER incidence data for calibration."""
    return seer_inc.query("20 <= Age <= 84").reset_index(drop=True)


def load_acm_dr():
    """Load and process all-cause mortality for the DR model."""
    acm_dr = pd.read_excel("../data/acm_dr.xlsx", sheet_name="ACM_1Y")
    return np.array(list(map(func.probtoprob, acm_dr["Prob"].to_numpy()[:-1])))


def load_stage_distribution(dr_inc, dr_stage_dist):
    """Calculate stage-specific incidence rates for DR model."""
    stage_rates = dr_inc.values[:, np.newaxis] * dr_stage_dist.values
    return pd.DataFrame(
        {
            "Age": np.arange(20, 85),
            "Local Rate": stage_rates[:, 0],
            "Regional Rate": stage_rates[:, 1],
            "Distant Rate": stage_rates[:, 2],
            "Total Rate": dr_inc,
        }
    )


# Main Inputs Processing
if model_version == "US":
    # Load US-specific data
    acm_rate = load_acm_us(data_interval=5, age_layers=age_layers)
    seer_surv_us = (
        pd.read_excel("../data/survival_km.xlsx", sheet_name="Survival")
        .query("Age < 100")
        .reset_index(drop=True)
    )
    csd_rate = load_csd(seer_surv_us, data_interval)[:, 1:]
    seer_inc = pd.read_excel("../data/incidence_crude.xlsx", sheet_name="1975-1990 Adj")
    seer_inc = load_seer_incidence(seer_inc)

    # Load polyp prevalence data for calibration
    polyp_prev = pd.read_excel("../data/polyp_targets.xlsx", sheet_name="Sheet1")
    polyp_targets = polyp_prev["Value"].to_numpy()  # uCRC, polyp, uCRC + polyp

elif model_version == "DR":
    # Set transition points for DR model
    if model_tps == "non_progress":
        points = [(0, 1), (3, 6), (4, 7), (5, 8)]

    # Load DR-specific data
    acm_rate = load_acm_dr()
    seer_surv_dr = (
        pd.read_excel("../data/survival_km.xlsx", sheet_name="Survival")
        .query("Age < 100")
        .reset_index(drop=True)
    )
    csd_rate = load_csd(seer_surv_dr, data_interval)

    # Load yearly cancer incidence and stage distribution
    dr_total_inc = pd.read_csv("../data/dr_inc_splined.csv")["Rate"]
    hgps_stage_dist = pd.read_excel(
        "../data/incidence_dr_globocan.xlsx", sheet_name="HGPS Stage"
    )["Percent"]
    dr_inc_stage = pd.read_excel(
        "../data/incidence_dr_globocan.xlsx", sheet_name="DR incidence factor"
    )
    if stage_DR == "HGPS":
        seer_inc = load_stage_distribution(dr_total_inc, hgps_stage_dist)
    else:
        seer_inc = load_seer_incidence(dr_inc_stage)

    # Load polyp prevalence data and apply DR adjustment factor
    polyp_prev = pd.read_excel("../data/polyp_targets.xlsx", sheet_name="Sheet1")
    polyp_targets = polyp_prev["Value"].to_numpy() * 0.416391039  # DR adjustment factor
