import pandas as pd
import numpy as np
import common_functions as func
from scipy.interpolate import interp1d

# Global Parameters
starting_age = 20
max_age = 100
N = 100000  # Size of sample populations
model_type = "interp"  # linear, logis_all, logis_healthy_lr, logis_linear
model_version = "US"  # US, DR
dr_stage_penalty = "Yearly"  # Yearly, Total

# Global strategy parameters
screen_start = 45
screen_end = 75
surveil_end = 85

OUTPUT_PATHS = {
    "interp": f"../out/{model_version}/interp/",
    "lin_log": f"../out/{model_version}/log_lin/",
    "flat": f"../out/{model_version}/flat/",
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
points_linear = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (4, 7), (5, 8)]
points_logis = [(0, 1)]
idx_linear = np.arange(1, 8)
idx_logis = np.array([1])

# Age indices for the model
age_layers = np.arange(0, (max_age - starting_age), 1)
ages_5y = (age_layers // 5) * 5
age_layers_5y = np.arange(0, len(ages_5y), 1)

# Initial population state
starting_pop = np.zeros((len(health_states), 1))
starting_pop[0, 0] = N  # Everyone starts in healthy state

# Inputs
data_interval = 1  # 1-year or 5-year data

if model_version == "US":
    # All cause mortality
    acm_1y = pd.read_excel("../data/acm_us.xlsx", sheet_name="ACM_1y")  # Age, Rate
    acm_1y["Age Group"] = (acm_1y["Age"] // 5) * 5
    acm_5y = acm_1y.groupby("Age Group")["Rate"].mean().reset_index()
    acm_5y = acm_5y[
        acm_5y["Age Group"] >= 20
    ].reset_index()  # age_layers 20-100 (16 items)
    acm_5y = acm_5y[
        acm_5y["Age Group"] < 100
    ].reset_index()  # age_layers 20-100 (16 items)
    acm_rate_5y = func.probtoprob(acm_5y["Rate"]).to_numpy()
    acm_1y = acm_1y[acm_1y["Age"] >= 20].reset_index()
    acm_1y = acm_1y[acm_1y["Age"] < 100].reset_index(drop=True)
    acm_rate_1y = func.probtoprob(acm_1y["Rate"]).to_numpy()
    acm_rate = acm_rate_1y if data_interval == 1 else acm_rate_5y
    acm_rate = acm_rate[: len(age_layers)]
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

    csd_rate = (
        pd.DataFrame(csd_interp).apply(lambda col: func.probtoprob(col)).to_numpy()
    )
    csd_rate = csd_rate[: len(age_layers)]

    # Calibration Targets
    # Target 1: SEER Incidence
    seer_inc = pd.read_excel("../data/incidence_crude.xlsx", sheet_name="1975-1990 Adj")
    seer_inc = seer_inc[
        seer_inc["Age"] >= 20
    ].reset_index()  # single age_layers, 20-84 (65 age_layers)
    seer_inc = seer_inc[
        seer_inc["Age"] <= 84
    ].reset_index()  # starting age 20, 65 age_layers

    # Target 2: Polyp prevalence
    polyp_prev = pd.read_excel("../data/polyp_targets.xlsx", sheet_name="Sheet1")
    polyp_targets = polyp_prev["Value"].to_numpy()  # uCRC, polyp, uCRC + polyp

elif model_version == "DR":
    ### Inputs
    # All cause mortality
    acm_dr = pd.read_excel("../data/acm_dr.xlsx", sheet_name="ACM_1Y")
    acm_rate = acm_dr["Prob"].to_numpy()[:-1]
    acm_rate = np.array(list(map(func.probtoprob, acm_rate)))

    # Cancer specific death
    seer_surv = pd.read_excel(
        "../data/survival_km.xlsx", sheet_name="Survival"
    ).reset_index(
        drop=True
    )  # In 5y age layers
    seer_surv = seer_surv[seer_surv["Age"] < 100]
    age_points = np.arange(20, 100, 5)  # Original age points (every 5 years)
    new_age_points = np.arange(20, 100)  # New age points (every year)

    csd_interp = {}
    for col in ["Local", "Regional", "Distant"]:
        f = interp1d(age_points, seer_surv[col], kind="linear", fill_value="extrapolate")  # type: ignore
        csd_interp[col] = f(new_age_points)

    csd_rate = (
        pd.DataFrame(csd_interp).apply(lambda col: func.probtoprob(col)).to_numpy()
    )

    ### Calibration Targets
    # Target 1: SEER Incidence
    # dr_inc = pd.read_excel("../data/incidence_crude.xlsx", sheet_name="1975-1990 Adj")
    # dr_inc = dr_inc[dr_inc["Age"] >= 20].reset_index()  # single ages, 20-84 (65 ages)
    # dr_inc = dr_inc[dr_inc["Age"] <= 84].reset_index()  # starting age 20, 65 ages
    # dr_inc = pd.read_excel(
    #     "../data/incidence_dr_globocan.xlsx", sheet_name="DR incidence factor"
    # )  # US rate by stage * DR factor (per age)
    # seer_inc = dr_inc.iloc[:65, :]

    # Yearly cancer incidence
    dr_inc = pd.read_csv("../data/dr_inc_splined.csv")["Rate"]

    # Stage distribution (proportion of total, HGPS)
    dr_stage_dist = pd.read_excel(
        "../data/incidence_dr_globocan.xlsx", sheet_name="HGPS Stage"
    )["Percent"]

    # Structure like so for plotting
    stage_rates = dr_inc.values[:, np.newaxis] * dr_stage_dist.values

    # Create the DataFrame directly using the calculated stage rates
    seer_inc = pd.DataFrame(
        {
            "Age": np.arange(20, 85),
            "Local Rate": stage_rates[:, 0],
            "Regional Rate": stage_rates[:, 1],
            "Distant Rate": stage_rates[:, 2],
            "Total Rate": dr_inc,
        }
    )

    dr_stage_dist *= 100
    # Target 2: Polyp prevalence
    polyp_prev = pd.read_excel("../data/polyp_targets.xlsx", sheet_name="Sheet1")
    polyp_targets = polyp_prev["Value"].to_numpy()  # uCRC, polyp, uCRC + polyp
    dr_factor_flat = 0.416391039
    polyp_targets *= dr_factor_flat
