import pandas as pd
import numpy as np
import common_functions as func
from scipy.interpolate import interp1d

# Global Parameters
starting_age = 20
max_age = 100
N = 100000  # Size of sample populations
model_version = "US"  # US, DR
model_tps = "all"  # non_progress, all
data_interval = 1  # 1-year or 5-year data
inc_factor, polyp_factor = 1, 1 # Target ratios: [.5, 1, 2]
stage = "SEER"  # ["HGPS", "SEER"]
output_file = f"{stage}_I{inc_factor}_P{polyp_factor}" if model_version == "DR" else "bc" 

# Paths 
OUTPUT_PATHS = {
    "logs": f"../out/{model_version}/{output_file}/logs",  # inc and prev logs
    "tmats": f"../out/{model_version}/{output_file}/tmats",  # .npy tmats
    "plots": f"../out/{model_version}/{output_file}/plots",  # param and incidence plots
    "probs": f"../out/{model_version}/{output_file}/probs"  # probs for treeage
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

transitions_itos = zip(points, desired_transitions)

# Age indices for the model
ages_1y = np.arange(starting_age, max_age+1, 1)
ages_5y = np.arange(starting_age, max_age, 5)
age_layers_1y = np.arange(0, len(ages_1y), 1)
age_layers_5y = np.arange(0, len(ages_5y), 1)
age_layers = age_layers_5y if data_interval == 5 else age_layers_1y

# Initial population state
starting_pop = np.zeros((len(health_states), 1))
starting_pop[0, 0] = N  # Everyone starts in healthy state

# Helper Functions for Readability
def load_acm_us():
    """Load and process all-cause mortality for the US model."""
    acm_1y = pd.read_excel("../data/acm_us.xlsx", sheet_name="ACM_1y")
    acm_1y["Age Group"] = (acm_1y["Age"] // 5) * 5
    acm_5y = acm_1y.groupby("Age Group")["Rate"].mean().reset_index()
    acm_5y_100 = acm_5y.query("20 <= `Age Group` < 100").reset_index(drop=True)
    acm_5y_85 = acm_5y.query("20 <= `Age Group` < 85").reset_index(drop=True)
    acm_rate_5y_100 = func.probtoprob(acm_5y_100["Rate"]).to_numpy()
    acm_rate_5y_85 = func.probtoprob(acm_5y_85["Rate"]).to_numpy()

    acm_1y_100 = acm_1y.query("20 <= Age < 100").reset_index(drop=True)
    acm_1y_85 = acm_1y.query("20 <= Age < 85").reset_index(drop=True)
    acm_rate_1y_100 = func.probtoprob(acm_1y_100["Rate"]).to_numpy()
    acm_rate_1y_85 = func.probtoprob(acm_1y_85["Rate"]).to_numpy()

    acm_rate_100 = acm_rate_1y_100 if data_interval == 1 else acm_rate_5y_100
    acm_rate_85 = acm_rate_1y_85 if data_interval == 1 else acm_rate_5y_85
    return acm_rate_100, acm_rate_85

def load_csd(yrs):
    def manipulate_csd(data):
        """Load cancer-specific death rates."""
        age_columns = [col for col in data.columns if col.startswith('AGE_')]
        mean_values = data.loc[data['YEARS'] < 5, age_columns].mean()
        transformed_data = pd.DataFrame()

        # Repeat rows for each age group
        for age_group, mean_value in mean_values.items():
            # Determine the number of rows based on the age group range
            age_range = age_group.split('_')[1:]
            start_age = int(age_range[0]) + 1
            end_age = int(age_range[1]) if len(age_range) > 1 else start_age
            if end_age == 84:
                end_age = 99
            row_count = end_age - start_age + 1

            # Add repeated rows for this age group
            age_df = pd.DataFrame({
                'AGE': range(start_age, end_age + 1),           
                'VALUE': [mean_value] * row_count
            })

            transformed_data = pd.concat([transformed_data, age_df], ignore_index=True)
        
        return transformed_data

    def aggregate_to_5_year_intervals(transformed_data):
        """Aggregate data to 5-year intervals."""
        transformed_data['AGE_GROUP'] = (transformed_data['AGE'] // 5) * 5 + 2.5
        aggregated_data = (
            transformed_data.groupby('AGE_GROUP', as_index=False)
            .agg({'VALUE': 'mean'})
            .rename(columns={'AGE_GROUP': 'AGE'})
        )
        return aggregated_data

    def filter_and_aggregate(data, max_age, interval=5):
        """Filter and aggregate data for a given age range."""
        filtered_data = data.query('AGE < @max_age').reset_index(drop=True)
        if interval == 5:
            return aggregate_to_5_year_intervals(filtered_data)
        return filtered_data

    if data_interval == 5:
    
        csd_85 = np.zeros((13, 3))  # 13 intervals for ages < 85
        csd_100 = np.zeros((16, 3)) # 16 intervals for ages < 100
        i = 0
        for stage in ["loc", "reg", "dis"]:
            dat = pd.read_csv(f"../data/s8_probs_{stage}_{yrs}.csv")
            transformed_data = manipulate_csd(dat).query('AGE >= 20').reset_index(drop=True)

            # Filter and aggregate for ages < 85
            data_below_85 = filter_and_aggregate(transformed_data, max_age=85)
            csd_85[:, i] = data_below_85['VALUE'].to_numpy()

            # Filter and aggregate for ages < 100
            data_below_100 = filter_and_aggregate(transformed_data, max_age=100)
            csd_100[:, i] = data_below_100['VALUE'].to_numpy()
            
            i += 1
    else: 
        csd_85 = np.zeros((65, 3))  # 65 intervals for ages 20 - 85
        csd_100 = np.zeros((80, 3)) # 80 intervals for ages 20 - 100
        i = 0
        for stage in ["loc", "reg", "dis"]:
            dat = pd.read_csv(f"../data/s8_probs_{stage}_{yrs}.csv")
            transformed_data = manipulate_csd(dat).query('AGE >= 20').reset_index(drop=True)
            
            data_below_85 = transformed_data.query('AGE < 85').reset_index(drop=True)
            data_below_100 = transformed_data.query('AGE < 100').reset_index(drop=True)
            
            csd_85[:, i] = data_below_85['VALUE'].to_numpy()
            csd_100[:, i] = data_below_100['VALUE'].to_numpy()
            
            i += 1

    return csd_100, csd_85


def load_seer_incidence(seer_inc):
    """Load SEER incidence data for calibration."""
    seer_inc = seer_inc.query("20 <= Age <= 84").reset_index(drop=True)
    seer_inc['Local Rate'] *= inc_factor
    seer_inc['Regional Rate'] *= inc_factor
    seer_inc['Distant Rate'] *= inc_factor
    return seer_inc.query("20 <= Age <= 84").reset_index(drop=True)


def load_acm_dr():
    """Load and process all-cause mortality for the DR model."""
    if data_interval == 1:
        acm_dr = pd.read_excel("../data/acm_dr.xlsx", sheet_name="ACM_1Y")
    else:
        acm_dr = pd.read_excel("../data/acm_dr.xlsx", sheet_name="ACM_5Y")
    acm_dr = np.array(list(map(func.probtoprob, acm_dr["Prob"].to_numpy()[:-1])))
    return acm_dr, acm_dr[:65]


def load_stage_distribution(dr_inc, dr_stage_dist):
    """Calculate stage-specific incidence rates for DR model."""
    stage_rates = dr_inc.values[:, np.newaxis] * dr_stage_dist.values
    return pd.DataFrame(
        {
            "Age": np.arange(20, 85),
            "Local Rate": stage_rates[:, 0] * inc_factor,
            "Regional Rate": stage_rates[:, 1] * inc_factor,
            "Distant Rate": stage_rates[:, 2] * inc_factor,
            "Total Rate": dr_inc,
        }
    )


# Inputs
if model_version == "US":
    
    ## INPUTS ##
    
    # Load mortality inputs
    acm_rate_100, acm_rate_85 = load_acm_us()  # ACM
    csd_rate_100, csd_rate_85 = load_csd(yrs = "1996_1999")  # CSD (3, 80)
    acm_rate, csd_rate = acm_rate_100, csd_rate_100

    ## TARGETS ##
    
    # Load incidence
    seer_inc = pd.read_excel("../data/incidence_crude.xlsx", sheet_name="1975-1990 Adj")
    seer_inc = load_seer_incidence(seer_inc)

    # Load polyp prevalence data for calibration
    polyp_prev = pd.read_excel("../data/polyp_targets.xlsx", sheet_name="Sheet1")
    polyp_targets = polyp_prev["Value"].to_numpy()  # uCRC, polyp, uCRC + polyp
    polyp_targets *= polyp_factor  # apply factor per model version

# Inputs
elif model_version == "DR":
    
    ## INPUTS ##
    
    # Set transition points for DR model
    if model_tps == "non_progress":
        points = [(0, 1), (3, 6), (4, 7), (5, 8)]

    # Load mortality inputs
    acm_rate_100, acm_rate_85 = load_acm_dr()  # ACM
    csd_rate_100, csd_rate_85 = load_csd(yrs = "1975_1985")  # CSD
    acm_rate, csd_rate = acm_rate_85, csd_rate_85

    ## TARGETS ##
    
    # Load yearly cancer incidence and stage distribution
    dr_total_inc = pd.read_csv("../data/dr_inc_splined.csv")["Rate"]
    hgps_stage_dist = pd.read_excel("../data/incidence_dr_globocan.xlsx", sheet_name="HGPS Stage")["Percent"]
    dr_inc_stage = pd.read_excel("../data/incidence_dr_globocan.xlsx", sheet_name="DR incidence factor")
    if stage == "HGPS":
        seer_inc = load_stage_distribution(dr_total_inc, hgps_stage_dist)
    else:
        seer_inc = load_seer_incidence(dr_inc_stage)
    
    # Load polyp prevalence data and apply DR adjustment factor
    polyp_prev = pd.read_excel("../data/polyp_targets.xlsx", sheet_name="Sheet1")
    polyp_targets = polyp_prev["Value"].to_numpy() * 0.416391039 # DR adjustment factor
    polyp_targets *= polyp_factor  # apply factor per model version