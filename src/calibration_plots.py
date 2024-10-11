import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csaps import csaps  # https://csaps.readthedocs.io/en/latest/
import common_functions as func
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import seaborn as sns
import sys


# Function to extract transition probabilities
def extract_transition_probs(tmat, states, transitions):
    transition_probs = {}
    for from_state, to_state in transitions:
        from_idx = states[from_state]
        to_idx = states[to_state]
        params = tmat[:, from_idx, to_idx]
        transition_probs[f"{from_state} to {to_state}"] = params
    return transition_probs


def print_trans_probs(transition_probs, save_imgs=False, outpath=None, timestamp=None):
    print("Monthly transition probabilities")
    for transition, prob in transition_probs.items():
        min_prob = np.min(prob)
        max_prob = np.max(prob)
        avg_prob = np.mean(prob)
        print(
            f"{transition}: Min: {min_prob:.5f}, Max: {max_prob:.5f}, Average: {avg_prob:.5f}"
        )

    print("\nAnnual transition probabilities")
    for transition, prob in transition_probs.items():
        annual_probs = [func.probtoprob(p, 12, 1) for p in prob]
        min_annual_prob = np.min(annual_probs)
        max_annual_prob = np.max(annual_probs)
        avg_annual_prob = np.mean(annual_probs)
        print(
            f"{transition}: Min: {min_annual_prob:.5f}, Max: {max_annual_prob:.5f}, Average: {avg_annual_prob:.5f}"
        )

    if save_imgs:
        file_path = f"{outpath}/{timestamp}_transition_probs.txt"

        # Redirect stdout to a file
        with open(file_path, "w") as f:
            sys.stdout = f  # Redirect standard output to the file

            print("Monthly transition probabilities")
            for transition, prob in transition_probs.items():
                min_prob = np.min(prob)
                max_prob = np.max(prob)
                avg_prob = np.mean(prob)
                print(
                    f"{transition}: Min: {min_prob:.5f}, Max: {max_prob:.5f}, Average: {avg_prob:.5f}"
                )

            print("\nAnnual transition probabilities")
            for transition, prob in transition_probs.items():
                annual_probs = [func.probtoprob(p, 12, 1) for p in prob]
                min_annual_prob = np.min(annual_probs)
                max_annual_prob = np.max(annual_probs)
                avg_annual_prob = np.mean(annual_probs)
                print(
                    f"{transition}: Min: {min_annual_prob:.5f}, Max: {max_annual_prob:.5f}, Average: {avg_annual_prob:.5f}"
                )

        # Reset stdout back to default
        sys.stdout = sys.__stdout__


# Plotting
def plot_tps(curr_tmat, save_imgs=False, outpath=None, timestamp=None):
    plt.plot(func.probtoprob(curr_tmat[:, 3, 6], 12, 1), label="uLoc to dLoc")
    plt.plot(func.probtoprob(curr_tmat[:, 4, 7], 12, 1), label="uReg to dReg")
    plt.plot(func.probtoprob(curr_tmat[:, 5, 8], 12, 1), label="uDis to dDis")
    plt.legend()
    if save_imgs:
        plt.savefig(f"{outpath}/{timestamp}_params_detect.png")  # Save figure
        plt.close()
    else:
        plt.show()

    plt.plot(func.probtoprob(curr_tmat[:, 0, 1], 12, 1), label="Healthy to LR")
    plt.legend()
    if save_imgs:
        plt.savefig(f"{outpath}/{timestamp}_params_h_lr.png")  # Save figure
        plt.close()
    else:
        plt.show()

    plt.plot(func.probtoprob(curr_tmat[:, 1, 2], 12, 1), label="LR to HR")
    plt.plot(func.probtoprob(curr_tmat[:, 2, 3], 12, 1), label="HR to uLoc")
    plt.plot(func.probtoprob(curr_tmat[:, 3, 4], 12, 1), label="uLoc to uReg")
    plt.plot(func.probtoprob(curr_tmat[:, 4, 5], 12, 1), label="uReg to uDis")
    plt.legend()
    if save_imgs:
        plt.savefig(f"{outpath}/{timestamp}_params_progress.png")  # Save figure
        plt.close()
    else:
        plt.show()


def plot_vs_seer(curr_log, seer_inc, save_imgs=False, outpath=None, timestamp=None):
    """Plot model incidence by stage vs. SEER calibration incidence

    Args:
        curr_log (tuple): output log from run_markov. tuple containing inc_adj, ...
        seer_inc (df): item of comparison
    """
    inc_adj, _, _, _ = curr_log
    x_values = np.linspace(20, 99, 80)

    plt.plot(
        seer_inc["Age"],
        seer_inc["Local Rate"],
        label="Local (SEER)",
        color="b",
        linestyle="dotted",
    )
    plt.plot(
        seer_inc["Age"],
        seer_inc["Regional Rate"],
        label="Regional (SEER)",
        color="r",
        linestyle="dotted",
    )
    plt.plot(
        seer_inc["Age"],
        seer_inc["Distant Rate"],
        label="Distant (SEER)",
        color="g",
        linestyle="dotted",
    )
    plt.plot(x_values, inc_adj[6, :], label="Local (Model)", color="b")
    plt.plot(x_values, inc_adj[7, :], label="Regional (Model)", color="r")
    plt.plot(x_values, inc_adj[8, :], label="Distant (Model)", color="g")
    plt.legend()
    plt.title("Incidence of Local, Regional, and Distant States")
    plt.xlabel("Time Point / Age Group")
    plt.ylabel("incidence")
    if save_imgs:
        plt.savefig(f"{outpath}/{timestamp}_inc_stage.png")  # Save figure
        plt.close()
    else:
        plt.show()

    plt.plot(
        seer_inc["Age"],
        seer_inc["Local Rate"].cumsum(),
        label="Local (SEER)",
        color="b",
        linestyle="dotted",
    )
    plt.plot(
        seer_inc["Age"],
        seer_inc["Regional Rate"].cumsum(),
        label="Regional (SEER)",
        color="r",
        linestyle="dotted",
    )
    plt.plot(
        seer_inc["Age"],
        seer_inc["Distant Rate"].cumsum(),
        label="Distant (SEER)",
        color="g",
        linestyle="dotted",
    )
    plt.plot(x_values, inc_adj[6, :].cumsum(), label="Local (Model)", color="b")
    plt.plot(x_values, inc_adj[7, :].cumsum(), label="Regional (Model)", color="r")
    plt.plot(x_values, inc_adj[8, :].cumsum(), label="Distant (Model)", color="g")
    plt.legend()
    plt.title("Cumulative Incidence of Local, Regional, and Distant States")
    plt.xlabel("Time Point / Age Group")
    plt.ylabel("incidence")
    if save_imgs:
        plt.savefig(f"{outpath}/{timestamp}_inc_stage_cum.png")  # Save figure
        plt.close()
    else:
        plt.show()


def plot_prop_dr(inc_adj, stages, save_imgs=False, outpath=None, timestamp=None):
    """Plotting yearly incidence by stage, and dots for the total inc proportion"""
    calibration_proportions = stages
    model_total_inc = np.sum(inc_adj.iloc[6:9, :65], axis=0)
    model_proportions = [
        np.sum((inc_adj.iloc[stage, :65]) / np.sum(model_total_inc)) * 100
        for stage in [6, 7, 8]
    ]

    x = ["HGPS", "Model"]

    # Separate the stages into distinct arrays for plotting
    local = [calibration_proportions[0], model_proportions[0]]
    regional = [calibration_proportions[1], model_proportions[1]]
    distant = [calibration_proportions[2], model_proportions[2]]

    # Plot the bars with proper stacking
    plt.bar(x, local, color="b", label="Local")
    plt.bar(x, regional, bottom=local, color="r", label="Regional")
    plt.bar(
        x,
        distant,
        bottom=[i + j for i, j in zip(local, regional)],
        color="g",
        label="Distant",
    )

    # Customize the plot
    plt.ylabel("Percent")
    plt.title("Model vs Calibration Target - Cumulative Stage Distribution")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Save or show the plot
    if save_imgs:
        plt.savefig(f"{outpath}/{timestamp}_inc_stage.png")  # Save figure
        plt.close()
    else:
        plt.show()


def plot_vs_seer_total(
    curr_log, seer_inc, save_imgs=False, outpath=None, timestamp=None
):
    inc_adj, _, _, _ = curr_log
    x_values = np.arange(20, 100)
    seer_inc.loc[:, "Total Rate"] = (
        seer_inc["Local Rate"] + seer_inc["Regional Rate"] + seer_inc["Distant Rate"]
    )

    plt.plot(
        seer_inc["Age"],
        seer_inc["Total Rate"],
        label="SEER",
        color="b",
        linestyle="dotted",
    )
    plt.plot(x_values, np.sum(inc_adj[6:9, :], axis=0), label="Model", color="b")
    plt.legend()
    plt.title("Total Incidence (L+R+D)")
    plt.xlabel("Time Point / Age Group")
    plt.ylabel("incidence")
    if save_imgs:
        plt.savefig(f"{outpath}/{timestamp}_inc_total.png")  # Save figure
        plt.close()
    else:
        plt.show()

    plt.plot(
        seer_inc["Age"],
        seer_inc["Total Rate"].cumsum(),
        label="SEER",
        color="b",
        linestyle="dotted",
    )
    plt.plot(
        x_values, np.sum(inc_adj[6:9, :], axis=0).cumsum(), label="Model", color="b"
    )
    plt.legend()
    plt.title("Cumulative Incidence")
    plt.xlabel("Time Point / Age Group")
    plt.ylabel("incidence")
    if save_imgs:
        plt.savefig(f"{outpath}/{timestamp}_inc_total_cum.png")  # Save figure
        plt.close()
    else:
        plt.show()


def plot_prevs(inpath, outpath, timestamp):

    # Step 1: Read the CSV and transpose it
    pop = pd.read_csv(inpath, header=None).T  # Read and transpose

    # Step 2: Drop the first row (which could be headers or irrelevant data)
    pop = pop.drop(0)  # Equivalent to pop_t[-1, ] in R

    # Step 3: Assign column names to match the health states
    pop.columns = [
        "Healthy",
        "LR Polyp",
        "HR Polyp",
        "uLoc",
        "uReg",
        "uDis",
        "dLoc",
        "dReg",
        "dDis",
        "CSD",
        "healthy_ACM",
        "cancer_ACM",
        "polyp_ACM",
        "uCRC_ACM",
    ]

    # Step 4: Create a 'Year' column and group by each 12 rows, then calculate the mean
    pop["Year"] = np.repeat(
        np.arange(1, (len(pop) // 12) + 1), 12
    )  # Create 'Year' column
    pop_yr = (
        pop.groupby("Year").mean().reset_index()
    )  # Group by Year and calculate the mean

    # Step 5: Calculate the total ACM
    pop_yr["ACM"] = (
        pop_yr["healthy_ACM"]
        + pop_yr["cancer_ACM"]
        + pop_yr["polyp_ACM"]
        + pop_yr["uCRC_ACM"]
    )

    # Step 6: Convert the dataframe to a long format (equivalent to pivot_longer in R)
    pop_yr_long = pop_yr.melt(
        id_vars=["Year"],
        value_vars=[
            "Healthy",
            "LR Polyp",
            "HR Polyp",
            "uLoc",
            "uReg",
            "uDis",
            "dLoc",
            "dReg",
            "dDis",
            "CSD",
            "healthy_ACM",
            "cancer_ACM",
            "polyp_ACM",
            "uCRC_ACM",
            "ACM",
        ],
        var_name="Health_State",
        value_name="Value",
    )

    # Step 7: Normalize values for percentages (similar to dividing by 100,000)
    pop_yr_long["perc"] = pop_yr_long["Value"] / 100000

    # Step 8: Plot the health states
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=pop_yr_long, x="Year", y="perc", hue="Health_State", linewidth=1)

    # Customize the plot
    plt.title("Overlaid Health States with Areas and Lines")
    plt.xlabel("Year")
    plt.ylabel("Prevalence (Normalized)")
    plt.legend(title="Health State", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Step 9: Save the plot to the output directory
    plot_file = f"{outpath}/{timestamp}_health_states.png"
    plt.savefig(plot_file)
    plt.close()  # Close the plot to free memory
    print(f"Plot saved to {plot_file}")
