import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # For visualizations
from csaps import csaps  # https://csaps.readthedocs.io/en/latest/
import common_functions as func
import configs as c
import markov as m
import gof
import calibration_plots as p
from datetime import datetime
import calibration_interp as ci
from calibration_interp import simulated_annealing as interp_anneal
from calibration_flat import simulated_annealing as flat_anneal


def run_sa(
    type="interp",
    n_iterations=100000,
    step_size=0.1,
    start_tmat=None,
    n_adj=25,
    verbose=True,
    save_all=True,
):
    if type == "interp":
        result = interp_anneal(
            n_iterations=n_iterations,
            step_size=step_size,
            start_tmat=start_tmat,
            n_adj=n_adj,
            verbose=verbose,
        )
    elif type == "flat":
        result = flat_anneal(
            n_iterations=n_iterations,
            step_size=step_size,
            start_tmat=start_tmat,
            n_adj=n_adj,
            verbose=verbose,
        )
    else:
        print(f"Wrong model specification: {type}")
        return

    # Generate the current timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    curr_tmat = result.copy()
    curr_log = m.run_markov_new(curr_tmat, max_age=c.max_age)
    log_adj, log_prev, log_pop, log_inc = curr_log

    # Extract transition probabilities
    transition_probs = p.extract_transition_probs(
        curr_tmat, c.health_states, c.desired_transitions
    )

    # Saving
    if save_all:
        # Save the with the timestamp in the filenames
        output_dir = c.OUTPUT_PATHS[type]
        np.save(f"{output_dir}/tmats/{timestamp}_tmat.npy", curr_tmat)
        pd.DataFrame(log_adj).to_csv(f"{output_dir}/logs/{timestamp}_inc_adj.csv")
        pd.DataFrame(log_prev).to_csv(f"{output_dir}/logs/{timestamp}_prev.csv")
        pd.DataFrame(log_pop).to_csv(f"{output_dir}/logs/{timestamp}_pop.csv")
        pd.DataFrame(log_inc).to_csv(f"{output_dir}/logs/{timestamp}_inc_unadj.csv")

        p.print_trans_probs(
            transition_probs,
            save_imgs=True,
            outpath=f"{output_dir}/probs/",
            timestamp=timestamp,
        )
        p.plot_tps(
            curr_tmat,
            save_imgs=True,
            outpath=f"{output_dir}/plots",
            timestamp=timestamp,
        )
        p.plot_vs_seer(
            curr_log,
            c.seer_inc,
            save_imgs=True,
            outpath=f"{output_dir}/plots",
            timestamp=timestamp,
        )
        p.plot_vs_seer_total(
            curr_log,
            c.seer_inc,
            save_imgs=True,
            outpath=f"{output_dir}/plots",
            timestamp=timestamp,
        )
        out = np.zeros((len(c.points), len(c.age_layers)))
        for idx, (from_state, to_state) in enumerate(c.points):
            out[idx] = curr_tmat[:, from_state, to_state]

        pd.DataFrame(out).to_csv(f"{output_dir}/tmats/{timestamp}_tps.csv")

    else:
        p.print_trans_probs(transition_probs)
        p.plot_tps(curr_tmat)
        p.plot_vs_seer(curr_log, c.seer_inc)
        p.plot_vs_seer_total(curr_log, c.seer_inc)

    return curr_tmat


t = np.load("../out/US/interp/tmats/20240923_1243_tmat.npy")
tmat5c = np.copy(t)
tmat5c[:, 0, 1] = np.maximum(
    tmat5c[:, 0, 1], func.probtoprob(0.001)
)  # Set lower bound for Norm to LR
tmat5c[:, 1, 2] = np.maximum(
    tmat5c[:, 1, 2], func.probtoprob(0.0075)
)  # Set lower bound for LR to HR
tmat5c[:, 2, 3] = np.maximum(
    tmat5c[:, 2, 3], func.probtoprob(0.04)
)  # Set lower bound for HR to uLoc
tmat5c[:, 3, 4] = np.minimum(
    tmat5c[:, 3, 4], func.probtoprob(0.4)
)  # Set upper bound for uLoc to uReg
tmat5c[:, 4, 5] = np.mean(tmat5c[:, 4, 5])  # Make constant
tmat5c[:, 4, 7] = np.minimum(
    tmat5c[:, 4, 7], func.probtoprob(0.7)
)  # Set upper bound for detecting Reg cancer
tmat5c[:, 4, 7] = np.maximum(
    tmat5c[:, 4, 7], func.probtoprob(0.55)
)  # Set lower bound for detecting Reg cancer
tmat5c[:, 0, 1] *= 0.5

# def get_5y(tmat):
#     for from_state, to_state in c.points:
#         age_mids = c.ages_1y
#         all_ages = np.arange(22.5, 102.5, 5)
#         new_matrix = np.zeros(
#             (len(c.ages_5y), len(c.health_states), len(c.health_states))
#         )
#         for from_state, to_state in c.points:
#             smoothed_params = csaps(
#                 age_mids, tmat[:, from_state, to_state], smooth=0.05
#             )(all_ages).clip(0.000001, 0.4)
#             new_matrix[:, from_state, to_state] = smoothed_params
#     return new_matrix

tmat5c = ci.add_acm(tmat5c)
tmat5c = ci.add_csd(tmat5c)
tmat5c = ci.row_normalize(tmat5c)

run_sa(
    type=c.model_type,
    n_iterations=200000,
    step_size=0.01,
    n_adj=11,
    start_tmat=tmat5c,
    save_all=True,
)
