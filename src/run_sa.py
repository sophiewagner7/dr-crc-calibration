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
from calibration_interp import simulated_annealing as interp_anneal
from calibration_lin_log import simulated_annealing as lin_log_anneal
from calibration_flat import simulated_annealing as flat_anneal


def run_sa(
    type="interp",
    n_iterations=100000,
    step_size=0.1,
    start_tmat=None,
    n_adj=33,
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
    elif type == "linlog":
        result = lin_log_anneal(
            n_iterations=n_iterations,
            step_size=step_size,
            start_tmat=start_tmat,
            n_adj=n_adj,
            verbose=verbose,
        )
    else:
        print("Wrong model specification")
        return

    # Generate the current timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    curr_tmat = result.copy()
    curr_log = m.run_markov_new(curr_tmat)
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
        out = np.zeros((8, 80))
        for idx, (from_state, to_state) in enumerate(c.points):
            out[idx] = curr_tmat[:, from_state, to_state]

        pd.DataFrame(out).to_csv(f"{output_dir}/tmats/{timestamp}_tps.csv")

    else:
        p.print_trans_probs(transition_probs)
        p.plot_tps(curr_tmat)
        p.plot_vs_seer(curr_log, c.seer_inc)
        p.plot_vs_seer_total(curr_log, c.seer_inc)

    return curr_tmat


run_sa(
    c.model_type,
    200000,
    0.1,
    save_all=True,
)
