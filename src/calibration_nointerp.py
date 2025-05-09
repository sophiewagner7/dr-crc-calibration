import numpy as np
import pandas as pd
from csaps import csaps
from tqdm import tqdm

import configs as c
import common_functions as func
import markov as m
import gof
import calibration_plots as p


def row_normalize(matrix):
    for age_layer in range(matrix.shape[0]):  # Loop over each age layer
        layer = matrix[age_layer]
        # Calculate the sum of non-diagonal elements for each row
        sum_of_columns = np.sum(layer, axis=1) - np.diag(layer)
        # Set the diagonal elements
        np.fill_diagonal(layer, 1 - sum_of_columns)
    return matrix


def initialize_params():
    # Logis parameters: base, increase, offset, spread
    params = np.zeros((len(c.points), 4))
    params[0] = [0.005, 0.001, 50, 0.1]  # Healthy to LR
    params[1] = [0.015, 0.001, 50, 0.1]  # LR to HR
    params[2] = [0.05, 0.001, 50, 0.1]  # HR to uLoc
    params[3] = [0.45, 0.001, 50, 0.1]  # uLoc to uReg
    params[4] = [0.50, 0.001, 50, 0.1]  # uReg to uDis
    params[5] = [0.20, 0.001, 50, 0.1]  # uLoc to dLoc
    params[6] = [0.60, 0.001, 50, 0.1]  # uReg to dReg
    params[7] = [0.90, 0.001, 50, 0.1]  # uDis to dDis
    return params


def create_matrix(params=None):
    if params is None:
        params = initialize_params()

    matrix = np.zeros(
        (len(c.age_layers_5y), len(c.health_states), len(c.health_states))
    )
    for idx, (from_state, to_state) in enumerate(c.points):
        matrix[:, from_state, to_state] = func.probtoprob(params[idx, 0])  # monthly

    matrix = add_acm(matrix)  # ACM
    matrix = add_csd(matrix)  # CSD
    matrix = constrain_matrix(matrix)  # constrain
    matrix = row_normalize(matrix)  # normalize

    return params, matrix


def constrain_matrix(matrix):
    matrix = np.clip(matrix, 0.0, 0.5)

    # Progression Block
    matrix[:, 0, 1] = np.maximum(0.000001, matrix[:, 0, 1])  # not below 0
    matrix[:, 1, 2] = np.maximum(
        matrix[:, 0, 1], matrix[:, 1, 2]
    )  # HR to LR > healthy to uLoc
    matrix[:, 2, 3] = np.maximum(matrix[:, 1, 2], matrix[:, 2, 3])
    matrix[:, 3, 4] = np.maximum(matrix[:, 2, 3], matrix[:, 3, 4])
    matrix[:, 4, 5] = np.maximum(matrix[:, 3, 4], matrix[:, 4, 5])

    # Detection Block
    matrix[:, 3, 6] = np.maximum(0, matrix[:, 3, 6])  # not below 0
    matrix[:, 4, 7] = np.maximum(matrix[:, 3, 6], matrix[:, 4, 7])  # uR dR > uL dL
    matrix[:, 5, 8] = np.maximum(matrix[:, 4, 7], matrix[:, 5, 8])  # dR < dD

    return matrix


def add_acm(matrix):
    matrix[:, 0, 10] = c.acm_rate  # Healthy to ACM
    matrix[:, 1:3, 12] = c.acm_rate[:, np.newaxis]  # Polyp to ACM
    matrix[:, 3:6, 13] = c.acm_rate[:, np.newaxis]  # Undiagnosed to ACM
    matrix[:, 6:9, 11] = c.acm_rate[:, np.newaxis]  # Cancer to ACM
    matrix[:, 9, 9] = 1  # Stay in CSD
    matrix[:, 10, 10] = 1  # Stay in ACM
    matrix[:, 11, 11] = 1  # Stay in Cancer ACM
    matrix[:, 12, 12] = 1  # Stay in Polyp ACM
    matrix[:, 13, 13] = 1  # Stay in uCRC ACM

    return matrix


def add_csd(matrix):
    matrix[:, 6, 9] = c.csd_rate[:, 0]
    matrix[:, 7, 9] = c.csd_rate[:, 1]
    matrix[:, 8, 9] = c.csd_rate[:, 2]
    return matrix


def step(matrix, step_size, num_adj=21):
    new_matrix = np.copy(matrix)
    step_mat = np.random.choice(len(c.points), size=num_adj, replace=True)
    step_age = np.random.choice(len(c.age_layers[:65]), size=num_adj, replace=True)

    for i in range(num_adj):
        from_state, to_state = c.points[step_mat[i]][0], c.points[step_mat[i]][1]
        step_param = np.mean(matrix[:, from_state, to_state]) * step_size
        new_matrix[step_age[i], from_state, to_state] += np.random.uniform(
            low=-step_param, high=step_param
        )

    new_matrix = constrain_matrix(new_matrix)
    new_matrix = add_acm(new_matrix)
    new_matrix = add_csd(new_matrix)
    new_matrix = row_normalize(new_matrix)

    return new_matrix


import numpy as np
import configs as c


def run_markov(matrix, starting_age=20, max_age=100):
    current_age = starting_age
    stage = 0
    month_pop, pop_log = c.starting_pop, c.starting_pop
    inc_log = np.zeros(pop_log.shape)  # to track new incidences in each state
    age_layer = 0
    while current_age < max_age:
        mat = matrix[age_layer].T
        inflow_mat = np.tril(mat, k=-1)
        # Get incidence of current month's transitions
        month_inc = np.matmul(inflow_mat, month_pop)  # (14,14)(14,1)-->(14,1)
        # Actually make transitions
        month_pop = np.matmul(mat, month_pop)
        # Add to log
        inc_log = np.concatenate((inc_log, month_inc), axis=1)
        pop_log = np.concatenate((pop_log, month_pop), axis=1)
        stage += 1
        if stage % 12 == 0:
            current_age += 1
            if stage % 60 == 0:
                age_layer += 1  # Update age layer annually

    inc_log = inc_log[:, 1:]  # make (14,960)
    inc_rate = inc_log.copy()  # make (14,960)
    pop_log = pop_log[:, 1:]  # make (14,960)

    dead_factor = np.divide(
        c.N, c.N - pop_log[9:, :].sum(axis=0)
    )  # inc and prev denominator is out of living only
    prevalence = np.zeros(pop_log.shape)  # (14,80)

    for state in range(14):
        inc_rate[state, :] = np.multiply(inc_rate[state, :], dead_factor)
        prevalence[state, :] = np.multiply(pop_log[state, :], dead_factor)

    inc_rate = inc_rate.reshape(len(c.health_states), len(c.age_layers), 12).sum(
        axis=2
    )  # getting annual incidence (rate per 100k)
    inc_log = inc_log.reshape(len(c.health_states), len(c.age_layers), 12).sum(
        axis=2
    )  # getting inc unadjusted
    prevalence = prevalence.reshape(len(c.health_states), len(c.age_layers), 12).mean(
        axis=2
    )  # getting mean annual prevalence

    return inc_rate, prevalence, pop_log, inc_log


def simulated_annealing(
    n_iterations, step_size, start_tmat=None, n_adj=7, verbose=False
):

    if start_tmat is None:
        start_pmat = initialize_params()
        start_pmat, start_tmat = create_matrix(start_pmat)

    best_t = np.copy(start_tmat)
    best_log = m.run_markov_new(best_t)
    best_eval = gof.objective(best_log, 1)  # evaluate the initial point
    curr_t, curr_eval = best_t, best_eval  # current working solution
    ticker = 0

    with tqdm(
        total=n_iterations, desc="Simulated annealing progress", unit="iteration"
    ) as pbar:
        for i in range(n_iterations):
            # if ticker >= 25000: break

            # Run model
            candidate_t = np.copy(curr_t)
            candidate_t = step(candidate_t, step_size, n_adj)
            candidate_log = m.run_markov_new(candidate_t)
            candidate_eval = gof.objective(candidate_log, i)  # Evaluate candidate point

            # Update "best" if better than candidate
            if candidate_eval < best_eval:
                ticker = 0
                best_t, best_eval = np.copy(candidate_t), np.copy(candidate_eval)
                best_log = m.run_markov_new(best_t)

            else:
                ticker += 1

            # t = 10 / float(i+1)  # calculate temperature for current epoch
            t = 1 / (1 + np.log(i + 1))

            # Progress report
            if verbose and i % 1000 == 0:
                inc_log = best_log[3]
                total_dxd = np.sum(inc_log[6:9, :]) / c.N
                print(
                    i,
                    ": ",
                    best_eval,
                    "   CRC: ",
                    round(total_dxd, 5),
                    "   tick:",
                    ticker,
                )
                if i % 10000 == 0:
                    transition_probs = p.extract_transition_probs(
                        best_t, c.health_states, c.desired_transitions
                    )
                    print(f"Progress report, i = {i}")
                    p.print_trans_probs(transition_probs)

            # Check if we should update "curr"
            diff = (
                candidate_eval - curr_eval
            )  # difference between candidate and current point evaluation
            metropolis = np.exp(-diff / t)  # calculate metropolis acceptance criterion
            if (
                diff < 0 or np.random.random() < metropolis
            ):  # check if we should keep the new point
                curr_t, curr_eval = np.copy(candidate_t), np.copy(
                    candidate_eval
                )  # store the new current point
                ticker = 0

            pbar.update(1)

    print(best_eval)
    return best_t
