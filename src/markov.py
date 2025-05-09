import numpy as np

import configs as c


def run_markov_new(matrix, starting_age=20, max_age=100):
    current_age = starting_age
    stage = 0
    month_pop, pop_log = c.starting_pop, c.starting_pop
    inc_log = np.zeros(pop_log.shape)  # to track new incidences in each state
    age_layer = 0
    while current_age < max_age:
        mat = matrix[age_layer].T
        inflow_mat = np.tril(mat, k=-1)
        month_inc = np.matmul(inflow_mat, month_pop)  # (14,14)(14,1)-->(14,1)
        month_pop = np.matmul(mat, month_pop)
        inc_log = np.concatenate((inc_log, month_inc), axis=1)
        pop_log = np.concatenate((pop_log, month_pop), axis=1)
        stage += 1

        # Increase stage and age_layer
        if stage % 12 == 0:
            current_age += 1
            age_layer += 1

    inc_log = inc_log[:, 1:]  # make (14,960)
    inc_rate = inc_log.copy()  # make (14,960)
    pop_log = pop_log[:, 1:]  # make (14,960)

     # inc and prev denominator is out of living only
    dead_factor = np.divide(
        c.N, c.N - pop_log[9:, :].sum(axis=0)
    ) 
    prevalence = np.zeros(pop_log.shape)
    
    for state in range(14):
        inc_rate[state, :] = np.multiply(inc_rate[state, :], dead_factor)
        prevalence[state, :] = np.multiply(pop_log[state, :], dead_factor)
    
    # getting annual incidence (rate per 100k)
    inc_rate = inc_rate.reshape(len(c.health_states), len(c.age_layers_1y)-1, 12).sum(
        axis=2
    )  
    # getting inc unadjusted
    inc_log = inc_log.reshape(len(c.health_states), len(c.age_layers_1y)-1, 12).sum(
        axis=2
    )  
    # getting mean annual prevalence
    prevalence = prevalence.reshape(
        len(c.health_states), len(c.age_layers_1y)-1, 12
    ).mean(
        axis=2
    )  
    
    return inc_rate, prevalence, pop_log, inc_log
