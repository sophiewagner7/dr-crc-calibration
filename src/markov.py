import numpy as np
import configs as c


# def run_markov_old(tmat, max_age=100, starting_age=c.starting_age):
#     current_age = starting_age
#     i = 0
#     pop, pop_log = (
#         c.starting_pop.copy(),
#         c.starting_pop.copy(),
#     )  # current population in each state (14,1)
#     inc_log = np.zeros(pop.shape)  # to track new incidences in each state
#     age_layer = 0
#     mat = calculate_transition_probs(tmat, age_layer)[2]

#     while current_age < max_age:
#         yr_inc = np.zeros(pop.shape)  # Initialize yearly incidence array

#         for state in range(len(pop)):

#             if state >= 9:  # Terminal states, no outflow
#                 continue

#             acm_state = c.acm_states[state]  # Assign ACM state

#             # Handling ACM
#             acm_deaths = pop[state] * c.acm_rate[current_age]
#             yr_inc[acm_state] += acm_deaths
#             pop[state] -= acm_deaths

#             # Handling CSD
#             if (
#                 c.health_states["d_CRC_loc"] <= state <= c.health_states["d_CRC_dis"]
#             ):  # if in detected cancer state
#                 if state == 6:
#                     csd_deaths = pop[state] * c.csd_rate["Local"][age_layer]
#                 if state == 7:
#                     csd_deaths = pop[state] * c.csd_rate["Regional"][age_layer]
#                 if state == 8:
#                     csd_deaths = pop[state] * c.csd_rate["Distant"][age_layer]
#                 yr_inc[9] += csd_deaths
#                 pop[state] -= csd_deaths

#             # Handling detections
#             if (
#                 c.health_states["u_CRC_loc"] <= state <= c.health_states["u_CRC_dis"]
#             ):  # if in undetected cancer state
#                 detection = pop[state] * mat[state, state + 3]
#                 yr_inc[state + 3] += detection
#                 pop[state] -= detection

#             # Handling progression
#             if (
#                 state <= c.health_states["u_CRC_reg"]
#             ):  # Only states < 5 have progressions
#                 progression = pop[state] * mat[state, state + 1]
#                 yr_inc[state + 1] += progression
#                 pop[state] -= progression

#         pop += yr_inc  # Update population after all transitions
#         pop_log = np.concatenate((pop_log, pop), axis=1)  # will be (14,80*12=960+1)
#         inc_log = np.concatenate(
#             (inc_log, yr_inc), axis=1
#         )  # will be (14, 80*12=960+1) by end

#         i += 1
#         if i % 12 == 0:
#             current_age += 1
#             if current_age in [
#                 25,
#                 30,
#                 35,
#                 40,
#                 45,
#                 50,
#                 55,
#                 60,
#                 65,
#                 70,
#                 75,
#                 80,
#                 85,
#                 90,
#                 95,
#             ]:
#                 age_layer += 1
#                 mat = calculate_transition_probs(tmat, age_layer)[2]  # update mat

#     inc_log = inc_log[:, 1:]  # make (14,960)
#     inc_rate = inc_log.copy()  # make (14,960)
#     pop_log = pop_log[:, 1:]  # make (14,960)

#     dead_factor = np.divide(
#         c.N, c.N - pop_log[9:, :].sum(axis=0)
#     )  # inc and prev denominator is out of living only
#     prevalence = np.zeros(pop_log.shape)  # (14,80)

#     for state in range(14):
#         inc_rate[state, :] = np.multiply(inc_rate[state, :], dead_factor)
#         prevalence[state, :] = np.multiply(pop_log[state, :], dead_factor)

#     inc_rate = inc_rate.reshape(len(c.health_states), 80, 12).sum(
#         axis=2
#     )  # getting annual incidence (rate per 100k)
#     inc_log = inc_log.reshape(len(c.health_states), 80, 12).sum(
#         axis=2
#     )  # getting inc unadjusted
#     prevalence = prevalence.reshape(len(c.health_states), 80, 12).mean(
#         axis=2
#     )  # getting mean annual prevalence

#     return inc_rate, prevalence, pop_log, inc_log


def run_markov_new(matrix, starting_age=20, max_age=100):
    current_age = starting_age
    stage = 0
    month_pop, pop_log = c.starting_pop, c.starting_pop
    inc_log = np.zeros(pop_log.shape)  # to track new incidences in each state
    age_layer = 0
    mat = matrix[age_layer]
    mat = mat.T
    inflow_mat = np.tril(mat, k=-1)
    while current_age < max_age:
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
            age_layer = min(
                (current_age - starting_age) // 1, len(c.ages) - 1
            )  # Update age layer annually
            mat = matrix[age_layer].T
            inflow_mat = np.tril(mat, k=-1)

            # if current_age in [25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]:
            #    age_layer += 1
            # If we shift age_layer, need to update transition probs
            # mat = matrix[age_layer].T
            # inflow_mat = np.tril(mat, k=-1)

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

    inc_rate = inc_rate.reshape(len(c.health_states), 80, 12).sum(
        axis=2
    )  # getting annual incidence (rate per 100k)
    inc_log = inc_log.reshape(len(c.health_states), 80, 12).sum(
        axis=2
    )  # getting inc unadjusted
    prevalence = prevalence.reshape(len(c.health_states), 80, 12).mean(
        axis=2
    )  # getting mean annual prevalence

    return inc_rate, prevalence, pop_log, inc_log
