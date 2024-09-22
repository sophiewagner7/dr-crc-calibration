import numpy as np
from csaps import csaps
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
    params = np.zeros((len(c.points),4))
    params[0] = [0.005, 0.001, 50, 0.1] # Healthy to LR    
    params[1] = [0.015, 0.001, 50, 0.1] # LR to HR
    params[2] = [0.05, 0.001, 50, 0.1] # HR to uLoc
    params[3] = [0.45, 0.001, 50, 0.1] # uLoc to uReg
    params[4] = [0.50, 0.001, 50, 0.1] # uReg to uDis
    params[5] = [0.20, 0.001, 50, 0.1] # uLoc to dLoc
    params[6] = [0.60, 0.001, 50, 0.1] # uReg to dReg
    params[7] = [0.90, 0.001, 50, 0.1] # uDis to dDis
    return params
    

def create_matrix(params=None):
    if params is None:
        params = initialize_params()

    matrix = np.zeros((len(c.age_layers), len(c.health_states), len(c.health_states)))
    for idx, (from_state, to_state) in enumerate(c.points):
        matrix[:, from_state, to_state] = func.probtoprob(params[idx,0]) # monthly

    matrix = add_acm(matrix)  # ACM
    matrix = add_csd(matrix)  # CSD
    matrix = constrain_matrix(matrix)  # constrain
    matrix = row_normalize(matrix)  # normalize 
    
    return params, matrix


def constrain_matrix(matrix):
    matrix = np.clip(matrix,0.0,0.6)
   
    # Progression Block
    matrix[:,0,1] = np.maximum(0.000001, matrix[:,0,1])  # not below 0
    matrix[:,1,2] = np.maximum(matrix[:,0,1], matrix[:,1,2]) # HR to LR > healthy to uLoc
    matrix[:,2,3] = np.maximum(matrix[:,1,2], matrix[:,2,3])
    
    # Detection Block
    matrix[:,3,6] = np.maximum(0, matrix[:,3,6])  # not below 0
    matrix[:,4,7] = np.maximum(matrix[:,3,6], matrix[:,4,7])  # P[d_reg] > P[d_loc]
    matrix[:,5,8] = np.maximum(matrix[:,4,7], matrix[:,5,8])  # P[d_dis] > P[d_reg]
    
    return matrix


def add_acm(matrix):
    matrix[:,0,10] = c.acm_rate  # Healthy to ACM
    matrix[:,1:3,12] = c.acm_rate[:, np.newaxis] # Polyp to ACM
    matrix[:,3:6,13] = c.acm_rate[:, np.newaxis] # Undiagnosed to ACM
    matrix[:,6:9,11] = c.acm_rate[:, np.newaxis]  # Cancer to ACM
    matrix[:,9,9] = 1  # Stay in CSD
    matrix[:,10,10] = 1  # Stay in ACM
    matrix[:,11,11] = 1  # Stay in Cancer ACM
    matrix[:,12,12] = 1  # Stay in Polyp ACM
    matrix[:,13,13] = 1  # Stay in uCRC ACM

    return matrix 


def add_csd(matrix):
    matrix[:, 6, 9] = c.csd_rate[:, 0]
    matrix[:, 7, 9] = c.csd_rate[:, 1]
    matrix[:, 8, 9] = c.csd_rate[:, 2]
    return matrix


def interp_matrix(matrix):
    
    age_mids = np.arange(0, 65)  # Age midpoints for interpolation up to 85
    age_mids = np.append(age_mids, 100) # Add 100 to anchor the extrapolation
    all_ages = c.age_layers # Full range of age_layers
    interp_points = c.points # Exclude first transition from interpolation
  
    for (from_state, to_state) in interp_points: 
        """ Interpolate based on parameters up to age 85
            Anchor based on the mean of these parameter values
        """
        under_85 = matrix[:65, from_state, to_state]  # Extract transition probs up to age 85
        anchored = np.append(under_85, np.mean(under_85))  # Append mean to age 100
        tsmooth_spline = csaps(age_mids, anchored, smooth=.01)  
        matrix[:,from_state,to_state] = tsmooth_spline(all_ages).clip(0.000001, 0.4)

    return matrix


def step(matrix, step_size, num_adj=21):
    new_matrix = np.copy(matrix)
    step_mat = np.random.choice(len(c.points), size=num_adj, replace=True)
    step_age = np.random.choice(len(c.age_layers), size = num_adj, replace=True)

    for i in range(num_adj):
        new_matrix[step_age[i], c.points[step_mat[i]][0], c.points[step_mat[i]][1]] += np.random.uniform(low=-step_size, high=step_size)
   
    new_matrix = constrain_matrix(new_matrix)
    new_matrix = interp_matrix(new_matrix)
    new_matrix = add_acm(new_matrix)
    new_matrix = add_csd(new_matrix)
    new_matrix = row_normalize(new_matrix)

    return new_matrix


def simulated_annealing(n_iterations, step_size, start_tmat=None, n_adj=7, verbose=False):

    if start_tmat is None:
        start_pmat = initialize_params()
        start_pmat, start_tmat = create_matrix(start_pmat)
    
    best_t = np.copy(start_tmat)
    best_log = m.run_markov_new(best_t)
    best_eval = gof.objective(best_log,1)  # evaluate the initial point
    curr_t, curr_eval = best_t, best_eval  # current working solution
    ticker = 0
    
    for i in range(n_iterations):  
        if ticker >= 10000: break

        # Run model 
        candidate_t = np.copy(curr_t)
        candidate_t = step(candidate_t, step_size, n_adj)
        candidate_log = m.run_markov_new(candidate_t)
        candidate_eval = gof.objective(candidate_log,i)  # Evaluate candidate point
 
        # Update "best" if better than candidate
        if candidate_eval < best_eval:
            ticker = 0
            best_t, best_eval = np.copy(candidate_t), np.copy(candidate_eval) 
            best_log = m.run_markov_new(best_t)

        else: 
            ticker += 1

        # t = 10 / float(i+1)  # calculate temperature for current epoch
        t = 1 /(1+np.log(i+1))

        # Progress report
        if verbose and i%500==0:
            inc_log=best_log[3]
            total_dxd=np.sum(inc_log[6:9,:])/c.N 
            print(i, ": ", best_eval,"   CRC: ", round(total_dxd,5))
            if i%5000==0:
                transition_probs = p.extract_transition_probs(
                    best_t, c.health_states, c.desired_transitions
                )
                print(f"Progress report, i = {i}")
                p.print_trans_probs(transition_probs)
           
        # Check if we should update "curr"
        diff = candidate_eval - curr_eval  # difference between candidate and current point evaluation
        metropolis = np.exp(-diff / t)  # calculate metropolis acceptance criterion        
        if diff < 0 or np.random.random() < metropolis:  # check if we should keep the new point
            curr_t, curr_eval = np.copy(candidate_t), np.copy(candidate_eval) # store the new current point
        
    print(best_eval)
    return best_t
