import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csaps import csaps #https://csaps.readthedocs.io/en/latest/
import common_functions as func
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import seaborn as sns

# Function to extract transition probabilities
def extract_transition_probs(tmat, states, transitions):
    transition_probs = {}
    for from_state, to_state in transitions:
        from_idx = states[from_state]
        to_idx = states[to_state]
        params = tmat[:, from_idx, to_idx]
        transition_probs[f"{from_state} to {to_state}"] = params
    return transition_probs

# Print the transition probabilities in a readable format
def print_trans_probs(transition_probs):
    print("Monthly transition probs")
    for transition, prob in transition_probs.items():
        print(f"{transition}: {prob[30]:.5f}")

    print("\nAnnual transition probs")
    for transition, prob in transition_probs.items():
        annual_prob = func.probtoprob(prob[30], 12, 1)
        print(f"{transition}: {annual_prob:.5f}")
        


def plot_tps(curr_tmat):
    plt.plot(func.probtoprob(curr_tmat[:,0,1],12,1), label = "Healthy to LR")
    plt.plot(func.probtoprob(curr_tmat[:,3,6],12,1), label = "uLoc to dLoc")
    plt.plot(func.probtoprob(curr_tmat[:,4,7],12,1), label = "uReg to dReg")
    plt.plot(func.probtoprob(curr_tmat[:,5,8],12,1), label = "uDis to dDis")
    plt.legend()
    plt.show()

    plt.plot(func.probtoprob(curr_tmat[:,0,1],12,1), label = "Healthy to LR")
    plt.legend()
    plt.show()

    plt.plot(func.probtoprob(curr_tmat[:,1,2],12,1), label = "LR to HR")
    plt.plot(func.probtoprob(curr_tmat[:,2,3],12,1), label = "HR to uLoc")
    plt.plot(func.probtoprob(curr_tmat[:,3,4],12,1), label = "uLoc to uReg")
    plt.plot(func.probtoprob(curr_tmat[:,4,5],12,1), label = "uReg to uDis")
    plt.legend()
    plt.show()


### Plotting
def plot_vs_seer(curr_log, seer_inc):
    inc_adj, _,_,_ = curr_log
    x_values = np.linspace(20,99,80)

    plt.plot(seer_inc['Age'], seer_inc['Local Rate'], label = 'Local (SEER)', color='b',linestyle="dotted")
    plt.plot(seer_inc['Age'], seer_inc['Regional Rate'], label = 'Regional (SEER)', color='r', linestyle="dotted")
    plt.plot(seer_inc['Age'], seer_inc['Distant Rate'],  label='Distant (SEER)', color='g',linestyle="dotted")
    plt.plot(x_values, inc_adj[6,:], label='Local (Model)', color='b')
    plt.plot(x_values, inc_adj[7,:],  label='Regional (Model)', color='r')
    plt.plot(x_values, inc_adj[8,:],  label='Distant (Model)', color='g')
    plt.legend()
    plt.title('Incidence of Local, Regional, and Distant States')
    plt.xlabel('Time Point / Age Group')
    plt.ylabel('incidence')
    plt.show()

    plt.plot(seer_inc['Age'], seer_inc['Local Rate'].cumsum(), label = 'Local (SEER)', color='b',linestyle="dotted")
    plt.plot(seer_inc['Age'], seer_inc['Regional Rate'].cumsum(), label = 'Regional (SEER)', color='r',linestyle="dotted")
    plt.plot(seer_inc['Age'], seer_inc['Distant Rate'].cumsum(), label = 'Distant (SEER)', color='g',linestyle="dotted")
    plt.plot(x_values, inc_adj[6,:].cumsum(), label='Local (Model)', color='b')
    plt.plot(x_values, inc_adj[7,:].cumsum(), label='Regional (Model)', color='r')
    plt.plot(x_values, inc_adj[8,:].cumsum(), label='Distant (Model)', color='g')
    plt.legend()
    plt.title('Cumulative Incidence of Local, Regional, and Distant States')
    plt.xlabel('Time Point / Age Group')
    plt.ylabel('incidence')
    plt.show()