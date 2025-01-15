import numpy as np
import configs as c


# Calculate score based on difference between model outputs and targets
def objective(log, i):
    inc, _, inc_log, _ = log
    score1, score2, score = 0, 0, 0

    # Yearly incidence penalty (ages 20-84)
    score1 += np.square(inc[6, :] - c.seer_inc["Local Rate"]).sum() 
    score1 += np.square(inc[7, :] - c.seer_inc["Regional Rate"]).sum()
    score1 += np.square(inc[8, :] - c.seer_inc["Distant Rate"]).sum()

    # Polyp prevalence penalty (pooled)
    score2 += (1 / np.sqrt(35656)) * np.square(inc_log[12, :].sum() - c.N * c.polyp_targets[1])
    # score2 = score2/2 if obj == "pol" else score2/100  # weight of polyp loss contribution

    score = score1 + score2

    return score