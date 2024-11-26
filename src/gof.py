import numpy as np
import configs as c


# Calculate score based on difference between model outputs and targets
def objective(log, i):
    inc, _, _, inc_log = log
    score1, score2, score = 0, 0, 0

    # Yearly incidence penalty (ages 20-84)
    score1 += np.square(inc[6, :65] - c.seer_inc["Local Rate"]).sum()
    score1 += np.square(inc[7, :65] - c.seer_inc["Regional Rate"]).sum()
    score1 += np.square(inc[8, :65] - c.seer_inc["Distant Rate"]).sum()

    # Polyp prevalence penalty (pooled)
    score2 += (1 / np.sqrt(35656)) * np.square(
        inc_log[12, :].sum() - c.N * c.polyp_targets[1]
    )
    score2 += (1 / np.sqrt(31434)) * np.square(
        inc_log[13, :].sum() - c.N * c.polyp_targets[0]
    )  # uCRC
    score2 *= 0.5
    score = score1 + score2

    if i % 5000 == 0:
        print(f"SEER contribution:{round(score1,0)}, ({round(score1/score*100,0)}%)")
        print(f"Polyp contribution:{round(score2,0)}, ({round(score2/score*100,0)}%)")

    return score
