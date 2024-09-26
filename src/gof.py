import numpy as np
import configs as c


# Calculate score based on difference between model outputs and targets
def objective(log, i):
    inc, _, _, inc_log = log
    score = 0

    if c.model_version == "US":
        # Yearly incidence penalty (20-84)
        score += np.square(inc[6, :65] - c.seer_inc["Local Rate"]).sum()
        score += np.square(inc[7, :65] - c.seer_inc["Regional Rate"]).sum()
        score += np.square(inc[8, :65] - c.seer_inc["Distant Rate"]).sum()

    elif c.model_version == "DR":
        if c.dr_stage_penalty == "Pooled":
            model_total_inc = np.sum(inc[6:9, :], axis=0)
            model_stages = [
                np.sum((inc[stage]) / np.sum(model_total_inc)) * 100
                for stage in [6, 7, 8]
            ]
            # Yearly incidence penalty
            score += np.square(model_total_inc[:65] - c.dr_inc).sum()
            # Cumulative stage distribution
            score += np.square(model_stages - c.dr_stage_dist).sum() * 25

        elif c.dr_stage_penalty == "Yearly":
            score += np.square(inc[6, :65] - c.seer_inc["Local Rate"]).sum()
            score += np.square(inc[7, :65] - c.seer_inc["Regional Rate"]).sum()
            score += np.square(inc[8, :65] - c.seer_inc["Distant Rate"]).sum()

    # Polyp prevalence penalty (pooled)
    score += (1 / np.sqrt(35656)) * np.square(
        inc_log[12, :].sum() - c.N * c.polyp_targets[1]
    )  # polyps
    score += (1 / np.sqrt(31434)) * np.square(
        inc_log[13, :].sum() - c.N * c.polyp_targets[0]
    )  # uCRC

    return score
