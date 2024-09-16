import numpy as np


# Common functions
def probtoprob(rate, a=1, b=12):
    return 1 - (1 - rate) ** (a / b)


# Get alpha from rate
def get_alpha(rate, a=1, b=12):
    monthly_rate = probtoprob(rate, a, b)
    alpha = -np.log((0.5 - monthly_rate) / monthly_rate)
    # alpha = monthly_rate
    return alpha


# Get transition probability from
def get_tp(params, age_layer=30):
    """_summary_

    Args:
        params (array): base, increase, offset, spread_description_
        age_layer (int, optional): age index. Defaults to 30.

    Returns:
        double: annual transition probability for given transition at age_layer
    """
    base, increase, offset, spread = params
    tp = base + increase * (1 / (1 + np.exp(-spread * (age_layer - offset))))
    return tp
