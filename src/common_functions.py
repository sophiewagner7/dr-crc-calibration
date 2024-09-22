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


# Get tp from params
def get_tp(params, age_layers):
    """Calculate transition probabilities for a vector of ages using logistic function.

    Args:
        params (array): base, increase, offset, spread
        age_layers (array-like): array of age indices

    Returns:
        numpy array: annual transition probabilities for given transition across age_layers
    """
    base, increase, offset, spread = params
    # Convert age_layers to a NumPy array if it's not already
    age_layers = np.asarray(age_layers)
    tp = base + increase * (1 / (1 + np.exp(-spread * (age_layers - offset))))
    return tp


# Get tp from params
def get_tp_linear(params, age_layers):
    """Calculate transition probabilities for a vector of ages using linear function.

    Args:
        params (array): intercept, slope
        age_layers (array-like): array of age indices

    Returns:
        numpy array: annual transition probabilities for given transition across age_layers
    """
    int, slope, _, _ = params
    # Convert age_layers to a NumPy array if it's not already
    age_layers = np.asarray(age_layers)
    tp = int + slope * age_layers
    return tp


# Get tp from params
def get_tp_logis(params, age_layers):
    """Calculate transition probabilities for a vector of ages using logistic function.

    Args:
        params (array): base, increase, offset, spread
        age_layers (array-like): array of age indices

    Returns:
        numpy array: annual transition probabilities for given transition across age_layers
    """
    base, increase, offset, spread = params
    # Convert age_layers to a NumPy array if it's not already
    age_layers = np.asarray(age_layers)
    tp = base + increase * (1 / (1 + np.exp(-spread * (age_layers - offset))))
    return tp
