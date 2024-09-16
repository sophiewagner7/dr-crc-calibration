import numpy as np


# Common functions
def probtoprob(rate, a=1, b=12):
    return 1-(1-rate)**(a/b)

# Get alpha from rate
def get_alpha(rate, a=1, b=12):
    monthly_rate = probtoprob(rate, a, b)
    alpha = -np.log((0.5 - monthly_rate) / monthly_rate)
    #alpha = monthly_rate
    return alpha

# Get transition probability from alpha and beta
def get_tp(params, age_layer=6):
    # U,L control upper and lower asymptotes
    # a = slope
    # b = intercept
    a, b, U, L = params
    tp = L + ((U-L) / (1+np.exp(-((b)+age_layer*a))))
    #tp = a + (b*age_layer)
    return tp

