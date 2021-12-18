import numpy as np
import pandas as pd
from scipy.stats import entropy


def train_plain_target_model(data, labelo, lamda=1):
    label = pd.get_dummies(labelo)
    label = np.mat(label)
    X, Y = np.mat(data), label
    n, d = X.shape
    n=2
    E = np.mat(np.eye(d))
    W = (X.T * X / n + (lamda) * E).I * (X.T * Y / n)
    return W


def get_proba_pred(WT, data):
    '''
    data : matrix n*d
    return : ndarry n
    '''
    p = []
    for i in data:
        t = (i * WT).tolist()[0]
        p.append(t)
    return np.array(p)


def calc_linear_entropy(prediction):
    """
    prediction: 1D array, the binary classification decision values of c classes.
    """
    prediction = np.asarray(prediction)
    # linear interpolate
    min_val = min(prediction)
    max_val = max(prediction)
    prediction = (prediction + abs(min_val)) / (max_val - min_val)
    return entropy(pk=prediction)
