import numpy as np


def rmse(pred, true):
    return np.sqrt(np.mean(np.square(pred - true)))


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def weighted_mae(pred, true, wts):
    S = np.sum(wts)
    return np.mean(wts @ np.abs(pred - true) / S)

