import numpy as np

from .base import fitting_ge_and_plot


def get_rotate_angle(Ig, Qg, Ie, Qe):
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    theta = -np.arctan2((ye - yg), (xe - xg))
    return {"theta": theta}


def fit_ge_by_center(signals, plot=True):
    return fitting_ge_and_plot(signals, get_rotate_angle, plot)
