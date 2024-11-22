import numpy as np

from .base import fitting_and_plot


def get_rotate_angle(Ig, Qg, Ie, Qe):
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    theta = -np.arctan2((ye - yg), (xe - xg))
    return {"theta": theta}


def fit_by_center(Is, Qs, plot=True):
    return fitting_and_plot(Is, Qs, get_rotate_angle, plot)
