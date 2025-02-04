import numpy as np

from .base import fitting_and_plot


def get_rotate_angle(Ig, Qg, Ie, Qe):
    # use logictic regression to get the rotate angle
    from sklearn.linear_model import LogisticRegression

    X = np.vstack((np.column_stack((Ig, Qg)), np.column_stack((Ie, Qe))))
    Y = np.hstack((np.zeros(len(Ig)), np.ones(len(Ie))))

    model = LogisticRegression(
        penalty="l1", C=100, solver="liblinear", max_iter=1000
    ).fit(X, Y)

    # ax + by + c = 0
    a, b = model.coef_[0]
    c = model.intercept_[0]

    theta = -np.arctan2(b, a)
    threshold = -c / np.sqrt(a**2 + b**2)

    return {"theta": theta, "threshold": threshold}


def fit_by_regression(Is, Qs, plot=True):
    return fitting_and_plot(Is, Qs, get_rotate_angle, plot)
