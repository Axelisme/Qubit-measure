import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display


def init_show(X, x_label, y_label, title=None):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    curve = ax.plot(X, np.zeros_like(X), "-o")[0]
    dh = display(fig, display_id=True)
    return fig, ax, dh, curve


def update_show(fig, ax, dh, curve, Y):
    curve.set_ydata(Y)
    ax.relim()
    ax.autoscale(axis="y")
    dh.update(fig)


def init_show2d(X, Y, x_label, y_label, title=None):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    im = ax.imshow(
        np.zeros((len(Y), len(X))),
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[X[0], X[-1], Y[0], Y[-1]],
    )
    dh = display(fig, display_id=True)
    return fig, ax, dh, im


def update_show2d(fig, ax, dh, im, Z):
    im.set_data(Z)
    im.autoscale()
    dh.update(fig)


def clear_show():
    clear_output()
