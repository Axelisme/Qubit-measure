import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def init_show(X, x_label, y_label, title=None):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    curve = ax.plot(X, np.zeros_like(X))[0]
    dh = display(fig, display_id=True)
    return fig, ax, dh, curve


def update_show(fig, ax, dh, curve, X, Y):
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
    ax.pcolormesh(X, Y, np.zeros((len(Y), len(X))))
    dh = display(fig, display_id=True)
    return fig, ax, dh


def update_show2d(fig, ax, dh, X, Y, Z):
    ax.pcolormesh(X, Y, Z)
    dh.update(fig)


def clear_show(dh):
    clear_output()
