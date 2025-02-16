import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


def init_show1d(X, x_label, y_label, title=None, **kwargs):
    fig, ax = plt.subplots()
    fig.tight_layout(pad=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("marker", ".")
    curve = ax.plot(X, np.zeros_like(X), **kwargs)[0]
    dh = display(fig, display_id=True)
    return fig, ax, dh, curve


def update_show1d(fig, ax, dh, curve, Y, X=None):
    if X is not None:
        curve.set_xdata(X)
    curve.set_ydata(Y)
    ax.relim()
    if X is None:
        ax.autoscale(axis="y")
    else:
        ax.autoscale_view()
    dh.update(fig)


def init_show2d(X, Y, x_label, y_label, title=None, **kwargs):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("interpolation", "none")
    kwargs.setdefault("aspect", "auto")
    im = ax.imshow(
        np.zeros((len(Y), len(X))),
        extent=[X[0], X[-1], Y[0], Y[-1]],
        **kwargs,
    )
    fig.tight_layout()
    dh = display(fig, display_id=True)
    return fig, ax, dh, im


def update_show2d(fig, ax, dh, im, Z, XY: tuple = None):
    if XY is not None:
        X, Y = XY
        im.set_extent([X[0], X[-1], Y[0], Y[-1]])
    im.set_data(Z.T)
    im.autoscale()
    dh.update(fig)


def init_show(*ticks, x_label, y_label, title=None, **kwargs):
    if len(ticks) == 1:
        return init_show1d(ticks[0], x_label, y_label, title, **kwargs)
    elif len(ticks) == 2:
        return init_show2d(ticks[0], ticks[1], x_label, y_label, title, **kwargs)
    else:
        raise ValueError("Invalid number of ticks")


def update_show(fig, ax, dh, contain, data, ticks=None):
    if len(data.shape) == 1:
        update_show1d(fig, ax, dh, contain, data, ticks)
    elif len(data.shape) == 2:
        update_show2d(fig, ax, dh, contain, data, ticks)
    else:
        raise ValueError("Unkown data shape")


def close_show(fig, dh):
    plt.close(fig)
