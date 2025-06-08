import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .general import figsize


def mux_reset_analyze(xs, ys, signal2D, xlabel=None, ylabel=None, smooth=1):
    signal2D = gaussian_filter(signal2D, smooth)

    amp2D = np.abs(signal2D - np.mean(signal2D))

    x_max_id = np.argmax(np.max(amp2D, axis=1))
    y_max_id = np.argmax(np.max(amp2D, axis=0))
    x_max = xs[x_max_id]
    y_max = ys[y_max_id]

    plt.figure(figsize=figsize)
    plt.imshow(
        amp2D.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
    )
    plt.scatter(x_max, y_max, color="r", s=2)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.colorbar(label="Signal (a.u.)")
    plt.show()

    return x_max, y_max
