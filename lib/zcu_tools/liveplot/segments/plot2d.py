from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .base import AbsSegment


class Plot2DSegment(AbsSegment):
    def __init__(self, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self.im: Optional[plt.AxesImage] = None

    def init_ax(self, ax: plt.Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)

        self.im = ax.imshow(
            [[0, 1]], aspect="auto", origin="lower", interpolation="nearest"
        )

    def update(
        self,
        ax: plt.Axes,
        xs: np.ndarray,
        ys: np.ndarray,
        signals: np.ndarray,
        title: Optional[str] = None,
    ) -> None:
        if self.im is None:
            raise RuntimeError("Image not initialized.")

        self.im.set_extent([xs[0], xs[-1], ys[0], ys[-1]])
        self.im.set_data(signals.T)
        self.im.autoscale()

        if title is not None:
            ax.set_title(title)

    def clear(self, ax: plt.Axes) -> None:
        ax.clear()
        self.im = None
