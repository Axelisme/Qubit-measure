from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage, AxesImage
from matplotlib.ticker import ScalarFormatter
import numpy as np

from .base import AbsSegment


class Plot2DSegment(AbsSegment):
    def __init__(
        self, xlabel: str, ylabel: str, title: Optional[str] = None, flip: bool = False
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.flip = flip

        self.im: Optional[AxesImage] = None

    def init_ax(self, ax: plt.Axes) -> None:
        if self.flip:
            ax.set_xlabel(self.ylabel)
            ax.set_ylabel(self.xlabel)
        else:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)

        if self.title is not None:
            ax.set_title(self.title)

        # 設定自動格式化刻度
        formatter = ScalarFormatter(useMathText=True)  # 用漂亮的 10^n 標記
        formatter.set_powerlimits((-2, 2))  # 超出此範圍就改用科學記號
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        self.im = ax.imshow(
            [[0, 1e-6]], aspect="auto", origin="lower", interpolation="nearest"
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

        dx = 0.5 * (xs[-1] - xs[0]) / (len(xs) - 1)
        dy = 0.5 * (ys[-1] - ys[0]) / (len(ys) - 1)
        if self.flip:
            self.im.set_extent([ys[0] - dy, ys[-1] + dy, xs[0] - dx, xs[-1] + dx])
            self.im.set_data(signals)
        else:
            self.im.set_extent([xs[0] - dx, xs[-1] + dx, ys[0] - dy, ys[-1] + dy])
            self.im.set_data(signals.T)

        self.im.autoscale()

        if title is not None:
            ax.set_title(title)

    def clear(self, ax: plt.Axes) -> None:
        ax.clear()
        self.im = None


class PlotNonUniform2DSegment(AbsSegment):
    def __init__(
        self, xlabel: str, ylabel: str, title: Optional[str] = None, flip: bool = False
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.flip = flip

        self.im: Optional[AxesImage] = None

    def init_ax(self, ax: plt.Axes) -> None:
        if self.flip:
            ax.set_xlabel(self.ylabel)
            ax.set_ylabel(self.xlabel)
        else:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)

        if self.title is not None:
            ax.set_title(self.title)

        self.im = NonUniformImage(ax, cmap="viridis", interpolation="nearest")
        ax.add_image(self.im)

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

        dx = 0.5 * (xs[-1] - xs[0]) / (len(xs) - 1)
        dy = 0.5 * (ys[-1] - ys[0]) / (len(ys) - 1)
        if self.flip:
            self.im.set_extent([ys[0] - dy, ys[-1] + dy, xs[0] - dx, xs[-1] + dx])
            self.im.set_data(ys, xs, signals)
        else:
            self.im.set_extent([xs[0] - dx, xs[-1] + dx, ys[0] - dy, ys[-1] + dy])
            self.im.set_data(xs, ys, signals.T)

        self.im.autoscale()

        if title is not None:
            ax.set_title(title)

    def clear(self, ax: plt.Axes) -> None:
        ax.clear()
        self.im = None
