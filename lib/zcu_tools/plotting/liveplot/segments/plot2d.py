from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage, NonUniformImage
from matplotlib.ticker import ScalarFormatter
from numpy.typing import NDArray

from .base import AbsSegment


class Plot2DSegment(AbsSegment):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: str | None = None,
        flip: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.flip = flip
        self.vmin = vmin
        self.vmax = vmax

        self.im: AxesImage | None = None

    def init_ax(self, ax: Axes) -> None:
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
            [[0, 1e-6]],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=self.vmin,
            vmax=self.vmax,
            cmap="RdBu_r",
        )

    def update(
        self,
        ax: Axes,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
    ) -> None:
        if self.im is None:
            raise RuntimeError("Image not initialized.")
        im = self.im

        dx = 0.5 * (xs[-1] - xs[0]) / max(1, (len(xs) - 1))
        dy = 0.5 * (ys[-1] - ys[0]) / max(1, (len(ys) - 1))
        if self.flip:
            im.set_extent((ys[0] - dy, ys[-1] + dy, xs[0] - dx, xs[-1] + dx))
            im.set_data(signals)
        else:
            im.set_extent((xs[0] - dx, xs[-1] + dx, ys[0] - dy, ys[-1] + dy))
            im.set_data(signals.T)

        if self.vmin is not None or self.vmax is not None:
            im.set_clim(vmin=self.vmin, vmax=self.vmax)
        if self.vmin is None or self.vmax is None:
            im.autoscale()

        if title is not None:
            ax.set_title(title)

    def clear(self, ax: Axes) -> None:
        ax.clear()
        self.im = None


class PlotNonUniform2DSegment(AbsSegment):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: str | None = None,
        flip: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.flip = flip
        self.vmin = vmin
        self.vmax = vmax

        self.im: NonUniformImage | None = None

    def init_ax(self, ax: Axes) -> None:
        if self.flip:
            ax.set_xlabel(self.ylabel)
            ax.set_ylabel(self.xlabel)
        else:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)

        if self.title is not None:
            ax.set_title(self.title)

        self.im = NonUniformImage(ax, cmap="RdBu_r", interpolation="nearest")
        self.im.set_extent((0, 1, 0, 1))
        self.im.set_data([0, 1], [0, 1], [[0, 1e-6], [0, 1e-6]])
        self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
        ax.add_image(self.im)

    def update(
        self,
        ax: Axes,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
    ) -> None:
        if self.im is None:
            raise RuntimeError("Image not initialized.")
        im = self.im

        dx = 0.5 * (xs[-1] - xs[0]) / max(1, len(xs) - 1)
        dy = 0.5 * (ys[-1] - ys[0]) / max(1, len(ys) - 1)
        if self.flip:
            im.set_extent((ys[0] - dy, ys[-1] + dy, xs[0] - dx, xs[-1] + dx))
            im.set_data(ys, xs, signals)
        else:
            im.set_extent((xs[0] - dx, xs[-1] + dx, ys[0] - dy, ys[-1] + dy))
            im.set_data(xs, ys, signals.T)

        if self.vmin is not None or self.vmax is not None:
            im.set_clim(vmin=self.vmin, vmax=self.vmax)
        if self.vmin is None or self.vmax is None:
            im.autoscale()

        if title is not None:
            ax.set_title(title)

    def clear(self, ax: Axes) -> None:
        ax.clear()
        self.im = None
