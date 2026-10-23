from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
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
        vmin: float | None = None,
        vmax: float | None = None,
        *,
        flip: bool = False,
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
            np.full((1, 2), np.nan),
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
        figure = ax.get_figure()
        if figure is None or ax not in figure.axes or im not in ax.images:
            raise RuntimeError(
                "Live plot image was removed from its axes or figure; "
                "do not clear or remove template artists before updating."
            )

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
    """Render monotonic sample-centered coordinates in either scan direction.

    Uniform and nonuniform axes use the same nearest-sample rendering rule.
    Updates accept signals shaped (len(xs), len(ys)) without changing inputs.
    Each axis follows its input coordinate order; flip swaps the displayed axes.
    """

    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        *,
        flip: bool = False,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.flip = flip
        self.vmin = vmin
        self.vmax = vmax

        self.im: NonUniformImage | None = None

    def init_ax(self, ax: Axes, *, cmap: str | Colormap = "RdBu_r") -> None:
        """Initialize labeled axes with a nearest-sample image in the given cmap.

        cmap is a Matplotlib Colormap or registered name; invalid names raise
        ValueError. The default preserves the liveplot diverging palette.
        """
        if self.flip:
            ax.set_xlabel(self.ylabel)
            ax.set_ylabel(self.xlabel)
        else:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)

        if self.title is not None:
            ax.set_title(self.title)

        self.im = NonUniformImage(ax, cmap=cmap, interpolation="nearest")
        self.im.set_extent((0, 1, 0, 1))
        self.im.set_data([0, 1], [0, 1], np.full((2, 2), np.nan))
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
        figure = ax.get_figure()
        if figure is None or ax not in figure.axes or im not in ax.images:
            raise RuntimeError(
                "Live plot image was removed from its axes or figure; "
                "do not clear or remove template artists before updating."
            )

        dx = 0.5 * (xs[-1] - xs[0]) / max(1, len(xs) - 1)
        dy = 0.5 * (ys[-1] - ys[0]) / max(1, len(ys) - 1)
        if self.flip:
            im.set_extent((ys[0] - dy, ys[-1] + dy, xs[0] - dx, xs[-1] + dx))
        else:
            im.set_extent((xs[0] - dx, xs[-1] + dx, ys[0] - dy, ys[-1] + dy))

        # NonUniformImage searches ascending centers. Keep the scan-order axis
        # limits set above, but reverse centers and their data together.
        if xs[-1] < xs[0]:
            xs, signals = xs[::-1], signals[::-1, :]
        if ys[-1] < ys[0]:
            ys, signals = ys[::-1], signals[:, ::-1]
        if self.flip:
            im.set_data(ys, xs, signals)
        else:
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
