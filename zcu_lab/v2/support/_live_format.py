"""Line, heatmap and scatter artists for the lab live tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage, NonUniformImage
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray


class LineStyle(TypedDict, total=False):
    """Matplotlib line properties used by the lab live tools.

    marker selects the point symbol; linestyle selects the connecting stroke;
    markersize is in points; alpha is opacity; color is a Matplotlib color;
    label names the legend entry. Omitted keys use Matplotlib defaults, except
    marker '.', linestyle '-' and markersize 5 supplied by LineFormat.
    """

    marker: str
    linestyle: str
    markersize: float
    alpha: float
    color: str
    label: str


class LineFormat:
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: str | None = None,
        num_lines: int = 1,
        line_kwargs: Sequence[LineStyle | None] | None = None,
        *,
        show_grid: bool = True,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.num_line = num_lines
        self.title = title
        self.show_grid = show_grid

        if line_kwargs is None:
            line_kwargs = [None] * self.num_line

        assert len(line_kwargs) == self.num_line, (
            f"Expected {self.num_line} line_kwargs, got {len(line_kwargs)}."
        )

        self.line_kwargs: list[LineStyle] = [
            {"marker": ".", "linestyle": "-", "markersize": 5, **(kwargs_i or {})}
            for kwargs_i in line_kwargs
        ]

        self.lines: list[Line2D] | None = None

    def init_ax(self, ax: Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        # Leave room for full MHz values on narrow scan-line panels.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if self.title is not None:
            ax.set_title(self.title)
        if self.show_grid:
            ax.grid()

        lines = []
        for kwargs in self.line_kwargs:
            assert kwargs is not None
            (line,) = ax.plot([], [], **kwargs)
            lines.append(line)
        self.lines = lines

        if any("label" in (kwargs or {}) for kwargs in self.line_kwargs):
            ax.legend()

    def update(
        self,
        ax: Axes,
        xs: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
    ) -> None:
        if self.lines is None:
            raise RuntimeError("Lines not initialized.")
        figure = ax.get_figure()
        if (
            figure is None
            or ax not in figure.axes
            or any(line not in ax.lines for line in self.lines)
        ):
            raise RuntimeError(
                "Live plot lines were removed from their axes or figure; "
                "do not clear or remove template artists before updating."
            )
        if np.iscomplexobj(signals):
            raise ValueError(
                "LineFormat.update requires real-valued signals; convert "
                "complex data with abs, real, or imag before plotting."
            )

        if signals.ndim == 1:
            signals = signals[None, :]
        elif signals.ndim > 2:
            signals = signals.reshape(-1, signals.shape[-1])

        for i, line in enumerate(self.lines):
            line.set_data(xs, signals[i, :].astype(np.float64))

        if title is not None:
            ax.set_title(title)

        ax.relim(visible_only=True)
        ax.autoscale_view()


class HeatmapFormat:
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


class NonUniformHeatmapFormat:
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


class ScatterFormat:
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: str | None = None,
        *,
        show_grid: bool = True,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.show_grid = show_grid

        self.scatter: PathCollection | None = None

    def init_ax(self, ax: Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)
        if self.show_grid:
            ax.grid()

        self.scatter = ax.scatter([], [], marker=".", s=2)

    def update(
        self,
        ax: Axes,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        colors: Sequence[str]
        | Sequence[tuple[float, float, float]]
        | Sequence[tuple[float, float, float, float]]
        | NDArray[np.float64]
        | None = None,
        title: str | None = None,
    ) -> None:
        if self.scatter is None:
            raise RuntimeError("Scatter not initialized.")
        figure = ax.get_figure()
        if (
            figure is None
            or ax not in figure.axes
            or self.scatter not in ax.collections
        ):
            raise RuntimeError(
                "Live plot scatter was removed from its axes or figure; "
                "do not clear or remove template artists before updating."
            )

        offsets = np.c_[xs.flatten(), ys.flatten()]
        self.scatter.set_offsets(offsets)
        if colors is not None:
            if isinstance(colors, np.ndarray):
                self.scatter.set_array(colors)
                self.scatter.set_clim(
                    np.nanmin(colors).item(), np.nanmax(colors).item()
                )
            else:
                self.scatter.set_color(colors)

        if title is not None:
            ax.set_title(title)

        ax.ignore_existing_data_limits = True
        ax.update_datalim(self.scatter.get_datalim(ax.transData))
        ax.autoscale_view()
