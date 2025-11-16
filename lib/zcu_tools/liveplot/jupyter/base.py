import warnings
from itertools import chain
from threading import Lock
from typing import List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
from IPython.display import display

from ..segments import AbsSegment

# Generic type variable used to correctly type the context-manager methods so that
# subclasses don't need to override ``__enter__`` just to narrow the return type.
T_JupyterPlotMixin = TypeVar("T_JupyterPlotMixin", bound="JupyterPlotMixin")


def instant_plot(fig: plt.Figure, figsize) -> None:
    # this ensures the figure is rendered in Jupyter notebooks right now and can be updated later
    canvas = fig.canvas

    if not hasattr(canvas, "toolbar_visible"):
        warnings.warn(
            "Warning: The matplotlib backend should be set to 'widget' for live plotting."
        )

    canvas.toolbar_visible = False
    canvas.header_visible = False
    canvas.footer_visible = False
    canvas.layout.width = f"{int(figsize[0] * 75)}px"
    canvas.layout.height = f"{int(figsize[1] * 75)}px"

    canvas._handle_message(canvas, {"type": "send_image_mode"}, [])
    canvas._handle_message(canvas, {"type": "refresh"}, [])
    canvas._handle_message(canvas, {"type": "initialized"}, [])
    canvas._handle_message(canvas, {"type": "draw"}, [])
    display(canvas)


def make_plot_frame(
    n_row: int, n_col: int, **kwargs
) -> Tuple[plt.Figure, List[List[plt.Axes]]]:
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
    fig, axs = plt.subplots(n_row, n_col, **kwargs)

    instant_plot(fig, kwargs["figsize"])

    return fig, axs


class JupyterPlotMixin:
    """live plotters in Jupyter notebooks."""

    def __init__(
        self,
        segments: List[List[AbsSegment]],
        existed_axes: Optional[List[List[plt.Axes]]] = None,
        auto_close: bool = True,
        disable: bool = False,
    ) -> None:
        if len(list(chain.from_iterable(segments))) == 0:
            raise ValueError("At least one segment is required.")
        n_row = len(segments)
        n_col = len(segments[0])

        # validate check
        for s_row in segments:
            if len(s_row) != n_col:
                raise ValueError(
                    "Number of segments in each row must match number of columns."
                )

        self.segments = segments
        self.update_lock = Lock()
        self.disable = disable
        self.auto_close = auto_close

        if disable:
            return  # early return

        if existed_axes is not None:
            # validate check
            valid = len(existed_axes) == n_row
            for a_row in existed_axes:
                if len(a_row) != n_col:
                    valid = False
            if not valid:
                raise ValueError(
                    "The shape of provided axes must match the shape of segments."
                )

            # if provided axes and display handle, use them
            self.fig = None
            self.axs = existed_axes
        else:
            # if not provided axes, create figure and display handle
            self.fig, self.axs = make_plot_frame(n_row, n_col)

    def clear(self) -> None:
        if self.disable:
            return

        with self.update_lock:
            for ax_row, seg_row in zip(self.axs, self.segments):
                for ax, segment in zip(ax_row, seg_row):
                    segment.clear(ax)
            self._refresh_while_lock()

    def _refresh_while_lock(self) -> None:
        assert self.update_lock.locked()

        if self.disable:
            return

        if self.fig is None:
            raise RuntimeError(
                "Try to refresh a plotter, but it have no own figure, this should happen when the figure is hosted outside of the plotter, you need to refresh the figure manually."
            )

        self.fig.canvas.draw()

    def refresh(self) -> None:
        if self.disable:
            return

        with self.update_lock:
            self._refresh_while_lock()

    def __enter__(self: T_JupyterPlotMixin) -> T_JupyterPlotMixin:
        if self.disable:
            return self

        for ax_row, seg_row in zip(self.axs, self.segments):
            for ax, segment in zip(ax_row, seg_row):
                segment.init_ax(ax)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.disable:
            return

        if self.auto_close and self.fig is not None:
            plt.close(self.fig)
