from itertools import chain
from threading import Lock
from typing import List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
from IPython.display import display

from ..segments import AbsSegment

# Generic type variable used to correctly type the context-manager methods so that
# subclasses don't need to override ``__enter__`` just to narrow the return type.
T_JupyterPlotMixin = TypeVar("T_JupyterPlotMixin", bound="JupyterPlotMixin")


def make_plot_frame(
    n_row: int, n_col: int, **kwargs
) -> Tuple[plt.FigureBase, List[List[plt.Axes]]]:
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 3 * n_row))
    fig, axs = plt.subplots(n_row, n_col, **kwargs)
    fig.tight_layout(pad=2.0)

    # this ensures the figure is rendered in Jupyter notebooks right now and can be updated later
    canvas = fig.canvas
    display(canvas)
    canvas._handle_message(canvas, {"type": "send_image_mode"}, [])
    canvas._handle_message(canvas, {"type": "refresh"}, [])
    canvas._handle_message(canvas, {"type": "initialized"}, [])
    canvas._handle_message(canvas, {"type": "draw"}, [])

    return fig, axs


class JupyterPlotMixin:
    """live plotters in Jupyter notebooks."""

    def __init__(
        self,
        segments: List[List[AbsSegment]],
        existed_frames: Optional[Tuple[plt.FigureBase, List[List[plt.Axes]]]] = None,
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

        if disable:
            return  # early return

        if existed_frames is not None:
            # if provided axes and display handle, use them
            provided_fig, provided_axs = existed_frames

            # validate check
            valid = len(provided_axs) == n_row
            for a_row in provided_axs:
                if len(a_row) != n_col:
                    valid = False
            if not valid:
                raise ValueError(
                    "The shape of provided axes must match the shape of segments."
                )

            self.fig = provided_fig
            self.axs = provided_axs
            self.auto_close = False  # force not to close the figure
        else:
            # if not provided axes, create figure and display handle
            self.fig, self.axs = make_plot_frame(n_row, n_col)
            self.auto_close = auto_close

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

        if self.auto_close:
            plt.close(self.fig)
