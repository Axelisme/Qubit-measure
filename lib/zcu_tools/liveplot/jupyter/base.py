import warnings
from itertools import chain
from threading import Lock
from typing import List, Optional, Tuple, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..segments import AbsSegment

# Generic type variable used to correctly type the context-manager methods so that
# subclasses don't need to override ``__enter__`` just to narrow the return type.
T_JupyterPlotMixin = TypeVar("T_JupyterPlotMixin", bound="JupyterPlotMixin")


def instant_plot(fig: Figure) -> None:
    # this ensures the figure is rendered in Jupyter notebooks right now and can be updated later
    canvas = fig.canvas

    if not hasattr(canvas, "toolbar_visible"):
        warnings.warn(
            "Warning: The matplotlib backend should be set to 'widget' for live plotting."
        )

    figsize = fig.get_size_inches()

    pixel_per_inch = 70

    canvas.toolbar_visible = False  # type: ignore
    canvas.header_visible = False  # type: ignore
    canvas.footer_visible = False  # type: ignore
    canvas.layout.width = f"{int(figsize[0] * pixel_per_inch)}px"  # type: ignore
    canvas.layout.height = f"{int(figsize[1] * pixel_per_inch)}px"  # type: ignore

    canvas._handle_message(canvas, {"type": "send_image_mode"}, [])  # type: ignore
    canvas._handle_message(canvas, {"type": "refresh"}, [])  # type: ignore
    canvas._handle_message(canvas, {"type": "initialized"}, [])  # type: ignore
    canvas._handle_message(canvas, {"type": "draw"}, [])  # type: ignore

    display(canvas)


def make_plot_frame(
    n_row: int, n_col: int, plot_instant=True, **kwargs
) -> Tuple[Figure, List[List[Axes]]]:
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
    fig, axs = plt.subplots(n_row, n_col, **kwargs)

    if plot_instant:
        instant_plot(fig)

    return fig, axs


def grab_frame_with_instant_plot(writer: FFMpegWriter, **savefig_kwargs) -> None:
    """
    Equivalent to writer.grab_frame, but it work with figure setting by instant_plot.
    """
    # docstring inherited
    if mpl.rcParams["savefig.bbox"] == "tight":
        raise ValueError(
            f"{mpl.rcParams['savefig.bbox']=} must not be 'tight' as it "
            "may cause frame size to vary, which is inappropriate for animation."
        )
    for k in ("dpi", "bbox_inches", "format"):
        if k in savefig_kwargs:
            raise TypeError(f"grab_frame got an unexpected keyword argument {k!r}")

    # Readjust the figure size in case it has been changed by the user.
    # All frames must have the same size to save the movie correctly.
    # TODO: currently it have bug work with instant plot, maybe fix later
    # writer.fig.set_size_inches(writer._w, writer._h)

    # Save the figure data to the sink, using the frame format and dpi.
    writer.fig.savefig(
        writer._proc.stdin,  # type: ignore
        format=writer.frame_format,
        dpi=writer.dpi,
        **savefig_kwargs,
    )


class JupyterPlotMixin:
    """live plotters in Jupyter notebooks."""

    def __init__(
        self,
        segments: List[List[AbsSegment]],
        existed_axes: Optional[List[List[Axes]]] = None,
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

        plt.ioff()

        for ax_row, seg_row in zip(self.axs, self.segments):
            for ax, segment in zip(ax_row, seg_row):
                segment.init_ax(ax)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.disable:
            return

        plt.ion()

        if self.auto_close and self.fig is not None:
            plt.close(self.fig)
