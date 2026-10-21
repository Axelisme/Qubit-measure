from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import LivePlotBackend

if TYPE_CHECKING:
    from matplotlib.animation import FFMpegWriter


def instant_plot(fig: Figure) -> None:
    from IPython.display import display

    # Force ipympl to display the live canvas before subsequent update calls.
    canvas = fig.canvas

    if not hasattr(canvas, "toolbar_visible"):
        warnings.warn(
            "Warning: The matplotlib backend should be set to 'widget' for live plotting.",
            stacklevel=2,
            # Direct calls and backend dispatch have different frame depths.
            skip_file_prefixes=(os.path.dirname(os.path.dirname(__file__)) + os.sep,),
        )

    figsize = fig.get_size_inches()

    # tolerated: ipympl defines toolbar visibility, not Figure.canvas's backend-neutral type.
    canvas.toolbar_visible = False  # pyright: ignore[reportAttributeAccessIssue]
    # tolerated: ipympl defines header visibility, not Figure.canvas's backend-neutral type.
    canvas.header_visible = False  # pyright: ignore[reportAttributeAccessIssue]
    # tolerated: ipympl defines footer visibility, not Figure.canvas's backend-neutral type.
    canvas.footer_visible = False  # pyright: ignore[reportAttributeAccessIssue]
    # tolerated: Widget layout is not in FigureCanvasBase; non-widget calls must fail here.
    canvas.layout.width = f"{int(figsize[0] * fig.dpi)}px"  # pyright: ignore[reportAttributeAccessIssue]
    # tolerated: Widget layout is not in FigureCanvasBase; retain the displayed pixel height.
    canvas.layout.height = f"{int(figsize[1] * fig.dpi)}px"  # pyright: ignore[reportAttributeAccessIssue]
    # tolerated: ipympl's refresh handshake has no backend-neutral canvas API.
    canvas._handle_message(canvas, {"type": "refresh"}, [])  # pyright: ignore[reportAttributeAccessIssue]
    # tolerated: ipympl's draw handshake has no backend-neutral canvas API.
    canvas._handle_message(canvas, {"type": "draw"}, [])  # pyright: ignore[reportAttributeAccessIssue]
    # tolerated: ipympl's image-mode handshake has no backend-neutral canvas API.
    canvas._handle_message(canvas, {"type": "send_image_mode"}, [])  # pyright: ignore[reportAttributeAccessIssue]
    # tolerated: ipympl's initialized handshake has no backend-neutral canvas API.
    canvas._handle_message(canvas, {"type": "initialized"}, [])  # pyright: ignore[reportAttributeAccessIssue]

    display(canvas)


def grab_frame_with_instant_plot(writer: FFMpegWriter, **savefig_kwargs) -> None:
    """Grab one ffmpeg frame from a figure prepared by ``instant_plot``."""
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
    # instant_plot-owned canvases keep their displayed size; do not resize here.
    # Do not resize the instant_plot canvas to the writer's frame size here.

    # Save the figure data to the sink, using the frame format and dpi.
    writer.fig.savefig(
        # tolerated: Matplotlib's stubs omit the live pipe; public grab_frame would resize.
        writer._proc.stdin,  # pyright: ignore[reportAttributeAccessIssue]
        format=writer.frame_format,
        dpi=writer.dpi,
        **savefig_kwargs,
    )


class JupyterBackend(LivePlotBackend):
    """Render into a Jupyter notebook (ipympl/widget canvas)."""

    def make_plot_frame(
        self, n_row: int, n_col: int, *, plot_instant: bool = False, **kwargs: Any
    ) -> tuple[Figure, list[list[Axes]]]:
        import matplotlib.pyplot as plt
        import numpy as np

        kwargs.setdefault("squeeze", False)
        kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
        fig, axs_nd = plt.subplots(n_row, n_col, **kwargs)
        # plt.subplots(squeeze=False) returns ndarray; convert to list[list[Axes]]
        # to satisfy the LivePlotBackend contract.
        axs: list[list[Axes]] = np.asarray(axs_nd).tolist()
        if plot_instant:
            instant_plot(fig)
        return fig, axs

    def instant_plot(self, fig: Figure) -> None:
        instant_plot(fig)

    def refresh_figure(self, fig: Figure) -> None:
        fig.canvas.draw()

    def close_figure(self, fig: Figure) -> None:
        import matplotlib.pyplot as plt

        plt.close(fig)
