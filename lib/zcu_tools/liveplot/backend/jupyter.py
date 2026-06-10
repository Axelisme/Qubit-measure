from __future__ import annotations

import warnings
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import LivePlotBackend


def instant_plot(fig: Figure) -> None:
    # this ensures the figure is rendered in Jupyter notebooks right now and can be updated later
    canvas = fig.canvas

    if not hasattr(canvas, "toolbar_visible"):
        warnings.warn(
            "Warning: The matplotlib backend should be set to 'widget' for live plotting."
        )

    # # add hook for set_size_inches to update canvas size accordingly
    # # TODO: this is a bit hacky, but it works for now. We can consider a more elegant solution in the future.
    # original_set_size_inches = fig.set_size_inches

    # def patched_set_size_inches(*args, **kwargs):
    #     original_set_size_inches(*args, **kwargs)
    #     try:
    #         figsize = fig.get_size_inches()
    #         canvas.layout.width = f"{int(figsize[0] * fig.dpi)}px"  # type: ignore
    #         canvas.layout.height = f"{int(figsize[1] * fig.dpi)}px"  # type: ignore
    #         canvas._handle_message(canvas, {"type": "refresh"}, [])  # type: ignore
    #         canvas._handle_message(canvas, {"type": "draw"}, [])  # type: ignore
    #     except Exception as e:
    #         warnings.warn(f"Failed to update canvas size: {e}")

    # fig.set_size_inches = patched_set_size_inches  # type: ignore

    figsize = fig.get_size_inches()

    canvas.toolbar_visible = False  # type: ignore
    canvas.header_visible = False  # type: ignore
    canvas.footer_visible = False  # type: ignore
    canvas.layout.width = f"{int(figsize[0] * fig.dpi)}px"  # type: ignore
    canvas.layout.height = f"{int(figsize[1] * fig.dpi)}px"  # type: ignore
    canvas._handle_message(canvas, {"type": "refresh"}, [])  # type: ignore
    canvas._handle_message(canvas, {"type": "draw"}, [])  # type: ignore
    canvas._handle_message(canvas, {"type": "send_image_mode"}, [])  # type: ignore
    canvas._handle_message(canvas, {"type": "initialized"}, [])  # type: ignore

    display(canvas)


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


class JupyterBackend(LivePlotBackend):
    """Render into a Jupyter notebook (ipympl/widget canvas)."""

    def make_plot_frame(
        self, n_row: int, n_col: int, plot_instant: bool = False, **kwargs: Any
    ) -> tuple[Figure, list[list[Axes]]]:
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
        plt.close(fig)
