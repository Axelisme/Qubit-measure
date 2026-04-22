from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def instant_plot(fig: Figure) -> None:
    # this ensures the figure is rendered in Jupyter notebooks right now and can be updated later
    canvas = fig.canvas

    if not hasattr(canvas, "toolbar_visible"):
        warnings.warn(
            "Warning: The matplotlib backend should be set to 'widget' for live plotting."
        )

    # add hook for set_size_inches to update canvas size accordingly
    # TODO: this is a bit hacky, but it works for now. We can consider a more elegant solution in the future.
    original_set_size_inches = fig.set_size_inches

    def patched_set_size_inches(*args, **kwargs):
        original_set_size_inches(*args, **kwargs)
        figsize = fig.get_size_inches()
        canvas.layout.width = f"{int(figsize[0] * fig.dpi)}px"  # type: ignore
        canvas.layout.height = f"{int(figsize[1] * fig.dpi)}px"  # type: ignore
        canvas._handle_message(canvas, {"type": "refresh"}, [])  # type: ignore
        canvas._handle_message(canvas, {"type": "draw"}, [])  # type: ignore

    fig.set_size_inches = patched_set_size_inches  # type: ignore

    canvas.toolbar_visible = False  # type: ignore
    canvas.header_visible = False  # type: ignore
    canvas.footer_visible = False  # type: ignore
    canvas._handle_message(canvas, {"type": "send_image_mode"}, [])  # type: ignore
    canvas._handle_message(canvas, {"type": "initialized"}, [])  # type: ignore

    display(canvas)


def make_plot_frame(
    n_row: int, n_col: int, plot_instant=False, **kwargs
) -> tuple[Figure, list[list[Axes]]]:
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
    fig, axs = plt.subplots(n_row, n_col, **kwargs)

    if plot_instant:
        instant_plot(fig)

    return fig, axs


def refresh_figure(fig: Figure) -> None:
    fig.canvas.draw()


def close_figure(fig: Figure) -> None:
    plt.close(fig)
