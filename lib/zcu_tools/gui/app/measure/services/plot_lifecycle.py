"""Release presentation without destroying detached analysis Figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.plotting.plots import Plots


def discard_unpublished_plots(plots: Plots) -> None:
    """Stop an unsuccessful producer and detach even if its final draw fails."""
    try:
        plots.finish(present=False)
    finally:
        plots.release()
