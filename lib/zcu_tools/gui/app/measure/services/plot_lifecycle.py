"""Release presentation without destroying detached analysis Figures."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import RetiredPaneResources
    from zcu_tools.plotting.plots import Plots

logger = logging.getLogger(__name__)


def discard_unpublished_plots(plots: Plots) -> None:
    """Stop an unsuccessful producer and detach even if its final draw fails."""
    try:
        plots.finish(present=False)
    finally:
        plots.release()


def release_retired_plots(retired: RetiredPaneResources) -> None:
    """Committed State is not rolled back if old presentation cleanup fails."""
    for plots in retired.plots:
        try:
            plots.release()
        except Exception:
            logger.exception("Retired analysis plot release failed")
