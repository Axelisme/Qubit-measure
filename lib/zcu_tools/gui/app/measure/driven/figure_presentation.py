"""Notify measure presentation after native Qt canvas attachment."""

from collections.abc import Callable

from matplotlib.figure import Figure

from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.ports import OwnerScheduler


class MeasurePlotHost(QtPlotHost):
    """Keep QtPlotHost rendering and publish its changed membership on the owner.

    The callback reads the operation's original named collection, not a copied
    registry. It runs synchronously after attachment; failures propagate to the
    producer just like attachment errors. Release remains QtPlotHost's job.
    """

    def __init__(
        self,
        container: FigureContainer,
        owner: OwnerScheduler,
        on_present: Callable[[], None],
    ) -> None:
        super().__init__(container, owner)
        self._on_present = on_present

    def present(self, figure: Figure) -> None:
        super().present(figure)
        self._on_present()
