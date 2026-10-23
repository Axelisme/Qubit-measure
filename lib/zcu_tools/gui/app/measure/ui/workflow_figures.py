"""Qt presentation for retained workflow figures; data stays with the facet."""

from collections.abc import Callable

from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from zcu_tools.experiment.workflows import WorkflowPlots
from zcu_tools.gui.plotting.container import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.ports import OwnerScheduler

from .figure_pager import FigurePager
from .figure_pane_layout import FigurePaneLayout
from .named_figure_presentation import present_named_figure_tiles, sync_figure_selection


class WorkflowFigures(QWidget):
    """Present named native figures in pages of four using the shared Qt plot host.

    Construct on the GUI thread with the session owner and a read-only WorkflowPlots
    query for this tab. The producer must stop and release its plots before
    dispose or Qt deletion. No run, source capture or result data is owned here.
    """

    def __init__(
        self,
        owner: OwnerScheduler,
        plots: Callable[[], WorkflowPlots | None],
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self._plots = plots
        self._figures: dict[str, Figure] = {}
        layout = QVBoxLayout(self)
        self._selector = QComboBox(self)
        self._selector.setObjectName("workflow.figures.selector")
        self._selector.hide()
        self._stack = FigurePaneLayout()
        placeholder = QLabel("Workflow plots", self._stack)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(placeholder)
        self._stack.currentChanged.connect(
            lambda index: sync_figure_selection(self._selector, index)
        )
        layout.addWidget(FigurePager(self._stack, "workflow"))
        layout.addWidget(self._stack, 1)
        self._container = FigureContainer(self._stack, placeholder)
        self._host = QtPlotHost(self._container, owner)

    def call[T](self, callback: Callable[[], T]) -> T:
        """Run a native artist update on the session owner, propagating failures."""
        return self._host.call(callback)

    def present(self, figure: Figure) -> None:
        """Present a retained named figure on-owner; reject unknown figures.

        Keep the current tile/page when adding or re-presenting figures. The first
        attached figure becomes current until a page change or canvas click.
        """
        plots = self._plots()
        if plots is None:
            raise ValueError("This workflow tab has no retained plots")
        name = next(
            (name for name, retained in plots.figures().items() if retained is figure),
            None,
        )
        if name is None:
            raise ValueError("Figure is not retained by this workflow tab")
        selected = self._selector.currentText()
        self._host.present(figure)
        self._figures[name] = figure
        self._present_tiles(selected)

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        """Refresh a presented figure through the shared coalescing Qt host."""
        self._host.refresh(figure, final=final)

    def release(self, figure: Figure) -> None:
        """Release the figure's canvas and pending redraw without clearing artists."""
        selected = self._selector.currentText()
        self._host.release(figure)
        for name, retained in tuple(self._figures.items()):
            if retained is figure:
                del self._figures[name]
        self._present_tiles(selected)

    def dispose(self) -> None:
        """Cancel this adapter's remaining redraws after the producer has stopped."""
        for figure in tuple(self._figures.values()):
            self.release(figure)

    def get_current_figure(self) -> Figure | None:
        """Borrow the current native Figure on the Qt owner thread, or None.

        A page change selects that page's first figure; a tile click selects that
        figure. An empty/placeholder pane returns None. A mounted canvas without
        a Figure raises RuntimeError. This never renders or transfers ownership.
        """
        if self._stack.currentIndex() == 0:
            return None
        figure = getattr(self._stack.currentWidget(), "figure", None)
        if not isinstance(figure, Figure):
            raise RuntimeError("Workflow canvas has no matplotlib figure")
        return figure

    def _present_tiles(self, selected: str) -> None:
        present_named_figure_tiles(
            self._stack,
            self._selector,
            list(self._figures),
            selected=selected,
            pane="workflow",
        )
