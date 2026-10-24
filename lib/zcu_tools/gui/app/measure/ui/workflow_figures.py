"""Qt presentation for retained workflow figures; data stays with the facet."""

from collections.abc import Callable

from matplotlib.figure import Figure
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QVBoxLayout, QWidget

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
    Follow latest starts checked and selects newly presented or live-refreshed
    figures. A canvas click unchecks it and pins the tile; final flushes keep focus.
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
        self._follow = QCheckBox("Follow latest", self)
        self._follow.setObjectName("workflow.figures.follow")
        self._follow.setChecked(True)
        layout.addWidget(self._follow)
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

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if (
            a1 is not None
            and a1.type() == QEvent.Type.MouseButtonPress
            and isinstance(a0, QWidget)
            and self._stack.indexOf(a0) > 0
        ):
            self._follow.setChecked(False)
        return super().eventFilter(a0, a1)

    def call[T](self, callback: Callable[[], T]) -> T:
        """Run a native artist update on the session owner, propagating failures."""
        return self._host.call(callback)

    def present(self, figure: Figure) -> None:
        """Present a retained named figure on-owner; reject unknown figures.

        Follow latest selects a newly attached figure and its page. Otherwise
        keep the current tile/page, including when re-presenting an attached figure.
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
        if name not in self._figures and self._follow.isChecked():
            selected = name
        self._host.present(figure)
        self._figures[name] = figure
        self._present_tiles(selected)

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        """Refresh a presented figure, following live updates when checked.

        A final flush renders the last data without moving the current tile/page.
        """
        self._host.refresh(figure, final=final)
        if not final and self._follow.isChecked():
            self._stack.setCurrentIndex(list(self._figures.values()).index(figure) + 1)

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
        for index in range(1, self._stack.count()):
            canvas = self._stack.widget(index)
            if canvas is not None:
                canvas.installEventFilter(self)
