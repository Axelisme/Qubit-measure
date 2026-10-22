"""Qt presentation for retained workflow figures; data stays with the facet."""

from collections.abc import Callable

from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QComboBox, QLabel, QStackedWidget, QVBoxLayout, QWidget

from zcu_tools.experiment.workflows import WorkflowPlots
from zcu_tools.gui.plotting.container import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.ports import OwnerScheduler


class WorkflowFigures(QWidget):
    """Present named native figures with a selector, using the shared Qt plot host.

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
        self._choices = QStandardItemModel(self._selector)
        self._selector.setModel(self._choices)
        self._selector.currentTextChanged.connect(self._select)
        layout.addWidget(self._selector)
        self._selector.hide()
        self._stack = QStackedWidget(self)
        placeholder = QLabel("Workflow plots", self._stack)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(placeholder)
        layout.addWidget(self._stack, 1)
        self._container = FigureContainer(self._stack, placeholder)
        self._host = QtPlotHost(self._container, owner)

    def call[T](self, callback: Callable[[], T]) -> T:
        """Run a native artist update on the session owner, propagating failures."""
        return self._host.call(callback)

    def present(self, figure: Figure) -> None:
        """Present a retained named figure on-owner; reject unknown figures."""
        plots = self._plots()
        if plots is None:
            raise ValueError("This workflow tab has no retained plots")
        name = next(
            (name for name, retained in plots.figures().items() if retained is figure),
            None,
        )
        if name is None:
            raise ValueError("Figure is not retained by this workflow tab")
        self._host.present(figure)
        if name not in self._figures:
            self._figures[name] = figure
            self._rebuild_selector(name)
        self._selector.setCurrentText(name)
        self._selector.setVisible(bool(self._figures))

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        """Refresh a presented figure through the shared coalescing Qt host."""
        self._host.refresh(figure, final=final)

    def release(self, figure: Figure) -> None:
        """Release the figure's canvas and pending redraw without clearing artists."""
        self._host.release(figure)
        for name, retained in tuple(self._figures.items()):
            if retained is figure:
                del self._figures[name]
        self._rebuild_selector(self._selector.currentText())
        self._selector.setVisible(bool(self._figures))

    def dispose(self) -> None:
        """Cancel this adapter's remaining redraws after the producer has stopped."""
        for figure in tuple(self._figures.values()):
            self.release(figure)

    def _rebuild_selector(self, selected: str) -> None:
        self._selector.blockSignals(True)
        try:
            self._choices.clear()
            for title, operation in (("Workflow", False), ("Current operation", True)):
                names = [
                    name
                    for name in self._figures
                    if name.startswith("operation/") == operation
                ]
                if not names:
                    continue
                heading = QStandardItem(title)
                heading.setEnabled(False)
                self._choices.appendRow(heading)
                for name in names:
                    self._choices.appendRow(QStandardItem(name))
            if selected not in self._figures:
                selected = next(iter(self._figures), "")
            self._selector.setCurrentText(selected)
        finally:
            self._selector.blockSignals(False)
        self._select(selected)

    def _select(self, name: str) -> None:
        figure = self._figures.get(name)
        if figure is not None and isinstance(figure.canvas, QWidget):
            self._container.set_current_canvas(figure.canvas)
