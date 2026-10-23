"""Independent results window and its read-only browser component."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.registry import ExperimentResolution
from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from .results_comparison import ResultsComparison
from .results_details import result_details
from .results_gallery import ResultsGallery

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.controller import Controller


@dataclass(frozen=True)
class _Focus:
    entry: str | None = None
    experiment: str | None = None
    point: tuple[str, str | None] | None = None
    run_key: str | None = None
    figure: str | None = None


def _point_label(point: tuple[str, str | None]) -> str:
    return f"{point[0]} / {point[1] or 'Global / unknown'}"


class ResultsBrowserWindow(QMainWindow):
    """Retain one read-only results view; the measure window owns its lifetime."""

    def __init__(
        self, controller: Controller, parent: QMainWindow | None = None
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("results.browser")
        self.setWindowTitle("Results browser")
        self.resize(1400, 850)
        index = controller.query_results()
        self._browser = ResultsBrowser(
            index,
            controller.tab_control.open_tab_from_file,
            self,
            resolve_experiment=controller.resolve_experiment,
        )
        self.setCentralWidget(self._browser)

    def set_writable(self, *, writable: bool) -> None:
        """Gate new-tab commands while retaining read-only browsing and local pins."""
        self._browser.set_writable(writable=writable)


class ResultsBrowser(QWidget):
    """Browse an immutable index without changing the opened entry or point.

    open_run accepts a resolved adapter name and native data path, returning the
    existing Load outcome or raising a user-facing error. resolve_experiment is
    the catalog owner's query; one result per historical identity is shared by
    navigation, filters and opening. All callbacks run on the Qt owner thread.
    """

    def __init__(
        self,
        index: ResultsIndex,
        open_run: Callable[[str, str], LoadTabResultOutcome],
        parent: QWidget | None = None,
        *,
        resolve_experiment: Callable[[str | None], ExperimentResolution],
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._resolutions = {
            experiment: resolve_experiment(experiment)
            for experiment in {run.experiment for run in index.runs}
        }
        self._writable = True
        self._open_run = open_run
        self._focus = _Focus()
        self._history = [self._focus]
        self._history_position = 0
        self._navigation: list[tuple[QTreeWidgetItem, _Focus]] = []
        self._navigation_runs: tuple[ResultRun, ...] | None = None
        layout = QVBoxLayout(self)
        layout.addLayout(self._build_filters())
        layout.addLayout(self._build_history())
        splitter = QSplitter()
        self._tree = QTreeWidget()
        self._tree.setObjectName("results.navigation")
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._navigate)
        splitter.addWidget(self._tree)
        self._gallery = ResultsGallery(index)
        splitter.addWidget(self._gallery)
        splitter.addWidget(self._build_details())
        splitter.setSizes([220, 740, 390])
        layout.addWidget(splitter, 1)
        self._feedback = QLabel("")
        self._feedback.setObjectName("results.feedback")
        make_label_selectable(self._feedback)
        self._feedback.setWordWrap(True)
        layout.addWidget(self._feedback)
        self._render()

    def _build_filters(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self._group = QComboBox()
        self._group.setObjectName("results.group")
        self._group.addItems(["Experiment → point", "Point → experiment"])
        self._group.currentIndexChanged.connect(self._change_group)
        layout.addWidget(self._group)
        self._experiment = QComboBox()
        self._experiment.setObjectName("results.experiment")
        self._experiment.addItem("All experiments")
        self._experiment.addItems(
            sorted({self._experiment_name(run) for run in self._index.runs})
        )
        self._experiment.currentIndexChanged.connect(self._render)
        layout.addWidget(self._experiment)
        self._search = QLineEdit()
        self._search.setObjectName("results.search")
        self._search.setPlaceholderText("Search entries, roles, cfg and analysis")
        self._search.textChanged.connect(self._render)
        layout.addWidget(self._search, 1)
        self._date_from = self._date_filter(
            "results.date_from", "From YYYY-MM-DD", layout
        )
        self._date_to = self._date_filter("results.date_to", "To YYYY-MM-DD", layout)
        return layout

    def _date_filter(self, name: str, hint: str, layout: QHBoxLayout) -> QLineEdit:
        edit = QLineEdit()
        edit.setObjectName(name)
        edit.setPlaceholderText(hint)
        edit.setFixedWidth(140)
        edit.setToolTip(
            "Inclusive UTC date; runs without a timestamp are excluded when set"
        )
        edit.editingFinished.connect(self._render)
        layout.addWidget(edit)
        return edit

    def _build_history(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self._back = QPushButton("Back")
        self._back.clicked.connect(lambda: self._move_history(-1))
        self._forward = QPushButton("Forward")
        self._forward.clicked.connect(lambda: self._move_history(1))
        home = QPushButton("All results")
        home.clicked.connect(lambda: self._visit(_Focus()))
        for button in (self._back, self._forward, home):
            layout.addWidget(button)
        self._focus_label = QLabel()
        self._focus_label.setObjectName("results.focus")
        make_label_selectable(self._focus_label)
        layout.addWidget(self._focus_label, 1)
        return layout

    def _build_details(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        self._current_figure = QLabel("Current figure: none")
        self._current_figure.setObjectName("results.current_figure")
        make_label_selectable(self._current_figure)
        layout.addWidget(self._current_figure)
        self._details = QPlainTextEdit()
        self._details.setObjectName("results.details")
        self._details.setReadOnly(True)
        layout.addWidget(self._details, 2)
        self._open_button = QPushButton("Open in new analysis tab")
        self._open_button.clicked.connect(self._open_selected)
        layout.addWidget(self._open_button)
        self._pin_button = QPushButton("Add to comparison")
        self._pin_button.clicked.connect(self._pin_selected)
        layout.addWidget(self._pin_button)
        self.comparison = ResultsComparison(self._index)
        layout.addWidget(self.comparison, 1)
        return pane

    def set_writable(self, *, writable: bool) -> None:
        """Enable new-tab Load only for Local, without disabling navigation or pins."""
        self._writable = writable
        self._render_details(self._focused(self._filtered()))

    def _change_group(self) -> None:
        self._navigation_runs = None
        self._render()

    def _visit(self, focus: _Focus) -> None:
        if focus != self._focus:
            self._history = self._history[: self._history_position + 1]
            self._history.append(focus)
            self._history_position += 1
            self._focus = focus
        self._render()

    def _move_history(self, delta: int) -> None:
        target = self._history_position + delta
        if 0 <= target < len(self._history):
            self._history_position = target
            self._focus = self._history[target]
            self._render()

    def _experiment_name(self, run: ResultRun) -> str:
        resolution = self._resolutions[run.experiment]
        return resolution.adapter_name or run.experiment or "Unknown experiment"

    def _filtered(self) -> tuple[ResultRun, ...]:
        lower = self._date_from.text().strip()
        upper = self._date_to.text().strip()
        try:
            for value in (lower, upper):
                if value:
                    date.fromisoformat(value)
            if lower and upper and lower > upper:
                raise ValueError("From date must not follow To date")
        except ValueError as error:
            self._feedback.setText(f"Invalid date filter: {error}")
            return ()
        search = self._search.text().casefold().strip()
        experiment = (
            self._experiment.currentText() if self._experiment.currentIndex() else None
        )
        return tuple(
            run
            for run in self._index.runs
            if (experiment is None or self._experiment_name(run) == experiment)
            and search in run.search_text
            and (not lower or (run.at is not None and run.at[:10] >= lower))
            and (not upper or (run.at is not None and run.at[:10] <= upper))
        )

    def _focused(self, runs: tuple[ResultRun, ...]) -> tuple[ResultRun, ...]:
        focus = self._focus
        return tuple(
            run
            for run in runs
            if (focus.entry is None or run.entry == focus.entry)
            and (
                focus.experiment is None
                or self._experiment_name(run) == focus.experiment
            )
            and (focus.point is None or (run.entry, run.point) == focus.point)
            and (focus.run_key is None or run.key == focus.run_key)
        )

    def _render(self) -> None:
        filtered = self._filtered()
        runs = self._focused(filtered)
        self._back.setEnabled(self._history_position > 0)
        self._forward.setEnabled(self._history_position + 1 < len(self._history))
        title = self._focus.entry or "All results"
        if self._focus.experiment is not None:
            title = self._focus.experiment
        if self._focus.point is not None:
            title = _point_label(self._focus.point)
        if self._focus.run_key is not None:
            title = self._index.get_run(self._focus.run_key).run_path.name
        self._focus_label.setText(f"{title} · {len(runs)} runs")
        self._render_navigation(filtered)
        self._gallery.present(
            runs,
            self._activate_figure,
            self._figure_action,
            all_figures=self._focus.run_key is not None,
        )
        self._render_details(runs)

    def _run_branch(self, run: ResultRun) -> tuple[tuple[str, _Focus], ...]:
        entry = _Focus(entry=run.entry)
        experiment = self._experiment_name(run)
        point = (run.entry, run.point)
        if self._group.currentIndex() == 1:
            first = replace(entry, point=point)
            second = replace(first, experiment=experiment)
            captions = (run.point or "Global / unknown", experiment)
        else:
            first = replace(entry, experiment=experiment)
            second = replace(first, point=point)
            captions = (experiment, run.point or "Global / unknown")
        return (
            (run.entry, entry),
            (captions[0], first),
            (captions[1], second),
            (run.run_path.name, replace(second, run_key=run.key)),
        )

    def _render_navigation(self, runs: tuple[ResultRun, ...]) -> None:
        if self._navigation_runs != runs:
            expanded = {focus for item, focus in self._navigation if item.isExpanded()}
            self._tree.clear()
            self._navigation.clear()
            nodes: dict[_Focus, QTreeWidgetItem] = {}
            for run in runs:
                parent = self._tree.invisibleRootItem()
                for caption, focus in self._run_branch(run):
                    if focus not in nodes:
                        item = QTreeWidgetItem(parent, [caption])
                        item.setExpanded(focus in expanded)
                        nodes[focus] = item
                        self._navigation.append((item, focus))
                    parent = nodes[focus]
            self._navigation_runs = runs
        for item, focus in self._navigation:
            if focus == replace(self._focus, figure=None):
                self._tree.setCurrentItem(item)
                break
        else:
            self._tree.clearSelection()

    def _navigate(self, item: QTreeWidgetItem, _column: int) -> None:
        for node, focus in self._navigation:
            if node is item:
                self._visit(focus)
                break

    def _activate_figure(self, run: ResultRun, figure: str) -> None:
        self._visit(
            _Focus(
                entry=run.entry,
                experiment=self._experiment_name(run),
                point=(run.entry, run.point),
                run_key=run.key,
                figure=figure,
            )
        )

    def _render_details(self, runs: tuple[ResultRun, ...]) -> None:
        run = (
            self._index.get_run(self._focus.run_key)
            if self._focus.run_key and runs
            else None
        )
        self._details.setPlainText(
            result_details(self._index, run)
            if run
            else "Select a run to inspect cfg and analysis"
        )
        reason = "Select a run"
        if run is not None:
            reason = "No data.h5 in this saved run" if run.data_path is None else ""
        if not self._writable:
            reason = "Local control is required to open an analysis tab"
        self._open_button.setEnabled(not reason)
        self._open_button.setToolTip(reason)
        self._pin_button.setEnabled(run is not None)
        self._current_figure.setText(f"Current figure: {self._focus.figure or 'none'}")

    def _pin_selected(self) -> None:
        if self._focus.run_key is not None:
            self.comparison.pin_run(
                self._index.get_run(self._focus.run_key), self._focus.figure
            )

    def _open_selected(self) -> None:
        if self._focus.run_key is None:
            return
        run = self._index.get_run(self._focus.run_key)
        resolution = self._resolutions[run.experiment]
        if resolution.adapter_name is None:
            self._feedback.setText(f"Cannot open analysis tab: {resolution.reason}")
            return
        try:
            outcome = self._open_run(resolution.adapter_name, str(run.data_path))
        except Exception as error:
            logger.exception("Cannot open saved history in a new tab")
            self._feedback.setText(f"Cannot open analysis tab: {error}")
        else:
            messages = [f"Loaded history: {outcome.data_path}"]
            if outcome.analysis_source_issue is not None:
                messages.append(
                    f"History loaded, no analysis source: {outcome.analysis_source_issue}"
                )
            self._feedback.setText("\n".join(messages))
            if outcome.analysis_error:
                self._feedback.setText(
                    self._feedback.text()
                    + f"\nAnalysis preparation failed: {outcome.analysis_error}"
                )

    def _figure_action(self, run: ResultRun, figure: str, action: str) -> None:
        clipboard = QApplication.clipboard()
        if action == "Add to comparison":
            self.comparison.pin_run(run, figure)
            return
        if clipboard is None:
            self._feedback.setText("Clipboard unavailable")
            return
        if action == "Copy path":
            path = next(path for path in run.figure_paths if path.name == figure)
            clipboard.setText(str(path))
            self._feedback.setText(f"Copied path: {path}")
        elif action == "Copy image":
            try:
                image = QImage.fromData(self._index.read_figure(run.key, figure))
                if image.isNull():
                    raise ValueError("Cannot decode saved PNG")
            except (OSError, ValueError, KeyError) as error:
                self._feedback.setText(f"Cannot copy image: {error}")
                return
            clipboard.setImage(image)
            self._feedback.setText(f"Copied image: {figure}")
