"""Independent results window and its read-only browser component."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from qtpy.QtCore import QDate, Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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
from .results_details import analysis_details, result_details
from .results_gallery import ResultImageAction, ResultsGallery

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
        self.setStyleSheet("""
            QWidget { background: #f6f8fb; color: #1e293b; font-size: 12px; }
            QTreeWidget, QTableWidget, QPlainTextEdit, QLineEdit, QComboBox, QDateEdit {
                background: white; border: 1px solid #cbd5e1; border-radius: 3px; }
            QPushButton { background: #e7edf6; border: 1px solid #b7c9df;
                padding: 6px 9px; border-radius: 4px; }
            QPushButton:hover { background: #d3e5ff; }
            QPushButton:disabled { color: #94a3b8; }
            QHeaderView::section { background: #e9eff7; padding: 5px; border: 0; }
            QTreeWidget::item:selected { background: #dceaff; color: #17365c; }
        """)
        self._writable = True
        self._open_run = open_run
        self._focus = _Focus()
        self._history = [self._focus]
        self._history_position = 0
        self._navigation: list[tuple[QTreeWidgetItem, _Focus]] = []
        self._navigation_runs: tuple[ResultRun, ...] | None = None
        layout = QVBoxLayout(self)
        heading = QLabel("Results browser  /  Saved results · read only")
        heading.setStyleSheet("font-size: 17px; font-weight: 600; padding: 5px;")
        layout.addWidget(heading)
        splitter = QSplitter()
        left = QWidget()
        filters = QVBoxLayout(left)
        filters.setContentsMargins(0, 0, 5, 0)
        filters.addLayout(self._build_filters())
        self._tree = QTreeWidget()
        self._tree.setObjectName("results.navigation")
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._navigate)
        self._tree.setMinimumWidth(300)
        self._tree.setIndentation(15)
        filters.addWidget(self._tree, 1)
        splitter.addWidget(left)
        center = QWidget()
        central = QVBoxLayout(center)
        central.setContentsMargins(4, 0, 4, 0)
        central.addLayout(self._build_history())
        self._tabs = QTabWidget()
        self._tabs.setObjectName("results.tabs")
        self._gallery = ResultsGallery(index)
        self._tabs.addTab(self._gallery, "Gallery")
        self.comparison = ResultsComparison(self._index)
        self.comparison.run_activated.connect(self._open_compared)
        self._tabs.addTab(self.comparison, "Comparison")
        central.addWidget(self._tabs, 1)
        splitter.addWidget(center)
        splitter.addWidget(self._build_details())
        splitter.setSizes([340, 700, 350])
        layout.addWidget(splitter, 1)
        self._feedback = QLabel("")
        self._feedback.setObjectName("results.feedback")
        make_label_selectable(self._feedback)
        self._feedback.setWordWrap(True)
        layout.addWidget(self._feedback)
        self._render()

    def _build_filters(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        row = QHBoxLayout()
        self._group = QComboBox()
        self._group.setObjectName("results.group")
        self._group.addItems(["Point → experiment", "Experiment → point"])
        self._group.currentIndexChanged.connect(self._change_group)
        row.addWidget(self._group)
        self._experiment = QComboBox()
        self._experiment.setObjectName("results.experiment")
        self._experiment.addItem("All experiments")
        self._experiment.addItems(
            sorted({self._experiment_name(run) for run in self._index.runs})
        )
        self._experiment.currentIndexChanged.connect(self._render)
        row.insertWidget(0, self._experiment, 1)
        layout.addLayout(row)
        self._search = QLineEdit()
        self._search.setObjectName("results.search")
        self._search.setPlaceholderText("Search entries, roles, cfg and analysis")
        self._search.textChanged.connect(self._render)
        layout.addWidget(self._search, 1)
        dates = QHBoxLayout()
        self._dates_enabled = QCheckBox("UTC dates")
        self._dates_enabled.setObjectName("results.dates_enabled")
        dates.addWidget(self._dates_enabled)
        known = sorted(run.at[:10] for run in self._index.runs if run.at)
        self._date_from = self._date_filter(
            "results.date_from", known[0] if known else "2000-01-01", dates
        )
        dates.addWidget(QLabel("to"))
        self._date_to = self._date_filter(
            "results.date_to",
            known[-1] if known else QDate.currentDate().toString("yyyy-MM-dd"),
            dates,
        )
        self._dates_enabled.toggled.connect(self._toggle_dates)
        layout.addLayout(dates)
        self._visible_count = QLabel()
        self._visible_count.setObjectName("results.visible_count")
        make_label_selectable(self._visible_count)
        layout.addWidget(self._visible_count)
        return layout

    def _date_filter(self, name: str, initial: str, layout: QHBoxLayout) -> QDateEdit:
        edit = QDateEdit(QDate.fromString(initial, "yyyy-MM-dd"))
        edit.setObjectName(name)
        edit.setDisplayFormat("yyyy-MM-dd")
        edit.setCalendarPopup(True)
        edit.setEnabled(False)
        edit.setToolTip(
            "Inclusive UTC date; runs without a timestamp are excluded when enabled"
        )
        edit.dateChanged.connect(self._render)
        layout.addWidget(edit)
        return edit

    def _toggle_dates(self) -> None:
        enabled = self._dates_enabled.isChecked()
        self._date_from.setEnabled(enabled)
        self._date_to.setEnabled(enabled)
        self._render()

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
        self._open_button = QPushButton("Open in new analysis tab")
        self._open_button.clicked.connect(self._open_selected)
        layout.addWidget(self._open_button)
        self._pin_button = QPushButton("Add to comparison")
        self._pin_button.clicked.connect(self._pin_selected)
        layout.addWidget(self._pin_button)
        layout.addWidget(self._current_figure)
        tabs = QTabWidget()
        tabs.setObjectName("results.detail_tabs")
        self._details = QPlainTextEdit()
        self._details.setObjectName("results.details")
        self._details.setReadOnly(True)
        tabs.addTab(self._details, "Metadata / cfg")
        self._analysis = QTableWidget(0, 2)
        self._analysis.setObjectName("results.analysis")
        self._analysis.setHorizontalHeaderLabels(["Key", "Value"])
        self._analysis.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._analysis.horizontalHeader()
        if header is None:
            raise RuntimeError("Analysis table has no header")
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._analysis.setColumnWidth(1, 110)
        tabs.addTab(self._analysis, "Analysis")
        layout.addWidget(tabs, 1)
        return pane

    def set_writable(self, *, writable: bool) -> None:
        """Enable new-tab Load only for Local, without disabling navigation or pins."""
        self._writable = writable
        self._render_details(self._focused(self._filtered()))

    def _change_group(self) -> None:
        self._navigation_runs = None
        self._render_navigation(self._filtered())

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
        enabled = self._dates_enabled.isChecked()
        lower = self._date_from.date().toString("yyyy-MM-dd") if enabled else ""
        upper = self._date_to.date().toString("yyyy-MM-dd") if enabled else ""
        if lower > upper:
            self._feedback.setText(
                "Invalid date filter: From date must not follow To date"
            )
            return ()
        self._feedback.clear()
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
        self._visible_count.setText(f"{len(filtered)} / {len(self._index.runs)} runs")
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
        if self._group.currentIndex() == 0:
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
            (
                f"{run.run_path.name[-6:]}  ·  {run.at[11:19] if run.at else 'Unknown time'}",
                replace(second, run_key=run.key),
            ),
        )

    def _render_navigation(self, runs: tuple[ResultRun, ...]) -> None:
        if self._navigation_runs != runs:
            expanded = {focus for item, focus in self._navigation if item.isExpanded()}
            self._tree.clear()
            self._navigation.clear()
            nodes: dict[_Focus, QTreeWidgetItem] = {}
            counts: dict[_Focus, int] = {}
            for run in runs:
                parent = self._tree.invisibleRootItem()
                for caption, focus in self._run_branch(run):
                    if focus not in nodes:
                        item = QTreeWidgetItem(parent, [caption])
                        item.setExpanded(
                            focus in expanded
                            or (
                                focus.run_key is None
                                and (focus.point is None or focus.experiment is None)
                            )
                        )
                        item.setToolTip(
                            0,
                            f"{run.entry} / {run.point or 'Global / unknown'}\n{run.experiment or 'Unknown experiment'}\n{run.run_path.name}\n{run.at or 'Unknown time'}\n{run.run_path}",
                        )
                        nodes[focus] = item
                        self._navigation.append((item, focus))
                    parent = nodes[focus]
                    counts[focus] = counts.get(focus, 0) + 1
            for focus, item in nodes.items():
                if focus.run_key is None:
                    item.setText(0, f"{item.text(0)}  ({counts[focus]})")
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
        rows = analysis_details(self._index, run) if run else ()
        self._analysis.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            for column, text in enumerate((key, value)):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self._analysis.setItem(row, column, item)
        self._analysis.resizeRowsToContents()
        reason = "Select a run"
        if run is not None:
            reason = "No data.h5 in this saved run" if run.data_path is None else ""
        if not self._writable:
            reason = "Local control is required to open an analysis tab"
        self._open_button.setEnabled(not reason)
        self._open_button.setToolTip(reason)
        self._pin_button.setEnabled(run is not None)
        self._current_figure.setText(f"Current figure: {self._focus.figure or 'none'}")

    def _open_compared(self, key: str) -> None:
        run = self._index.get_run(key)
        self._visit(
            _Focus(
                entry=run.entry,
                experiment=self._experiment_name(run),
                point=(run.entry, run.point),
                run_key=key,
                figure=self.comparison.selected_figure(key),
            )
        )
        self._tabs.setCurrentIndex(0)

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

    def _figure_action(
        self, run: ResultRun, figure: str, action: ResultImageAction
    ) -> None:
        clipboard = QApplication.clipboard()
        if action is ResultImageAction.ADD_TO_COMPARISON:
            self.comparison.pin_run(run, figure)
            return
        if clipboard is None:
            self._feedback.setText("Clipboard unavailable")
            return
        if action is ResultImageAction.COPY_PATH:
            path = next(path for path in run.figure_paths if path.name == figure)
            clipboard.setText(str(path))
            self._feedback.setText(f"Copied path: {path}")
        elif action is ResultImageAction.COPY_IMAGE:
            try:
                image = QImage.fromData(self._index.read_figure(run.key, figure))
                if image.isNull():
                    raise ValueError("Cannot decode saved PNG")
            except (OSError, ValueError, KeyError) as error:
                self._feedback.setText(f"Cannot copy image: {error}")
                return
            clipboard.setImage(image)
            self._feedback.setText(f"Copied image: {figure}")
