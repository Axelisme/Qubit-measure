"""Independent results window and its read-only browser component."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QDate, QEvent, QObject, QPoint, QPointF, QSignalBlocker, Qt
from qtpy.QtGui import QAction, QHideEvent, QImage, QMouseEvent, QPixmap, QResizeEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSplitter,
    QStatusBar,
    QStyle,
    QStyleOptionComboBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.registry import ExperimentResolution
from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
from zcu_tools.gui.app.measure.services.results_index import ResultRun, ResultsIndex
from zcu_tools.gui.app.measure.services.results_index_owner import ResultsScanState
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from .results_comparison import ResultsComparison
from .results_details import ResultsDetailsTree
from .results_gallery import ResultImageAction, ResultsGallery
from .results_lightbox import ResultLightbox

logger = logging.getLogger(__name__)

_LOAD_TOOLTIP = (
    "Load this run's raw data into a new analysis tab in the main window "
    "for re-analysis; no measurement."
)

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


class _CalendarDateEdit(QDateEdit):
    def __init__(self, date: QDate) -> None:
        super().__init__(date)
        editor = self.lineEdit()
        if editor is None:
            raise RuntimeError("Date edit has no text editor")
        editor.installEventFilter(self)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if (
            a0 is self.lineEdit()
            and isinstance(a1, QMouseEvent)
            and a1.type() == QEvent.Type.MouseButtonPress
            and a1.button() == Qt.MouseButton.LeftButton
            and self.calendarPopup()
        ):
            # Route the text click through Qt's own calendar-arrow handling.
            option = QStyleOptionComboBox()
            option.initFrom(self)
            option.editable = True
            style = self.style()
            assert style is not None
            arrow = style.subControlRect(
                QStyle.ComplexControl.CC_ComboBox,
                option,
                QStyle.SubControl.SC_ComboBoxArrow,
                self,
            ).center()
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            super().mousePressEvent(
                QMouseEvent(
                    a1.type(),
                    QPointF(arrow),
                    QPointF(self.mapToGlobal(arrow)),
                    a1.button(),
                    a1.buttons(),
                    a1.modifiers(),
                )
            )
            return True
        return super().eventFilter(a0, a1)


class ResultsBrowserWindow(QMainWindow):
    """Show the retained app-owned scan without doing filesystem I/O on construction.

    Close hides this top-level window. MainWindow calls dispose at teardown
    to detach the subscription before deleting the view (ADR-0067).
    """

    def __init__(
        self, controller: Controller, parent: QMainWindow | None = None
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("results.browser")
        self.setWindowTitle("Results browser")
        self.resize(1400, 850)
        index = ResultsIndex(Path(controller.get_project_root()).absolute(), ())
        self._lifetime = LifetimeScope(self)

        def open_run(adapter: str, path: str) -> LoadTabResultOutcome:
            outcome = controller.tab_control.open_tab_from_file(adapter, path)
            if parent is not None:
                parent.raise_()
                parent.activateWindow()
            return outcome

        self._browser = ResultsBrowser(
            index,
            open_run,
            self,
            resolve_experiment=controller.resolve_experiment,
        )
        self.setCentralWidget(self._browser)
        self._lifetime.add_cleanup(self._browser.dispose)
        self._index_status = QStatusBar(self)
        self._index_status.setObjectName("results.index_status")
        self.setStatusBar(self._index_status)
        self._lifetime.add_cleanup(
            controller.watch_results_index(self._lifetime.callback(self._present_scan))
        )

    def _present_scan(self, state: ResultsScanState) -> None:
        self._browser.present_index(state.index)
        runs = len(state.index.runs)
        match state.phase:
            case "indexing":
                message = (
                    "Indexing results…"
                    if state.total_entries is None
                    else f"Indexing results: {state.completed_entries} / {state.total_entries} entries · {runs} runs"
                )
            case "ready":
                issues = len(state.index.issues) + sum(
                    len(run.issues) for run in state.index.runs
                )
                message = f"Index ready: {runs} runs · {issues} issues"
            case "cancelled":
                message = f"Index cancelled: {runs} runs"
            case "failed":
                message = f"Index failed: {state.error}"
        self._index_status.showMessage(message)

    def dispose(self) -> None:
        """Retire the view and its index subscription without stopping the scan."""
        self._lifetime.close()
        self.close()

    def set_writable(self, *, writable: bool) -> None:
        """Gate new-tab commands while retaining read-only browsing and local pins."""
        self._browser.set_writable(writable=writable)


class ResultsBrowser(QWidget):
    """Browse delivered index snapshots without changing the opened entry or point.

    index is the initial snapshot, often empty. present_index applies later
    snapshots from the same scan; only delivered runs are browsable or openable.
    open_run accepts a resolved adapter name and native data path, returning the
    existing Load outcome or raising a user-facing error. resolve_experiment is
    the catalog owner's query, called only when Open is requested. Navigation,
    filters and saved documents use historical strings without catalog access.
    All callbacks run on the Qt owner thread. The owner calls dispose before
    deleting the browser, not when hiding it.
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
        self.setObjectName("results.content")
        self._lifetime = LifetimeScope(self)
        self._index = index
        self._resolve_experiment = resolve_experiment
        self._dates_edited = False
        self.setStyleSheet("""
            QWidget { background: #f6f8fb; color: #1e293b; font-size: 12px; }
            QTreeWidget, QTableWidget, QPlainTextEdit, QLineEdit, QComboBox {
                background: white; border: 1px solid #cbd5e1; border-radius: 3px; }
            QDateEdit:disabled { background: #e2e8f0; color: #94a3b8; border-color: #cbd5e1; }
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
        self._navigation: dict[_Focus, QTreeWidgetItem] = {}
        self._navigation_runs: tuple[ResultRun, ...] | None = None
        layout = QVBoxLayout(self)
        splitter = QSplitter()
        left = QWidget()
        filters = QVBoxLayout(left)
        filters.setContentsMargins(0, 0, 5, 0)
        filters.addLayout(self._build_filters())
        self._tree = QTreeWidget()
        self._tree.setObjectName("results.navigation")
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._navigate)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._run_context_menu)
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
        self.comparison.image_activated.connect(self._show_compared_image)
        self._tabs.addTab(self.comparison, "Comparison")
        self.comparison.pins_changed.connect(self._update_pin_count)
        central.addWidget(self._tabs, 1)
        splitter.addWidget(center)
        splitter.addWidget(self._build_details())
        splitter.setSizes([340, 700, 350])
        layout.addWidget(splitter, 1)
        self._lightbox = ResultLightbox(self)
        self._render()

    def dispose(self) -> None:
        """Retire date edit callbacks before the owner deletes this view (ADR-0067)."""
        self._lifetime.close()

    def present_index(self, index: ResultsIndex) -> None:
        """Show a newer snapshot of the same scan on the Qt owner thread.

        New entries extend the existing sorted rows. Preserve current search,
        filter, focus, pins and command feedback. Commands remain available on
        already-delivered runs.
        """
        if self._index == index:
            return
        self._index = index
        self._gallery.replace_index(index)
        self.comparison.replace_index(index)
        self._refresh_filter_choices()
        self._show_item_issues()
        self._render_filtered(self._filtered(update_feedback=False))

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._lightbox.setGeometry(self.rect())

    def hideEvent(self, a0: QHideEvent | None) -> None:
        """Dismiss the overlay when its owning browser is hidden or closed."""
        self._lightbox.hide()
        super().hideEvent(a0)

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
        self._experiment.currentIndexChanged.connect(self._render)
        row.insertWidget(0, self._experiment, 1)
        layout.addLayout(row)
        self._search = QLineEdit()
        self._search.setObjectName("results.search")
        self._search.setPlaceholderText("Search entries, roles, cfg and analysis")
        self._search.textChanged.connect(self._render)
        layout.addWidget(self._search, 1)
        dates = QHBoxLayout()
        self._dates_enabled = QCheckBox("Filter by date (UTC)")
        self._dates_enabled.setObjectName("results.dates_enabled")
        dates.addWidget(self._dates_enabled)
        self._date_from = self._date_filter("results.date_from", "2000-01-01", dates)
        dates.addWidget(QLabel("to"))
        self._date_to = self._date_filter(
            "results.date_to", QDate.currentDate().toString("yyyy-MM-dd"), dates
        )
        self._refresh_filter_choices()
        self._dates_enabled.toggled.connect(self._toggle_dates)
        layout.addLayout(dates)
        self._visible_count = QLabel()
        self._visible_count.setObjectName("results.visible_count")
        make_label_selectable(self._visible_count)
        layout.addWidget(self._visible_count)
        self._issues_label = QLabel()
        self._issues_label.setObjectName("results.issues")
        self._issues_label.setWordWrap(True)
        make_label_selectable(self._issues_label)
        layout.addWidget(self._issues_label)
        self._show_item_issues()
        return layout

    def _show_item_issues(self) -> None:
        issues = self._index.issues
        entries = ", ".join(sorted({issue.entry or "Database" for issue in issues}))
        self._issues_label.setText(f"Cannot index: {entries}")
        self._issues_label.setToolTip(
            "\n".join(
                f"{issue.entry or 'Database'}: {issue.path}: {issue.reason}"
                for issue in issues
            )
        )
        self._issues_label.setVisible(bool(issues))

    def _refresh_filter_choices(self) -> None:
        selected = self._experiment.currentText()
        names = sorted({self._experiment_name(run) for run in self._index.runs})
        with QSignalBlocker(self._experiment):
            self._experiment.clear()
            self._experiment.addItem("All experiments")
            self._experiment.addItems(names)
            self._experiment.setCurrentText(selected)
        if not self._dates_edited and not self._dates_enabled.isChecked():
            dates = [run.at[:10] for run in self._index.runs if run.at]
            if dates:
                for field, value in (
                    (self._date_from, min(dates)),
                    (self._date_to, max(dates)),
                ):
                    with QSignalBlocker(field):
                        field.setDate(QDate.fromString(value, "yyyy-MM-dd"))

    def _date_filter(self, name: str, initial: str, layout: QHBoxLayout) -> QDateEdit:
        edit = _CalendarDateEdit(QDate.fromString(initial, "yyyy-MM-dd"))
        edit.setObjectName(name)
        edit.setDisplayFormat("yyyy-MM-dd")
        edit.setCalendarPopup(True)
        edit.setEnabled(False)
        edit.setToolTip(
            "Enable Filter by date (UTC), then click the date text or right-hand arrow "
            "to choose a date. From / To are inclusive; runs without a UTC time are excluded."
        )
        self._lifetime.connect(edit.dateChanged, self._date_changed)
        layout.addWidget(edit)
        return edit

    def _date_changed(self) -> None:
        self._dates_edited = True
        self._render()

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
        self._open_button = QPushButton("Load into analysis tab")
        self._open_button.clicked.connect(self._open_selected)
        layout.addWidget(self._open_button)
        layout.addWidget(self._current_figure)
        self._feedback = QLabel("")
        self._feedback.setObjectName("results.feedback")
        make_label_selectable(self._feedback)
        self._feedback.setWordWrap(True)
        layout.addWidget(self._feedback)
        tabs = QTabWidget()
        tabs.setObjectName("results.detail_tabs")
        self._details = ResultsDetailsTree()
        self._details.setObjectName("results.details")
        tabs.addTab(self._details, "Metadata / cfg")
        self._analysis = ResultsDetailsTree()
        self._analysis.setObjectName("results.analysis")
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
        return run.experiment or "Unknown experiment"

    def _filtered(self, *, update_feedback: bool = True) -> tuple[ResultRun, ...]:
        enabled = self._dates_enabled.isChecked()
        lower = self._date_from.date().toString("yyyy-MM-dd") if enabled else ""
        upper = self._date_to.date().toString("yyyy-MM-dd") if enabled else ""
        invalid_dates = lower > upper
        if update_feedback:
            self._feedback.setText(
                "Invalid date filter: From date must not follow To date"
                if invalid_dates
                else ""
            )
        if invalid_dates:
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
        self._render_filtered(self._filtered())

    def _render_filtered(self, filtered: tuple[ResultRun, ...]) -> None:
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
        previous = self._navigation_runs
        if previous != runs:
            if (
                previous
                and len(previous) < len(runs)
                and previous[-1].entry < runs[len(previous)].entry
                and runs[: len(previous)] == previous
            ):
                # A new entry has its own branches. Keep existing branches and
                # their expansion state instead of rebuilding the tree per batch.
                self._add_navigation(runs[len(previous) :], set())
            else:
                expanded = {
                    focus
                    for focus, item in self._navigation.items()
                    if item.isExpanded()
                }
                self._tree.clear()
                self._navigation.clear()
                self._add_navigation(runs, expanded)
            self._navigation_runs = runs
        selected = self._navigation.get(replace(self._focus, figure=None))
        if selected is None:
            self._tree.clearSelection()
        else:
            self._tree.setCurrentItem(selected)

    def _add_navigation(
        self, runs: tuple[ResultRun, ...], expanded: set[_Focus]
    ) -> None:
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
                parent = nodes[focus]
                counts[focus] = counts.get(focus, 0) + 1
        for focus, item in nodes.items():
            if focus.run_key is None:
                item.setText(0, f"{item.text(0)}  ({counts[focus]})")
        self._navigation.update(nodes)

    def _navigate(self, item: QTreeWidgetItem, _column: int) -> None:
        for focus, node in self._navigation.items():
            if node is item:
                self._visit(focus)
                break

    def _run_context_menu(self, point: QPoint) -> None:
        item = self._tree.itemAt(point)
        focus = next(
            (focus for focus, node in self._navigation.items() if node is item), None
        )
        if focus is None or focus.run_key is None:
            return
        run = self._index.get_run(focus.run_key)
        menu = QMenu(self._tree)
        pin = QAction("Add to comparison", menu)
        load = QAction("Load into analysis tab", menu)
        menu.addActions([pin, load])
        reason = self._load_unavailable_reason(run)
        load.setEnabled(not reason)
        load.setToolTip(reason or _LOAD_TOOLTIP)
        viewport = self._tree.viewport()
        assert viewport is not None
        selected = menu.exec(viewport.mapToGlobal(point))
        if selected is pin:
            self.comparison.pin_run(run)
            self._tabs.setCurrentIndex(1)
        elif selected is load:
            self._load_run(run)
        menu.deleteLater()

    def _activate_figure(self, run: ResultRun, figure: str) -> None:
        already_in_run = self._focus.run_key == run.key
        self._visit(
            _Focus(
                entry=run.entry,
                experiment=self._experiment_name(run),
                point=(run.entry, run.point),
                run_key=run.key,
                figure=figure,
            )
        )
        if already_in_run:
            self._lightbox.present(
                self._gallery.original_pixmap(run, figure),
                f"{run.run_path.name[-6:]} / {figure}",
            )

    def _render_details(self, runs: tuple[ResultRun, ...]) -> None:
        run = (
            self._index.get_run(self._focus.run_key)
            if self._focus.run_key and runs
            else None
        )
        self._details.present_metadata(self._index, run)
        self._analysis.present_analysis(self._index, run)
        reason = self._load_unavailable_reason(run)
        self._open_button.setEnabled(not reason)
        self._open_button.setToolTip(reason or _LOAD_TOOLTIP)
        self._current_figure.setText(f"Current figure: {self._focus.figure or 'none'}")

    def _load_unavailable_reason(self, run: ResultRun | None) -> str:
        if not self._writable:
            return "Local control is required to open an analysis tab"
        if run is None:
            return "Select a run"
        return "No data.h5 in this saved run" if run.data_path is None else ""

    def _show_compared_image(self, key: str, figure: str, raster: QPixmap) -> None:
        run = self._index.get_run(key)
        self._lightbox.present(raster, f"{run.run_path.name[-6:]} / {figure}")

    def _open_compared(self, key: str) -> None:
        run = self._index.get_run(key)
        self._visit(
            _Focus(
                entry=run.entry,
                experiment=self._experiment_name(run),
                point=(run.entry, run.point),
                run_key=key,
            )
        )
        self._tabs.setCurrentIndex(0)

    def _update_pin_count(self) -> None:
        count = len(self.comparison.pinned_runs())
        self._tabs.setTabText(1, f"Comparison ({count})" if count else "Comparison")

    def _open_selected(self) -> None:
        if self._focus.run_key is None:
            return
        self._load_run(self._index.get_run(self._focus.run_key))

    def _load_run(self, run: ResultRun) -> None:
        try:
            resolution = self._resolve_experiment(run.experiment)
            if resolution.adapter_name is None:
                self._feedback.setText(f"Cannot open analysis tab: {resolution.reason}")
                return
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
            self.comparison.pin_run(run)
            self._tabs.setCurrentIndex(1)
            QApplication.processEvents()
            self.comparison.reveal_figure(figure)
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
