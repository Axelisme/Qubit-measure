"""AnalyzePanelWidget — the v2 analysis panel: Filter -> Search -> Show stages.

One panel (a MainWindow singleton) gathers the three post-selection analysis
steps under a stage bar ``1 Filter -> 2 Search -> 3 Show``. A stage is locked
until the earlier ones are done: Search until every sampled spectrum has a
published Filter mask, Show until Search is unlocked and a search result
exists, and both while a Filter draft is live. A running search's Cancel
ignores the locks.

- **Filter**: the published masks read-only, or the live cross-spectrum
  selector over the joint point cloud (``FilterStage``).
- **Search**: the database-search form + the search's native diagnostic figure.
- **Show**: the fit visualisation (background / simulation lines / points /
  constant-freq lines), with display tools — y/x axis limits (default matching
  the notebook's ``auto_derive_limits``), r_f / sample_f reference-line toggles,
  and a transition display subset.

The search runs on a worker thread. After recording its numeric result, the main
thread renders its diagnostic figure into a FigureContainer. Plot failures are
reported separately and do not invalidate the fit. Search problems (a missing
database, a rejected search) go to the Search status line and Show drawing
problems to the Show status line; neither opens a modal.

Public Qt seams (objectName): the stage buttons ``fluxdep.fit.stage.filter`` /
``search`` / ``show``, ``fluxdep.fit.search_db``, ``fluxdep.fit.bounds_preset``,
``fluxdep.fit.r_f`` / ``fluxdep.fit.sample_f`` with their ``.required`` marks,
``fluxdep.fit.search`` / ``cancel`` / ``export_params``, ``fluxdep.fit.status``,
``fluxdep.fit.freshness``; ``fluxdep.show.apply`` / ``auto_limits``,
``fluxdep.show.status``, the Show canvas ``fluxdep.show.canvas`` and its
navigation ``fluxdep.show.toolbar``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Literal

from matplotlib.backend_bases import Event
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import DatabaseSearchResult
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import (
    FitChangedPayload,
    InteractiveChangedPayload,
    SearchChangedPayload,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.search import SEARCH_OWNER_ID
from zcu_tools.gui.app.fluxdep.services.viz import FitDisplay, format_stale_reasons
from zcu_tools.gui.app.fluxdep.state import (
    FIT_VERSION_KEY,
    FitState,
    transitions_need_r_f,
    transitions_need_sample_f,
)
from zcu_tools.gui.app.fluxdep.ui.filter_stage import FilterStage
from zcu_tools.gui.app.fluxdep.ui.stage_bar import StageBar, stage_status
from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.plotting.figures import NamedFigures
from zcu_tools.plotting.fluxdep import make_search_diagnostic_figure
from zcu_tools.plotting.plots import Plots

from .error_messages import friendly_fit_message
from .frequency_field import FrequencyField
from .transitions_form import TransitionsForm

logger = logging.getLogger(__name__)

AnalyzeTab = Literal["filter", "search", "show"]
_ANALYZE_TABS: tuple[AnalyzeTab, ...] = ("filter", "search", "show")

# One Show drawing: the fit result version, every published source/graph
# version the renderer reads (stale reasons derive from them), and the
# applied display choices. Pending control edits are not part of it.
_ShowKey = tuple[int, tuple[tuple[str, int], ...], FitDisplay]


# (EJb, ECb, ELb) bound presets — the notebook's named fitting ranges (GHz),
# in the order the Bounds preset combo lists them.
_BOUND_PRESETS: dict[str, tuple[tuple[float, float], ...]] = {
    "general": ((2.0, 15.0), (0.2, 2.0), (0.1, 2.0)),
    "integer": ((3.0, 6.0), (0.8, 2.0), (0.08, 0.2)),
    "all": ((1.0, 20.0), (0.1, 4.0), (0.01, 3.0)),
}
# Preset combo entry shown while the bounds match no preset; choosing it keeps them.
_CUSTOM_PRESET = "custom"


class _ShowToolbar(NavigationToolbar2QT):
    """Matplotlib's Qt toolbar with only the view-navigation tools.

    Subplots, Customize and Save open blocking dialogs, which fluxdep dialogs
    avoid; matplotlib chooses the buttons from this class attribute.
    """

    toolitems = [
        item
        for item in NavigationToolbar2QT.toolitems
        if item[0] in {"Home", "Back", "Forward", "Pan", "Zoom"}
    ]


def _bound_spinbox(value: float, hi: float = 1000.0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-hi, hi)
    box.setDecimals(3)
    box.setSingleStep(0.1)
    box.setValue(value)
    return box


def _message_label(accessible_name: str, object_name: str) -> QLabel:
    """A selectable, wrapping message label that never widens the form (Ignored horizontal)."""
    label = QLabel("")
    make_label_selectable(label)
    label.setAccessibleName(accessible_name)
    label.setObjectName(object_name)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setWordWrap(True)
    return label


def _control_column(*widgets: QWidget) -> QWidget:
    """Stack widgets at the top of a column; spare height stays below them."""
    column = QWidget()
    layout = QVBoxLayout(column)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    column.setMaximumWidth(380)
    return column


def _published_inputs(
    fit: FitState,
) -> tuple[
    str,
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    TransitionDict,
    float | None,
    float | None,
]:
    """The search inputs the forms display, detached from results and captures.

    Result publication changes params, quality, the recorded cloud and the
    captured graph, none of which is an input the user edits here.
    """
    return (
        fit.search_db,
        fit.EJb,
        fit.ECb,
        fit.ELb,
        deepcopy(fit.transitions),
        fit.r_f,
        fit.sample_f,
    )


class AnalyzePanelWidget(QWidget):
    """Filter -> Search -> Show stages over the selected joint point cloud."""

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._writable = True
        self._bus_subs = EventSubscriptions()
        self._shown_token: int | None = None
        self._tab: AnalyzeTab = "filter"
        self._shown_fit: _ShowKey | None = None
        self._db_chooser: QFileDialog | None = None
        # True while the entered transitions need a blank r_f / sample_f.
        self._needs_frequency = False
        # Display choices taken by Apply; the controls may hold pending edits.
        self._applied_display = FitDisplay()

        self._build_ui()
        self._plot_host = QtPlotHost(self._diag_container, QtOwnerScheduler())
        self._diagnostics = Plots(self._plot_host)
        self._load_from_state()
        self._bus_subs.subscribe(ctrl.bus, FitChangedPayload, self._on_fit_changed)
        self._bus_subs.subscribe(
            ctrl.bus, SearchChangedPayload, self._on_search_changed
        )
        self._bus_subs.subscribe(
            ctrl.bus, InteractiveChangedPayload, self._on_interactive_changed
        )
        # Every fit upstream: spectrum collection, alignment/points, Filter masks.
        for upstream in (
            SpectrumAddedPayload,
            SpectrumRemovedPayload,
            SpectrumChangedPayload,
            SelectionChangedPayload,
        ):
            self._bus_subs.subscribe(ctrl.bus, upstream, self._on_upstream_changed)
        progress = ctrl.search.progress_control
        self._progress_dispose = (
            progress.attach_progress(SEARCH_OWNER_ID, self._refresh_progress)
            if progress is not None
            else None
        )
        self._refresh_search()
        if self._ctrl.state.fit.has_result:
            self._apply_auto_limits()
            self._apply_limit_boxes()
        self._refresh_show()
        self.set_writable(writable=self._writable)
        # MainWindow routes every Fit page entry through show_tab.

    # --- construction ----------------------------------------------------

    @property
    def figures(self) -> Mapping[str, Figure]:
        """Current diagnostic figures, independent of their Qt presentation."""
        return NamedFigures(self._diagnostics)

    def release_figures(self) -> None:
        """Detach presentation while keeping retained figures saveable."""
        self._diagnostics.finish(present=False)
        self._diagnostics.release()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        # Clicks reach only unlocked stages; StageBar disables locked buttons.
        self._stages: StageBar[AnalyzeTab] = StageBar(
            "fluxdep.fit.stage",
            (("filter", "Filter"), ("search", "Search"), ("show", "Show")),
            on_click=self.show_tab,
        )
        root.addWidget(self._stages)
        self._filter = FilterStage(self._ctrl)
        self._stage_pages: dict[AnalyzeTab, QWidget] = {
            "search": self._build_search_tab(),
            "show": self._build_show_tab(),
            "filter": self._filter,
        }
        self._pages = QStackedWidget()
        for stage_page in self._stage_pages.values():
            self._pages.addWidget(stage_page)
        self._pages.setCurrentWidget(self._filter)
        root.addWidget(self._pages, stretch=1)

    def _refresh_stages(self) -> None:
        """Mark each stage from published State and the live Filter draft.

        Filter is done once every sampled spectrum has a published mask
        (upstream changes drop masks, which reopens it); Search is done once
        a search result exists. Search needs a done Filter and no live Filter
        draft; Show needs an unlocked Search and a result, so an old result
        stays hidden behind a locked Search. A locked stage stays shown if it
        is current; its form (Search, with Search and Export) or display
        choices (Show) are disabled until it unlocks, and always for a
        non-holder. A running search's Cancel and Show's toolbar ignore the
        locks (see _refresh_cancel_button; the toolbar only moves the local
        view).
        """
        state = self._ctrl.state
        drafting = self._ctrl.interactive.current_cross_selection() is not None
        sampled = [e for e in state.spectrums.values() if e.point_count > 0]
        filtered = bool(sampled) and all(e.filter_mask is not None for e in sampled)
        has_result = state.fit.has_result
        search_locked = drafting or not filtered
        filter_status = stage_status(done=filtered, draft=drafting, locked=False)
        search_status = stage_status(done=has_result, draft=False, locked=search_locked)
        show_status = stage_status(
            done=False, draft=False, locked=search_locked or not has_result
        )
        self._stages.set_status("filter", filter_status)
        self._stages.set_status("search", search_status)
        self._stages.set_status("show", show_status)
        self._stages.set_current(self._tab)
        self._search_form.setEnabled(self._writable and search_status != "locked")
        self._display_box.setEnabled(self._writable and show_status != "locked")

    # --- Search tab ------------------------------------------------------

    def _build_search_tab(self) -> QWidget:
        from qtpy.QtWidgets import QSplitter

        form_box = self._build_search_form()
        # Cancel sits outside the form, so the Search stage lock that disables
        # the form never reaches it; _refresh_cancel_button gates it instead.
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("fluxdep.fit.cancel")
        self._cancel_btn.clicked.connect(self._on_cancel_search)
        self._cancel_btn.setEnabled(False)
        # Status lines sit outside the form, so a non-holder reads them enabled.
        self._status = _message_label("Status", "fluxdep.fit.status")
        # Outdated-fit line with its root-change reasons; blank while fresh.
        self._freshness = _message_label("Freshness", "fluxdep.fit.freshness")

        # Right: the search's native diagnostic figure, explicitly presented.
        self._diag_stack = QStackedWidget()
        diag_placeholder = QLabel("Search to see the diagnostic plot.")
        diag_placeholder.setEnabled(False)
        self._diag_stack.addWidget(diag_placeholder)
        self._diag_container = FigureContainer(self._diag_stack, diag_placeholder)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(
            _control_column(form_box, self._cancel_btn, self._status, self._freshness)
        )
        splitter.addWidget(self._diag_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 640])
        holder = QWidget()
        QVBoxLayout(holder).addWidget(splitter)
        return holder

    def _build_search_form(self) -> QGroupBox:
        form_box = QGroupBox("Search parameters")
        form = QFormLayout(form_box)
        self._search_form = form_box

        self._db_edit = QLineEdit()
        self._db_edit.setObjectName("fluxdep.fit.search_db")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_db)
        db_row = QHBoxLayout()
        db_row.setContentsMargins(0, 0, 0, 0)
        db_row.addWidget(self._db_edit, stretch=1)
        db_row.addWidget(browse)
        db_holder = QWidget()
        db_holder.setLayout(db_row)
        form.addRow("Search DB", db_holder)

        self._preset = QComboBox()
        self._preset.setObjectName("fluxdep.fit.bounds_preset")
        self._preset.addItems([*_BOUND_PRESETS, _CUSTOM_PRESET])
        self._preset.activated.connect(self._on_preset_selected)
        form.addRow("Bounds preset", self._preset)

        self._ej_lo, self._ej_hi = _bound_spinbox(2.0), _bound_spinbox(15.0)
        self._ec_lo, self._ec_hi = _bound_spinbox(0.2), _bound_spinbox(2.0)
        self._el_lo, self._el_hi = _bound_spinbox(0.1), _bound_spinbox(2.0)
        form.addRow("EJ (GHz)", self._bound_row(self._ej_lo, self._ej_hi))
        form.addRow("EC (GHz)", self._bound_row(self._ec_lo, self._ec_hi))
        form.addRow("EL (GHz)", self._bound_row(self._el_lo, self._el_hi))
        for box in self._bound_boxes():
            box.valueChanged.connect(self._sync_preset)

        self._add_frequency_rows(form)

        self._transitions_form = TransitionsForm()
        form.addRow(QLabel("Transitions"))
        form.addRow(self._transitions_form)
        self._transitions_form.changed.connect(self._refresh_frequency_marks)
        for field in (self._r_f, self._sample_f):
            field.on_change(self._refresh_frequency_marks)

        self._add_search_actions(form)
        return form_box

    def _add_frequency_rows(self, form: QFormLayout) -> None:
        """Add the r_f / sample_f fields, each with its hidden required mark.

        Both are optional (blank = unset). A field is marked while the entered
        transitions need it, and Search stays disabled until it is filled; a
        submitted search still leaves validation to the Controller.
        """
        self._r_f = FrequencyField(
            "fluxdep.fit.r_f", "Needed by the sideband transitions."
        )
        self._sample_f = FrequencyField(
            "fluxdep.fit.sample_f", "Needed by the mirror transitions."
        )
        form.addRow("r_f (GHz)", self._r_f)
        form.addRow("sample_f (GHz)", self._sample_f)

    def _add_search_actions(self, form: QFormLayout) -> None:
        """Add Search, the progress bar and Export params…."""
        self._search_btn = QPushButton("Search")
        self._search_btn.setObjectName("fluxdep.fit.search")
        self._search_btn.clicked.connect(self._on_search)
        form.addRow(self._search_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        form.addRow(self._progress)

        self._export_btn = QPushButton("Export params…")
        self._export_btn.setObjectName("fluxdep.fit.export_params")
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setEnabled(False)
        form.addRow(self._export_btn)

    # --- Show tab --------------------------------------------------------

    def _build_show_tab(self) -> QWidget:
        from qtpy.QtWidgets import QSplitter

        tools_box = QGroupBox("Display")
        self._display_box = tools_box
        tools = QFormLayout(tools_box)

        self._x_lo, self._x_hi = _bound_spinbox(0.0), _bound_spinbox(1.0)
        self._y_lo, self._y_hi = _bound_spinbox(0.0), _bound_spinbox(10.0)
        tools.addRow("x limits", self._bound_row(self._x_lo, self._x_hi))
        tools.addRow("y limits", self._bound_row(self._y_lo, self._y_hi))
        auto_btn = QPushButton("Auto limits")
        auto_btn.setObjectName("fluxdep.show.auto_limits")
        auto_btn.clicked.connect(self._apply_auto_limits)
        tools.addRow(auto_btn)

        self._show_const_freq = QCheckBox("r_f / sample_f reference lines")
        self._show_const_freq.setChecked(True)
        tools.addRow(self._show_const_freq)

        self._transitions_show = TransitionsForm()
        tools.addRow(QLabel("Transitions to show"))
        tools.addRow(self._transitions_show)

        redraw = QPushButton("Apply")
        redraw.setObjectName("fluxdep.show.apply")
        redraw.clicked.connect(self._apply_display)
        tools.addRow(redraw)

        # Show's drawing problems; the Search status line stays Search's.
        self._show_status = _message_label("Show status", "fluxdep.show.status")

        self._fit_figure = Figure(figsize=(6, 4))
        self._fit_canvas = FigureCanvasQTAgg(self._fit_figure)
        self._fit_canvas.setObjectName("fluxdep.show.canvas")
        # Show lays out once per draw; re-fit on resize so the title and axis
        # labels stay inside the canvas Qt actually gives the figure.
        self._fit_canvas.mpl_connect("resize_event", self._fit_show_layout)
        # View navigation only changes this canvas, so it stays usable for
        # every viewer; set_writable never gates it.
        self._fit_toolbar = _ShowToolbar(self._fit_canvas, None)
        self._fit_toolbar.setObjectName("fluxdep.show.toolbar")
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self._fit_toolbar)
        plot_layout.addWidget(self._fit_canvas, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(_control_column(tools_box, self._show_status))
        splitter.addWidget(plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 640])
        holder = QWidget()
        QVBoxLayout(holder).addWidget(splitter)
        return holder

    # --- helpers ---------------------------------------------------------

    def _bound_boxes(self) -> tuple[QDoubleSpinBox, ...]:
        """The Search EJ/EC/EL lower and upper bound boxes, in that order."""
        return (
            self._ej_lo,
            self._ej_hi,
            self._ec_lo,
            self._ec_hi,
            self._el_lo,
            self._el_hi,
        )

    def _bound_row(self, lo: QDoubleSpinBox, hi: QDoubleSpinBox) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(lo)
        row.addWidget(QLabel("–"))
        row.addWidget(hi)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    def _load_from_state(self) -> None:
        from .paths import default_database_file

        fit = self._ctrl.state.fit
        # Seed the database path: the prior fit's path, else a bundled default so
        # the user usually doesn't have to browse at all.
        self._db_edit.setText(
            fit.search_db
            or default_database_file(
                self._ctrl.state.project, self._ctrl.get_project_root()
            )
        )
        self._ej_lo.setValue(fit.EJb[0])
        self._ej_hi.setValue(fit.EJb[1])
        self._ec_lo.setValue(fit.ECb[0])
        self._ec_hi.setValue(fit.ECb[1])
        self._el_lo.setValue(fit.ELb[0])
        self._el_hi.setValue(fit.ELb[1])
        self._r_f.set_value(fit.r_f)
        self._sample_f.set_value(fit.sample_f)
        self._transitions_form.set_transitions(fit.transitions)
        self._transitions_show.set_transitions(fit.transitions)
        self._sync_preset()
        self._refresh_frequency_marks()
        self._applied_display = self._display_from_controls()
        self._export_btn.setEnabled(fit.has_result)
        # Compare published inputs, not results, to preserve local display edits.
        self._loaded_fit_inputs = _published_inputs(fit)

    def _on_fit_changed(self, payload: FitChangedPayload) -> None:
        del payload
        if _published_inputs(self._ctrl.state.fit) != self._loaded_fit_inputs:
            self._load_from_state()
        self._refresh_search()
        self._refresh_show()
        self._refresh_stages()

    def _on_upstream_changed(self, payload: object) -> None:
        """Re-read freshness and stage locks after a publication fit depends on."""
        del payload
        self._refresh_search()
        self._refresh_show()
        self._refresh_stages()

    @property
    def current_tab(self) -> AnalyzeTab:
        """Current stage: filter, search, or show, on the State owner."""
        self._ctrl.state.assert_owner_thread()
        return self._tab

    def show_tab(self, tab: AnalyzeTab) -> None:
        """Show and refresh a stage on the State-owner Qt thread.

        Locks gate the stage bar's clicks; this entry still shows a locked
        stage, as window routing and agent following need, with its inputs
        disabled. Filter reattaches a live
        context's Selector without resetting input/Undo, else shows the
        published masks; it never begins a context. Search/show detach the
        Filter view and never cancel its context. Search forms/progress refresh
        from the app owner. Invalid tab raises ValueError; foreign-thread
        access raises RuntimeError.
        This does not raise or activate the top-level window.
        """
        self._ctrl.state.assert_owner_thread()
        if tab not in _ANALYZE_TABS:
            raise ValueError(f"unknown analyze tab {tab!r}")
        self._tab = tab
        self._pages.setCurrentWidget(self._stage_pages[tab])
        if tab == "filter":
            self._filter.attach()
        else:
            self._filter.detach()
        self._refresh_stages()
        self._refresh_search()

    def _on_interactive_changed(self, payload: InteractiveChangedPayload) -> None:
        # The Filter stage follows its own view; opened/closed drafts move locks.
        if payload.kind == "selection" and payload.phase != "updated":
            self._refresh_stages()

    def activate(self) -> None:
        """Refresh the singleton; on Filter, rebuild the stage's view.

        Explicit refresh; MainWindow routes Fit page entries through show_tab.
        Retire the old view first, so hidden controls cannot Apply stale data.
        It never begins a Filter context; invalid inputs propagate.
        """
        self._refresh_search()
        if self._tab == "filter":
            self._filter.detach()
            self._filter.attach()
        self._refresh_stages()

    def detach(self) -> None:
        """Retire Filter's view without cancelling the app context.

        Call when leaving the Fit page. Other stages/forms remain intact.
        The app owns context invalidation on active/source/picker switches.
        """
        self._filter.detach()

    def set_writable(self, *, writable: bool) -> None:
        """Gate domain input and Show's display choices; viewing stays available.

        A non-holder keeps the stage bar, the status lines, the canvases and
        the Show toolbar; Filter's Edit/Discard and Selector controls follow.
        The holder's Search form and Show display choices also follow the
        stage locks (see _refresh_stages); a running search's Cancel does not.
        """
        self._writable = writable
        self._filter.set_writable(writable=writable)
        self._refresh_stages()
        self._refresh_cancel_button()

    # --- Search actions --------------------------------------------------

    @staticmethod
    def _show_subset_notice(
        transitions: TransitionDict,
        r_f: float | None,
        sample_f: float | None,
    ) -> str | None:
        """The Show notice when the chosen subset needs an unset r_f/sample_f.

        None when the subset can be drawn with the recorded frequencies;
        otherwise Show draws the fit transitions and reports this notice.
        """
        fallback = "Showing the fit transitions instead."
        if transitions_need_r_f(transitions) and r_f is None:
            return (
                "The transitions include a blue/red-side or mirror category, "
                f"which needs r_f. {fallback}"
            )
        if transitions_need_sample_f(transitions) and sample_f is None:
            return (
                "The transitions include a mirror category, which needs "
                f"sample_f. {fallback}"
            )
        return None

    def _commit_params(self) -> None:
        transitions = self._transitions_form.get_transitions()
        r_f = self._r_f.value()
        sample_f = self._sample_f.value()
        self._ctrl.set_fit_params(
            search_db=self._db_edit.text().strip(),
            EJb=(self._ej_lo.value(), self._ej_hi.value()),
            ECb=(self._ec_lo.value(), self._ec_hi.value()),
            ELb=(self._el_lo.value(), self._el_hi.value()),
            transitions=transitions,
            r_f=r_f,
            sample_f=sample_f,
        )

    def _on_preset_selected(self, index: int) -> None:
        """Fill the bound boxes from the chosen preset; custom keeps them."""
        preset = _BOUND_PRESETS.get(self._preset.itemText(index))
        if preset is None:
            self._sync_preset()
            return
        values = [bound for pair in preset for bound in pair]
        for box, value in zip(self._bound_boxes(), values, strict=True):
            box.setValue(value)

    def _sync_preset(self) -> None:
        """Name the preset the bound boxes hold exactly, else custom."""
        lo_hi = [box.value() for box in self._bound_boxes()]
        shown = (
            (lo_hi[0], lo_hi[1]),
            (lo_hi[2], lo_hi[3]),
            (lo_hi[4], lo_hi[5]),
        )
        name = next(
            (name for name, preset in _BOUND_PRESETS.items() if preset == shown),
            _CUSTOM_PRESET,
        )
        self._preset.setCurrentText(name)

    def _refresh_frequency_marks(self) -> None:
        """Mark r_f / sample_f while the entered transitions need a blank one.

        A shown mark also disables Search: that search could only be rejected,
        and its parameter replacement would clear the recorded result.
        """
        try:
            transitions = self._transitions_form.get_transitions()
        except ValueError:
            # Malformed pairs are reported when Search parses them; until then
            # no category counts as entered.
            transitions = TransitionDict()
        needs_r_f = self._r_f.mark_needed(needed=transitions_need_r_f(transitions))
        needs_sample_f = self._sample_f.mark_needed(
            needed=transitions_need_sample_f(transitions)
        )
        self._needs_frequency = needs_r_f or needs_sample_f
        self._refresh_search_button()

    def _refresh_search_button(self) -> None:
        """Enable Search unless a search is pending or a needed frequency is blank."""
        activity = self._ctrl.search.current
        pending = activity is not None and activity.status == "pending"
        self._search_btn.setEnabled(not pending and not self._needs_frequency)

    def _refresh_cancel_button(self) -> None:
        """Enable Cancel for the holder while a search is pending.

        Only control authority and the running search gate it; the stage
        locks do not, so a running search can always be stopped.
        """
        activity = self._ctrl.search.current
        pending = activity is not None and activity.status == "pending"
        self._cancel_btn.setEnabled(self._writable and pending)

    def _on_browse_db(self) -> None:
        from .paths import database_dir

        # Start in the current path's folder if set, else the project / bundled
        # simulation database directory.
        current = self._db_edit.text().strip()
        start = os.path.dirname(current) if current else ""
        if not start or not os.path.isdir(start):
            start = database_dir(
                self._ctrl.state.project, self._ctrl.get_project_root()
            )
        chooser = QFileDialog(
            self, "Select search DB", start, "HDF5 (*.h5 *.hdf5);;All files (*)"
        )
        chooser.setOption(QFileDialog.Option.DontUseNativeDialog)
        chooser.setFileMode(QFileDialog.FileMode.ExistingFile)
        # Non-blocking open() keeps the Qt event loop (and the control socket)
        # responsive; while it is open the owner turn still sees a Local modal.
        chooser.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        chooser.fileSelected.connect(self._db_edit.setText)
        # Hold a reference so open()'s immediate return doesn't let it be GC'd.
        self._db_chooser = chooser
        chooser.finished.connect(lambda _result=0: setattr(self, "_db_chooser", None))
        chooser.open()

    def _on_search(self) -> None:
        db_path = self._db_edit.text().strip()
        if not db_path:
            self._status.setText("Select a search DB first.")
            return
        if not os.path.isfile(db_path):
            self._status.setText(f"Search DB not found: {db_path}")
            return
        # Search is disabled while a needed r_f / sample_f is blank; any other
        # rejection comes from the Controller and the status line reports it.
        try:
            self._commit_params()
        except ValueError as exc:
            self._status.setText(f"Invalid parameters: {exc}")
            return
        previous = self._ctrl.search.current
        try:
            self._ctrl.search.start()
        except Exception as exc:
            logger.exception("search start failed")
            self._refresh_search()
            # An opened failure already set the status from its activity.
            if self._ctrl.search.current == previous:
                self._status.setText(friendly_fit_message("Search", exc))

    def _on_cancel_search(self) -> None:
        token = self._ctrl.search.active_token
        if token is not None:
            self._ctrl.search.cancel(token)

    def _on_search_changed(self, payload: SearchChangedPayload) -> None:
        del payload
        self._refresh_search()

    def _refresh_search(self) -> None:
        activity = self._ctrl.search.current
        pending = activity is not None and activity.status == "pending"
        self._refresh_search_button()
        self._refresh_cancel_button()
        self._progress.setVisible(pending)
        stale_reasons = self._stale_reasons()
        self._freshness.setText(format_stale_reasons(stale_reasons))
        # Stale params are refused by export; the reasons above say why.
        self._export_btn.setEnabled(
            self._ctrl.state.fit.has_result and not pending and not stale_reasons
        )
        if pending:
            self._status.setText("Searching…")
            self._refresh_progress()
        elif activity is not None:
            if activity.status == "finished":
                result = self._ctrl.search.result
                if result is not None and activity.token != self._shown_token:
                    self._shown_token = activity.token
                    self._present_search_result(result)
            elif activity.status == "failed":
                self._status.setText(
                    friendly_fit_message("Search", activity.error)
                    if activity.error is not None
                    else "Search failed."
                )
            else:
                self._status.setText("Search cancelled.")

    def _refresh_progress(self) -> None:
        progress = self._ctrl.search.progress_control
        bars = progress.progress_bars(SEARCH_OWNER_ID) if progress is not None else ()
        if bars:
            _bar_id, model = bars[-1]
            self._progress.setRange(0, model.qt_maximum())
            self._progress.setValue(model.qt_value())
            self._progress.setFormat(model.format())
        else:
            self._progress.setRange(0, 0)

    def _present_search_result(self, result: DatabaseSearchResult) -> None:
        EJ, EC, EL = result.params
        self._status.setText(f"EJ={EJ:.3f}  EC={EC:.3f}  EL={EL:.3f}")
        self._export_btn.setEnabled(True)
        try:
            self.release_figures()
            self._diagnostics = Plots(self._plot_host)
            self._diagnostics.adopt("diagnostic", make_search_diagnostic_figure(result))
            self._diagnostics.finish()
        except Exception as exc:
            logger.exception("Search diagnostic rendering failed")
            self._show_message(
                "Diagnostic plot failed",
                f"Search result retained; export is available.\n{exc}",
            )
        self._apply_auto_limits()
        self._apply_limit_boxes()
        self._refresh_show()
        # The Show stage is rendered and ready, but stay on Search so the result
        # (EJ/EC/EL + diagnostic) is visible; the user switches to Show when ready.

    def quiesce(self) -> None:
        """Drain the embedded selector only; search belongs to the app runner.

        The host must drain its search runner before disposing this panel.
        Hiding or detaching this view does not cancel app-owned search.
        """
        self._filter.quiesce()

    def dispose(self) -> None:
        """Release subscriptions and presentation after app search has drained."""
        self._bus_subs.unsubscribe_all()
        self._filter.dispose()
        if self._progress_dispose is not None:
            self._progress_dispose()
            self._progress_dispose = None
        self.release_figures()

    def _on_export(self) -> None:
        try:
            path = self._ctrl.export_params()
        except Exception as exc:  # noqa: BLE001 — surface to the panel
            self._status.setText("Export failed.")
            self._status.setToolTip("")
            self._show_message("Export failed", friendly_fit_message("Export", exc))
            return
        self._status.setText(f"Exported → {path}")
        self._status.setToolTip(path)

    def _show_message(self, title: str, message: str) -> None:
        from qtpy.QtWidgets import QMessageBox

        QMessageBox.warning(self, title, message)

    # --- Show actions ----------------------------------------------------

    def _apply_auto_limits(self) -> None:
        """Set the x/y limit boxes to the Controller's Show auto limits."""
        (x_lo, x_hi), (y_lo, y_hi) = self._ctrl.fit_auto_limits()
        self._x_lo.setValue(x_lo)
        self._x_hi.setValue(x_hi)
        self._y_lo.setValue(y_lo)
        self._y_hi.setValue(y_hi)

    def _stale_reasons(self) -> tuple[str, ...]:
        """State's root-change reasons for the stored result; empty while fresh."""
        return tuple(self._ctrl.state.fit_stale_reasons)

    def _display_from_controls(self) -> FitDisplay:
        """Snapshot every Show control as detached display choices."""
        return FitDisplay(
            xlim=(self._x_lo.value(), self._x_hi.value()),
            ylim=(self._y_lo.value(), self._y_hi.value()),
            show_const_freq=self._show_const_freq.isChecked(),
            plot_transitions=deepcopy(self._transitions_show.get_transitions()),
        )

    def _apply_display(self) -> None:
        """Apply: take every pending Show control and redraw with it."""
        self._applied_display = self._display_from_controls()
        self._redraw_show()

    def _apply_limit_boxes(self) -> None:
        """Take only the limit boxes (e.g. auto limits); other edits stay pending."""
        boxes = self._display_from_controls()
        self._applied_display = replace(
            self._applied_display, xlim=boxes.xlim, ylim=boxes.ylim
        )

    def _show_key(self) -> _ShowKey:
        state = self._ctrl.state
        return (
            state.version.get(FIT_VERSION_KEY),
            tuple((stage.name, stage.version) for stage in state.pipeline_snapshot()),
            self._applied_display,
        )

    def _refresh_show(self) -> None:
        """Redraw Show with the applied choices when result, sources or they changed.

        Pending control edits (transition subset, reference lines, limits) stay
        in the controls until Apply.
        """
        if self._ctrl.state.fit.params is None:
            return
        if self._show_key() != self._shown_fit:
            self._redraw_show()

    def _fit_show_layout(self, _event: Event) -> None:
        self._fit_figure.tight_layout()

    def _redraw_show(self) -> None:
        """Render the fit visualisation with the applied display choices."""
        fit = self._ctrl.state.fit
        if fit.params is None:
            return
        display = self._applied_display
        # The show-transitions subset may need an r_f/sample_f that's unset;
        # check before drawing so it surfaces a friendly message, not a crash.
        show_transitions = (
            display.plot_transitions
            if display.plot_transitions is not None
            else fit.transitions
        )
        notice = self._show_subset_notice(show_transitions, fit.r_f, fit.sample_f)
        if notice is not None:
            show_transitions = fit.transitions

        try:
            self._ctrl.render_fit(
                self._fit_figure,
                xlim=display.xlim,
                ylim=display.ylim,
                show_const_freq=display.show_const_freq,
                plot_transitions=show_transitions,
            )
        except Exception as exc:
            logger.exception("render_fit_figure failed")
            self._show_status.setText(f"Could not draw: {exc}")
            return
        self._show_status.setText(notice or "")
        self._shown_fit = self._show_key()
        # A new drawing starts a new navigation history: Home returns to it.
        self._fit_toolbar.update()
        self._fit_canvas.draw_idle()
