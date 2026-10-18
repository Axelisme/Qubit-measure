"""MainWindow — the fluxdep-gui shell.

  ┌───────────────────────────────────────────────────────────────┐
  │ Project: chip / qubit  [Local]       Project… Add spectrum…   │
  │                                      Restore… Export spectra… │
  ├───────────────────────────────────────────────────────────────┤
  │ [Spectra] [Fit]                                               │
  │ ┌────────────┬──────────────────────────────┐                 │
  │ │ spectra    │ [1 Align ✓] → [2 Sample]     │  Spectra page   │
  │ │  • a ...   │ editing area (stacked): the  │                 │
  │ │  • b ...   │ active spectrum's picker or  │                 │
  │ │ [Remove]   │ result preview               │                 │
  │ └────────────┴──────────────────────────────┘                 │
  └───────────────────────────────────────────────────────────────┘

The Spectra page shows the active spectrum's published preview or live
draft under a stage bar; clicking a stage opens that stage's picker. A
widget's ``finished`` signal writes its result back through the Controller,
which advances the stage and re-triggers the swap. A GUI Add opens the new
spectrum's line picker. The Fit page hosts the AnalyzePanel singleton
(Filter -> Search -> Show stages). Spectrum work never replaces the Fit page. Pages
switch only on human intent: a tab click, or a GUI Add / Restore (back to
Spectra). Agent-origin events update the content of the page they concern.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyze_panel import AnalyzePanelWidget, AnalyzeTab

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.fluxdep.controller import (
    Controller,
    PendingKind,
    pending_kind,
)
from zcu_tools.gui.app.fluxdep.event_bus import (
    ActiveSpectrumChangedPayload,
    InteractiveChangedPayload,
    ProjectChangedPayload,
    SearchChangedPayload,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.interactive import (
    ActiveInteractiveContext,
    CrossSelectionContext,
    LinePickContext,
    OneTonePickContext,
    TwoTonePickContext,
)
from zcu_tools.gui.app.fluxdep.ui.interactive.base import InteractiveMplWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import ResultPreviewWidget
from zcu_tools.gui.app.fluxdep.ui.spectrum_list import SpectrumListWidget
from zcu_tools.gui.app.fluxdep.ui.stage_bar import StageBar, stage_status
from zcu_tools.gui.control_identity import ControlStatus
from zcu_tools.gui.event_bus import EventMeta, EventSubscriptions
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.project import DEFAULT_CHIP, DEFAULT_QUBIT
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.widgets.control_indicator import ControlIndicator
from zcu_tools.gui.widgets.dialog_presenter import (
    DestructiveConfirmation,
    DialogPresenter,
    QtDialogPresenter,
)
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """The fluxdep analysis window shell."""

    def __init__(
        self,
        ctrl: Controller,
        *,
        interactive_runner: BackgroundRunner | None = None,
        search_runner: BackgroundRunner | None = None,
        dialog_presenter: DialogPresenter | None = None,
    ) -> None:
        """Mount views and retain app-owned interactive/search Qt runners.

        Both runners become child QObjects. Close requests search cancellation
        and refuses disposal until search_runner drains. Interactive disposal
        then precedes its runner drain. Omitted runners support headless
        composition; a caller injecting search background must also supply its
        lifecycle runner here so close can drain queued owner deliveries.
        """
        super().__init__()
        self._ctrl = ctrl
        self._dialogs = dialog_presenter or QtDialogPresenter()
        self._writable = ctrl.control_identity.status().holder == "gui"
        self._close_prompt_open = False
        self._close_confirmed = False
        self._control_disposed = False
        # Set once an accepted close starts disposing views and owners.
        self._shutting_down = False
        self._shutdown_application: QApplication | None = None
        self._interactive_runner = interactive_runner
        self._search_runner = search_runner
        if search_runner is not None:
            search_runner.setParent(self)
        if interactive_runner is not None:
            interactive_runner.setParent(self)
        self._bus_subs = EventSubscriptions()
        self._routing_editor = False
        self._routing_page = False
        self._editor_context_id: int | None = None
        # The stage whose picker the editing area shows; None for a preview.
        self._editor_stage: PendingKind | None = None
        # The stage an agent's Filter / Search leaves for the user's next Fit
        # entry; a waiting Filter is dropped when its context closes.
        self._fit_entry_tab: AnalyzeTab | None = None
        self.setWindowTitle("fluxdep-gui")
        self.resize(1100, 700)

        self._build_ui()
        self._subscribe_events()
        self._refresh_project()
        self._refresh_list()
        self._rebuild_editor(open_picker=True)
        self._dispose_control = ctrl.control_identity.watch(self._refresh_control)

    def _refresh_control(self, status: ControlStatus) -> None:
        self._indicator.set_status(status)
        self.set_writable(writable=status.holder == "gui")

    def set_writable(self, *, writable: bool) -> None:
        """Push Local write availability, preserving canvas and non-writing tabs."""
        self._writable = writable
        for control in self._write_controls:
            control.setEnabled(writable)
        if isinstance(self._current_editor, InteractiveMplWidget):
            self._current_editor.set_editable(editable=writable)
        if self._analyze_panel is not None:
            self._analyze_panel.set_writable(writable=writable)

    # --- construction ----------------------------------------------------

    def watch_application_shutdown(self, application: QApplication) -> None:
        """Bind app-local watch retirement before the runtime's adapter stop.

        Call once on the owner thread. Final view disposal removes this app
        connection, so QApplication cannot retain an already retired window.
        """
        if self._shutdown_application is not None or self._control_disposed:
            raise RuntimeError("window shutdown projection is already bound or retired")
        self._shutdown_application = application
        application.aboutToQuit.connect(self.dispose_control_watches)

    def dispose_control_watches(self) -> None:
        """Retire only owned authority projections, on the owner thread.

        This is idempotent. It does not cancel or drain domain work. Refused
        closes keep watches; accepted closes retire before view teardown.
        """
        if self._control_disposed:
            return
        self._control_disposed = True
        self._dispose_control()
        if self._shutdown_application is not None:
            self._shutdown_application.aboutToQuit.disconnect(
                self.dispose_control_watches
            )
            self._shutdown_application = None

    def deleteLater(self) -> None:
        """Retire watches before direct Qt deletion, without domain shutdown."""
        self.dispose_control_watches()
        super().deleteLater()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(self._build_top_bar())

        # Main area: Spectra and Fit pages. Spectrum work never replaces Fit.
        self._pages = QTabWidget()
        self._pages.setObjectName("fluxdep.pages")
        self._spectra_page = self._build_spectra_page()
        self._pages.addTab(self._spectra_page, "Spectra")
        # The Fit page hosts the AnalyzePanel SINGLETON: built on first use and
        # kept after, so its stages / form / paths / figures survive page switches.
        self._fit_page = QWidget()
        self._fit_page.setObjectName("fluxdep.page.fit")
        self._fit_layout = QVBoxLayout(self._fit_page)
        self._fit_layout.setContentsMargins(0, 0, 0, 0)
        self._analyze_panel: AnalyzePanelWidget | None = None
        self._pages.addTab(self._fit_page, "Fit")
        self._pages.currentChanged.connect(self._on_page_changed)
        root.addWidget(self._pages, stretch=1)
        self.setCentralWidget(central)
        self._write_controls = (
            self._project_btn,
            self._add_btn,
            self._remove_btn,
            self._reload_btn,
            self._export_btn,
        )

    def _build_top_bar(self) -> QHBoxLayout:
        """Project and control identity on the left, collection commands right."""
        bar = QHBoxLayout()
        self._project_label = QLabel()
        make_label_selectable(self._project_label)
        self._project_label.setObjectName("fluxdep.session.project")
        bar.addWidget(self._project_label)
        self._indicator = ControlIndicator(self._ctrl.control_identity, self)
        bar.addWidget(self._indicator)
        bar.addStretch(1)

        self._project_btn = self._command_button(
            "fluxdep.project", "Project…", self._on_project_clicked
        )
        self._add_btn = self._command_button(
            "fluxdep.spectrum.add", "Add spectrum…", self._on_load_clicked
        )
        self._reload_btn = self._command_button(
            "fluxdep.spectra.restore", "Restore…", self._on_reload_clicked
        )
        self._export_btn = self._command_button(
            "fluxdep.spectra.export", "Export spectra…", self._on_export_clicked
        )
        for button in (
            self._project_btn,
            self._add_btn,
            self._reload_btn,
            self._export_btn,
        ):
            bar.addWidget(button)
        return bar

    @staticmethod
    def _command_button(
        object_name: str, text: str, on_click: Callable[[], None]
    ) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.clicked.connect(on_click)
        return button

    def _build_spectra_page(self) -> QWidget:
        """Spectrum list with Remove below it, beside the stage-driven editor."""
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        self._list = SpectrumListWidget()
        self._list.currentItemChanged.connect(self._on_list_selection)
        left.addWidget(self._list, stretch=1)
        self._remove_btn = self._command_button(
            "fluxdep.spectrum.remove", "Remove", self._on_remove_clicked
        )
        left.addWidget(self._remove_btn)
        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setMinimumWidth(180)

        # The stage bar above the editing area marks the active spectrum's
        # stages; a click opens (holder) or views (non-holder) that stage.
        self._stage_bar: StageBar[PendingKind] = StageBar(
            "fluxdep.stage",
            (("align", "Align"), ("sample", "Sample")),
            on_click=self._on_stage_clicked,
        )
        # Editing area (stacked). The active spectrum's pipeline stage decides
        # which interactive widget is shown; _current_editor is the live one
        # (replaced on stage/active change).
        self._editor_stack = QStackedWidget()
        self._editor_stack.setObjectName("fluxdep.editor")
        self._placeholder = QLabel("Select or load a spectrum to begin.")
        self._placeholder.setEnabled(False)
        self._editor_stack.addWidget(self._placeholder)
        self._current_editor: QWidget | None = None
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self._stage_bar)
        right.addWidget(self._editor_stack, stretch=1)
        right_panel = QWidget()
        right_panel.setLayout(right)

        # A draggable splitter lets the user resize the spectrum list vs editor.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 880])
        page = QWidget()
        page.setObjectName("fluxdep.page.spectra")
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        return page

    def _subscribe_events(self) -> None:
        bus = self._ctrl.bus
        self._bus_subs.subscribe(bus, SpectrumAddedPayload, self._on_spectrum_added)
        self._bus_subs.subscribe(bus, SpectrumRemovedPayload, self._on_spectrum_removed)
        self._bus_subs.subscribe(bus, SpectrumChangedPayload, self._on_spectrum_changed)
        self._bus_subs.subscribe(
            bus, ActiveSpectrumChangedPayload, self._on_active_changed
        )
        self._bus_subs.subscribe_with_meta(
            bus, InteractiveChangedPayload, self._on_interactive_changed
        )
        self._bus_subs.subscribe(bus, ProjectChangedPayload, self._on_project_changed)
        # Filter publication changes the per-spectrum excluded counts in the list.
        self._bus_subs.subscribe(
            bus, SelectionChangedPayload, self._on_selection_changed
        )
        self._bus_subs.subscribe_with_meta(
            bus, SearchChangedPayload, self._on_search_changed
        )
        self.destroyed.connect(self._bus_subs.unsubscribe_all)

    def _on_interactive_changed(
        self, payload: InteractiveChangedPayload, meta: EventMeta
    ) -> None:
        if payload.phase != "updated":
            self._refresh_list()  # a draft opened or closed changes its row
        if payload.phase == "closed":
            if payload.context_id == self._editor_context_id:
                self._clear_editor()
                # The owner is still emitting; refill once this event returns.
                QTimer.singleShot(0, self._show_after_closed_draft)
            if payload.kind == "selection" and self._fit_entry_tab == "filter":
                # An agent's Filter routes the next Fit entry only while live.
                self._fit_entry_tab = None
            self._refresh_stage_bar()  # the closed draft's mark and current stage
            return
        if payload.phase == "opened":
            self._refresh_stage_bar()  # an opened draft marks its stage
        if meta.origin.kind != "agent" or self._routing_editor:
            return
        active = (
            self._ctrl.interactive.inspect_selection()
            if payload.spectrum_name is None
            else self._ctrl.interactive.inspect(payload.spectrum_name)
        )
        if active is not None and active.context_id == payload.context_id:
            self._present_context(active)

    def _present_context(self, active: ActiveInteractiveContext) -> None:
        self._routing_editor = True
        try:
            if isinstance(active.context, CrossSelectionContext):
                self._follow_fit_tab("filter")
                return
            if active.spectrum_name != self._ctrl.state.active_spectrum:
                self._ctrl.set_active_spectrum(active.spectrum_name)
                self._clear_editor()
                self._mount_picker(active.context)
            elif active.context_id == self._editor_context_id:
                if self._current_editor is not None:
                    self._editor_stack.setCurrentWidget(self._current_editor)
            else:
                self._clear_editor()
                self._mount_picker(active.context)
        finally:
            self._routing_editor = False
        self._refresh_stage_bar()

    def _on_search_changed(
        self, payload: SearchChangedPayload, meta: EventMeta
    ) -> None:
        if meta.origin.kind == "agent" and payload.status == "pending":
            activity = self._ctrl.search.current
            if activity is not None and activity.token == payload.token:
                self._follow_fit_tab("search")

    def _on_project_changed(self, _payload: ProjectChangedPayload) -> None:
        self._refresh_project()

    def _on_selection_changed(self, _payload: SelectionChangedPayload) -> None:
        self._refresh_list()

    def _on_spectrum_added(self, _payload: SpectrumAddedPayload) -> None:
        self._refresh_list()

    def _on_spectrum_removed(self, _payload: SpectrumRemovedPayload) -> None:
        self._refresh_list()
        self._rebuild_editor()

    def _on_spectrum_changed(self, payload: SpectrumChangedPayload) -> None:
        self._refresh_list()
        # A stage change (aligned / points_completed) on the active spectrum must
        # advance its editing widget (line-picker → point-selector → done).
        if payload.name == self._ctrl.state.active_spectrum:
            self._rebuild_editor(open_picker=True)

    # --- top bar and list refresh ---------------------------------------

    def _refresh_project(self) -> None:
        project = self._ctrl.state.project
        if project.chip_name == DEFAULT_CHIP and project.qub_name == DEFAULT_QUBIT:
            self._project_label.setText("Project: not set (use Project…)")
        else:
            self._project_label.setText(
                f"Project: {project.chip_name} / {project.qub_name}"
            )

    def _refresh_list(self) -> None:
        """Rebuild the spectrum list from the Controller's published rows."""
        # Rows first: the query may retire a dead draft and re-enter this refresh.
        rows = self._ctrl.spectrum_rows()
        self._list.show_rows(rows, self._ctrl.state.active_spectrum)

    # --- actions ---------------------------------------------------------

    def _on_list_selection(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        # Picking only shows the spectrum's draft or published preview; a stage
        # click is the separate intention to edit it.
        if name and name != self._ctrl.state.active_spectrum:
            self._ctrl.set_active_spectrum(name)

    def _on_active_changed(self, _payload: ActiveSpectrumChangedPayload) -> None:
        self._refresh_list()  # keep the list selection in sync with the active spectrum
        if not self._routing_editor:
            # Any origin only updates the Spectra page; the shown page stays.
            self._rebuild_editor()

    # --- editing area (stage-driven interactive widget) ------------------

    def _clear_editor(self) -> None:
        if self._current_editor is not None:
            if isinstance(
                self._current_editor,
                (LinePickerWidget, OneToneWidget, FindPointsWidget),
            ):
                self._current_editor.teardown()
            self._editor_stack.removeWidget(self._current_editor)
            self._current_editor.deleteLater()
            self._current_editor = None
        self._editor_context_id = None
        self._editor_stage = None
        self._editor_stack.setCurrentWidget(self._placeholder)

    def _show_after_closed_draft(self) -> None:
        """Refill an editing area a closed draft left empty, opening no input.

        It runs after the closing event has returned, so the rebuild's inspect
        never re-enters the owner during retirement. A view mounted meanwhile
        (a successor picker or a stage advance) stays, and a window that is
        shutting down stays empty.
        """
        if self._shutting_down or self._current_editor is not None:
            return
        self._rebuild_editor()

    def _rebuild_editor(self, *, open_picker: bool = False) -> None:
        """Show an existing draft or published data without creating input.

        open_picker marks a separate Local write intention, gated by writable:
        the window opening, a GUI Add, or a published stage advance. That
        intention may open the current stage picker. Display-change events,
        list picks and observer RPCs never request it.
        """
        # begin_* emits synchronously; this stage route already mounts its context.
        self._routing_editor = True
        try:
            self._clear_editor()
            name = self._ctrl.state.active_spectrum
            if name is None:
                return
            entry = self._ctrl.state.spectrums[name]

            active = self._ctrl.interactive.inspect(name)
            if active is not None:
                context = active.context
                if not isinstance(context, CrossSelectionContext):
                    self._mount_picker(context)
            elif open_picker and self._writable and not entry.aligned:
                self._mount_stage(name, "align")
            elif open_picker and self._writable and not entry.points_completed:
                self._mount_stage(name, "sample")
            else:
                self._mount(ResultPreviewWidget(entry))
        finally:
            self._routing_editor = False
            # The bar follows whatever the editing area now shows.
            self._refresh_stage_bar()

    def _mount(self, widget: QWidget) -> None:
        self._current_editor = widget
        if isinstance(widget, InteractiveMplWidget):
            widget.set_editable(editable=self._writable)
        self._editor_stack.addWidget(widget)
        self._editor_stack.setCurrentWidget(widget)

    def _mount_stage(self, name: str, stage: PendingKind) -> None:
        """Begin (or reuse) name's picker for stage and mount it as the editor."""
        if stage == "align":
            self._mount_picker(self._ctrl.interactive.begin_line_pick(name))
        else:
            self._mount_picker(self._ctrl.interactive.begin_sample_pick(name))

    def _mount_picker(
        self, context: LinePickContext | OneTonePickContext | TwoTonePickContext
    ) -> None:
        widget: LinePickerWidget | OneToneWidget | FindPointsWidget
        if isinstance(context, LinePickContext):
            entry = self._ctrl.state.spectrums[context.spectrum_name]
            widget = LinePickerWidget(context, onetone=entry.spec_type == "OneTone")
            widget.finished.connect(lambda: self._finish_lines(context.spectrum_name))
        elif isinstance(context, OneTonePickContext):
            widget = OneToneWidget(context)
            widget.finished.connect(
                lambda: self._ctrl.interactive.finish_onetone_pick(
                    context.spectrum_name
                )
            )
        else:
            widget = FindPointsWidget(context)
            widget.finished.connect(
                lambda: self._ctrl.interactive.finish_twotone_pick(
                    context.spectrum_name
                )
            )
        self._mount(widget)
        active = self._ctrl.interactive.inspect(context.spectrum_name)
        if active is None or active.context is not context:
            raise RuntimeError("picker context changed while attaching its view")
        self._editor_context_id = active.context_id
        self._editor_stage = (
            "align" if isinstance(context, LinePickContext) else "sample"
        )

    def _finish_lines(self, name: str) -> None:
        try:
            self._ctrl.interactive.finish_line_pick(name)
        except (FailedPreconditionError, ValueError, RuntimeError) as exc:
            self._show_error("Alignment failed", str(exc))

    def _refresh_stage_bar(self) -> None:
        """Mark the active spectrum's stages from State and its live draft.

        Align is done once alignment is published; Sample is locked until
        then and done once points are also published. A live draft of a stage
        marks it draft. The current stage is the one whose picker is shown.
        Without an active spectrum the bar is hidden.
        """
        name = self._ctrl.state.active_spectrum
        self._stage_bar.setVisible(name is not None)
        if name is None:
            return
        entry = self._ctrl.state.spectrums[name]
        draft = self._ctrl.interactive.inspect(name)
        pending = None if draft is None else pending_kind(draft)
        self._stage_bar.set_status(
            "align",
            stage_status(done=entry.aligned, draft=pending == "align", locked=False),
        )
        self._stage_bar.set_status(
            "sample",
            stage_status(
                done=entry.aligned and entry.points_completed,
                draft=pending == "sample",
                locked=not entry.aligned,
            ),
        )
        self._stage_bar.set_current(self._editor_stage)

    def _on_stage_clicked(self, stage: PendingKind) -> None:
        """A holder click opens stage's picker; a non-holder click only views."""
        if self._writable:
            self._open_stage(stage)
        else:
            self._view_stage(stage)

    def _view_stage(self, stage: PendingKind) -> None:
        """Show stage without creating input.

        A live draft of stage, such as an agent's, is shown as its picker;
        otherwise the published preview is shown. No draft is opened or
        discarded.
        """
        name = self._ctrl.state.active_spectrum
        if name is None:
            return
        draft = self._ctrl.interactive.inspect(name)
        if (
            draft is not None
            and pending_kind(draft) == stage
            and not isinstance(draft.context, CrossSelectionContext)
        ):
            if draft.context_id != self._editor_context_id:
                self._clear_editor()
                self._mount_picker(draft.context)
        elif not isinstance(self._current_editor, ResultPreviewWidget):
            self._clear_editor()
            self._mount(ResultPreviewWidget(self._ctrl.state.spectrums[name]))
        self._refresh_stage_bar()

    def _open_stage(self, stage: PendingKind) -> None:
        """Open the active spectrum's align or sample picker on a Local intent.

        This is the holder gate for the stage bar: a non-holder call does
        nothing. The picker opens as a draft;
        the published spectrum changes only on Finish. A live draft of the
        other stage is discarded only after the user confirms; declining keeps
        it.
        """
        name = self._ctrl.state.active_spectrum
        if not self._writable or name is None:
            return
        draft = self._ctrl.interactive.inspect(name)
        if draft is None or pending_kind(draft) == stage:
            self._switch_stage(name, stage)
            return
        other = "sample" if stage == "align" else "align"
        self._dialogs.destructive_confirm(
            self,
            DestructiveConfirmation(
                title=f"Discard the {other} draft?",
                message=f"{name} has an unfinished {other} draft. "
                f"Opening {stage} discards it; the published spectrum stays as it is.",
                action_text=f"Discard and {stage}",
            ),
            on_decision=lambda confirmed: self._on_discard_decision(
                name, stage, draft.context_id, confirmed=confirmed
            ),
        )

    def _on_discard_decision(
        self, name: str, stage: PendingKind, draft_id: int, *, confirmed: bool
    ) -> None:
        if not confirmed:
            return
        # Authority, display or the draft may change while the dialog is open;
        # the confirmation covers only the draft the user saw.
        current = self._ctrl.interactive.inspect(name)
        if (
            not self._writable
            or self._ctrl.state.active_spectrum != name
            or (current is not None and current.context_id != draft_id)
        ):
            return
        self._switch_stage(name, stage)

    def _switch_stage(self, name: str, stage: PendingKind) -> None:
        # begin_* emits synchronously; this route already mounts its context.
        self._routing_editor = True
        try:
            self._clear_editor()
            self._mount_stage(name, stage)
        finally:
            self._routing_editor = False
        self._refresh_stage_bar()

    # --- pages -----------------------------------------------------------

    def _on_page_changed(self, _index: int) -> None:
        """Route a user tab switch: entering Fit activates, leaving detaches."""
        if self._routing_page:
            return
        if self._pages.currentWidget() is self._fit_page:
            self._show_fit_page(None)
        elif self._analyze_panel is not None:
            # Leaving Fit retires the Filter view; the Filter context stays open.
            self._analyze_panel.detach()

    def _show_spectra_page(self) -> None:
        self._pages.setCurrentWidget(self._spectra_page)

    def _follow_fit_tab(self, tab: AnalyzeTab) -> None:
        """Follow an agent's Filter or Search on the Fit page, never leaving pages.

        A shown Fit page switches to stage tab now; otherwise tab waits for the
        user's next entry into Fit. A waiting Filter lasts only while a Filter
        context is live, so a finished agent Filter no longer picks the stage.
        """
        if self._pages.currentWidget() is self._fit_page:
            self._show_fit_page(tab)
        else:
            self._fit_entry_tab = tab

    def _show_fit_page(self, tab: AnalyzeTab | None) -> None:
        """Show the Fit page and its singleton panel on stage tab.

        None takes the stage an agent event left waiting, else keeps the
        current one. Entering Filter never opens input: it reattaches a live
        Filter context, such as an agent's, or shows the published masks.
        """
        from .analyze_panel import AnalyzePanelWidget

        if self._analyze_panel is None:
            self._analyze_panel = AnalyzePanelWidget(self._ctrl)
            self._analyze_panel.set_writable(writable=self._writable)
            self._fit_layout.addWidget(self._analyze_panel)
        panel = self._analyze_panel
        if tab is None:
            tab = self._fit_entry_tab or panel.current_tab
        self._fit_entry_tab = None
        self._routing_page = True
        try:
            self._pages.setCurrentWidget(self._fit_page)
        finally:
            self._routing_page = False
        panel.show_tab(tab)

    def _on_load_clicked(self) -> None:
        if not self._writable:
            return
        from .load_dialog import LoadSpectrumDialog
        from .paths import raw_spectrum_dir

        dialog = LoadSpectrumDialog(
            {
                name: (entry.flux_half, entry.flux_int)
                for name, entry in self._ctrl.state.spectrums.items()
            },
            parent=self,
            start_dir=raw_spectrum_dir(self._ctrl.state.project),
        )
        # Non-blocking open() keeps the Qt event loop (and the read-only control
        # socket) responsive; the post-accept logic moves onto the accepted signal.
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_accepted() -> None:
            req = dialog.result_request()
            if req is None:
                return
            try:
                name = self._ctrl.load_spectrum(
                    req.filepath,
                    req.spec_type,
                    inherit_from=req.inherit_from,
                    transpose_axes=req.transpose_axes,
                )
                self._ctrl.set_active_spectrum(name)
            except Exception as exc:
                logger.exception("load_spectrum failed")
                self._show_io_error("Load", req.filepath, exc)
                return
            self._show_spectra_page()
            # Adding is the Local intent to start aligning the new spectrum.
            self._rebuild_editor(open_picker=True)

        dialog.accepted.connect(_on_accepted)
        # Hold a reference so open()'s immediate return doesn't let the dialog be GC'd.
        self._load_dialog = dialog
        dialog.finished.connect(lambda _result=0: setattr(self, "_load_dialog", None))
        dialog.open()

    def _on_reload_clicked(self) -> None:
        """Restore a processed spectrums.hdf5 (aligned + selected spectra)."""
        if not self._writable:
            return
        from qtpy.QtWidgets import QFileDialog

        from .paths import processed_spectrum_dir

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load spectrums.hdf5",
            processed_spectrum_dir(self._ctrl.state.project),
            filter="HDF5 (*.hdf5 *.h5);;All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not filepath:
            return
        try:
            names = self._ctrl.load_processed_spectrums(filepath)
        except Exception as exc:
            logger.exception("load_processed_spectrums failed")
            self._show_io_error("Restore", filepath, exc)
            return
        if names:
            self._ctrl.set_active_spectrum(names[0])
        self._show_spectra_page()

    def _on_remove_clicked(self) -> None:
        if not self._writable:
            return
        name = self._ctrl.state.active_spectrum
        if name is None:
            return
        # Pick the spectrum to focus next (the following one, else the previous).
        names = self._ctrl.list_spectrums()
        idx = names.index(name) if name in names else -1
        successor = None
        if idx >= 0:
            rest = names[idx + 1 :] + names[:idx][::-1]
            successor = rest[0] if rest else None

        self._ctrl.remove_spectrum(name)
        if successor is not None:
            self._ctrl.set_active_spectrum(successor)

    def _on_project_clicked(self) -> None:
        """Edit the project chip/qubit names (and optional roots)."""
        if not self._writable:
            return
        from zcu_tools.gui.widgets.project_dialog import ProjectDialog

        dialog = ProjectDialog(
            self._ctrl.state.project,
            parent=self,
            db_label="Data root",
            project_root=self._ctrl.get_project_root(),
        )
        # Non-blocking open() keeps the Qt event loop (and the read-only control
        # socket) responsive; the post-accept logic moves onto the accepted signal.
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_accepted() -> None:
            self._ctrl.setup_project(dialog.result_project())

        dialog.accepted.connect(_on_accepted)
        # Hold a reference so open()'s immediate return doesn't let the dialog be GC'd.
        self._project_dialog = dialog
        dialog.finished.connect(
            lambda _result=0: setattr(self, "_project_dialog", None)
        )
        dialog.open()

    def _on_export_clicked(self) -> None:
        if not self._writable:
            return
        from .export_dialog import ExportSpectrumsDialog

        # The default path comes from the project's result dir (set via Project…);
        # the dialog only confirms / overrides the path.
        project = self._ctrl.state.project
        dialog = ExportSpectrumsDialog(result_dir=project.result_dir, parent=self)
        # Non-blocking open() keeps the Qt event loop (and the read-only control
        # socket) responsive; the post-accept logic moves onto the accepted signal.
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_accepted() -> None:
            path = dialog.export_path()
            if path is None:
                return
            try:
                resolved = self._ctrl.export_spectrums(filepath=path, mode="w")
                self._show_info("Exported", f"Wrote {resolved}")
            except Exception as exc:
                logger.exception("export failed")
                self._show_io_error("Export", path, exc)

        dialog.accepted.connect(_on_accepted)
        # Hold a reference so open()'s immediate return doesn't let the dialog be GC'd.
        self._export_dialog = dialog
        dialog.finished.connect(lambda _result=0: setattr(self, "_export_dialog", None))
        dialog.open()

    # --- dialogs ---------------------------------------------------------

    def _show_error(self, title: str, message: str) -> None:
        from qtpy.QtWidgets import QMessageBox

        QMessageBox.critical(self, title, message)

    def _show_io_error(self, action: str, filepath: str, exc: Exception) -> None:
        """Show a friendly message for a file IO failure (load / restore / export)."""
        from .error_messages import friendly_io_message

        self._show_error(f"{action} failed", friendly_io_message(action, filepath, exc))

    def _show_info(self, title: str, message: str) -> None:
        from qtpy.QtWidgets import QMessageBox

        QMessageBox.information(self, title, message)

    # --- close path --------------------------------------------------------

    def _on_close_decision(self, *, confirmed: bool) -> None:
        self._close_prompt_open = False
        if confirmed:
            # One accepted intent survives drain refusals and holder changes.
            self._close_confirmed = True
            self.close()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Confirm Remote close asynchronously, then wait for search to settle.

        Pending or declined decisions leave owners alive and admission open.
        Acceptance is remembered across drain refusals and holder changes.
        begin_close permanently blocks admission and requests cancellation.
        Keep subscriptions, figures and Qt owners alive on drain refusal.
        A later close attempt can finish once the runner and search owner drain.
        """
        if self._close_prompt_open:
            if a0 is not None:
                a0.ignore()
            return
        status = self._ctrl.control_identity.status()
        if status.holder != "gui" and not self._close_confirmed:
            if a0 is not None:
                a0.ignore()
            self._close_prompt_open = True
            self._dialogs.confirm_async(
                self,
                "Close remote-controlled window?",
                f"{status.holder.removeprefix('rpc:')} controls this window. Close and cancel pending work?",
                on_decision=lambda confirmed: self._on_close_decision(
                    confirmed=confirmed
                ),
                default=False,
            )
            return
        self._ctrl.search.begin_close()
        search_drained = self._search_runner is None or self._search_runner.quiesce()
        # Startup notifications can reenter close before any worker is submitted.
        if not search_drained or self._ctrl.search.active_token is not None:
            if a0 is not None:
                a0.ignore()
            return
        self._shutting_down = True
        self.dispose_control_watches()
        self._ctrl.interactive.dispose()
        if self._interactive_runner is not None:
            self._interactive_runner.quiesce()
        self._bus_subs.unsubscribe_all()
        if isinstance(self._current_editor, LinePickerWidget):
            self._current_editor.teardown()
        if isinstance(self._current_editor, FindPointsWidget):
            self._current_editor.quiesce()
        if self._analyze_panel is not None:
            self._analyze_panel.quiesce()
            self._analyze_panel.dispose()
        super().closeEvent(a0)
