"""Per-experiment tab widget for the measure-gui main window - capability driven."""

# tolerated(file-size): Figure presentation lives in ExperimentFigureArea, and
# ledger builders live in experiment_ledger. This owner keeps left forms,
# splitter width, source subscriptions, action admission and attach/detach
# together; another split would scatter coordination of the same tab lifecycle.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.artifacts import ArtifactSaveCenter
from zcu_tools.gui.app.measure.ui.interactive_frontend import InteractiveFrontend
from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.form import FormErrors, FormModel, FormState
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid
from zcu_tools.gui.session.ui.progress_stack import ProgressStack
from zcu_tools.gui.widgets import (
    DialogPresenter,
    QtDialogPresenter,
    secondary_colour,
    set_foreground,
)
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from ..writeback import WritebackWidget
from .experiment_figure_area import ExperimentFigureArea
from .experiment_ledger import (
    BLUE_PRIMARY_STYLESHEET,
    LedgerSection,
    ledger_column,
    ledger_form,
    primary_action_button,
)
from .guide_view import GuideView

logger = logging.getLogger(__name__)

_RED_STOP_STYLESHEET = (
    "background-color: #f44336; color: white; font-weight: bold; "
    "border: 1px solid #d32f2f; border-radius: 4px;"
)


if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.services import TabSnapshot
    from zcu_tools.gui.app.measure.state import TabInteractionState
    from zcu_tools.gui.app.measure.writeback import WritebackItem, WritebackPane
    from zcu_tools.plotting.figures import FigureCollection


class TabActions(Protocol):
    """Tab-level actions supplied by the top-level window boundary."""

    def refresh_interaction(self, tab_id: str) -> None: ...

    def run_or_stop(self, tab_id: str) -> None: ...

    def load_data(self, tab_id: str) -> None: ...

    def analyze(self, tab_id: str) -> None: ...

    def post_analyze(self, tab_id: str) -> None: ...

    def apply_writeback(self, tab_id: str) -> None: ...

    def apply_post_writeback(self, tab_id: str) -> None: ...

    def save_artifact(self, tab_id: str, ref: ArtifactRef) -> None:
        """Save every member of one declared artifact."""
        ...

    def save_all(self, tab_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Per-experiment tab widget
# ---------------------------------------------------------------------------


class _PanelEdgeHandle(QToolButton):
    """Boundary handle for collapsing/expanding the left panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(16, 42)
        # tolerated: QtPy promotes the binding enum members onto Qt at runtime.
        self.setCursor(Qt.PointingHandCursor)  # pyright: ignore[reportAttributeAccessIssue]
        self.setAutoRaise(True)
        self._collapsed = False

    def set_collapsed(self, *, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.setToolTip("Expand left panel" if collapsed else "Collapse left panel")
        self.update()

    def paintEvent(self, a0) -> None:
        del a0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        notch = 6
        path.moveTo(rect.right() - 1, rect.center().y())
        path.lineTo(rect.right() - notch, rect.top())
        path.lineTo(rect.left(), rect.top())
        path.lineTo(rect.left(), rect.bottom())
        path.lineTo(rect.right() - notch, rect.bottom())
        path.closeSubpath()

        fill = QColor(236, 238, 242)
        border = QColor(120, 126, 138)
        arrow = QColor(70, 76, 88)
        if self.underMouse():
            fill = QColor(224, 228, 236)
            border = QColor(96, 102, 114)

        painter.setPen(QPen(border, 1.2))
        painter.setBrush(fill)
        painter.drawPath(path)

        painter.setPen(QPen(arrow, 2))
        center_x = rect.center().x()
        center_y = rect.center().y()
        if self._collapsed:
            painter.drawLine(center_x - 2, center_y - 7, center_x + 2, center_y)
            painter.drawLine(center_x - 2, center_y + 7, center_x + 2, center_y)
        else:
            painter.drawLine(center_x + 2, center_y - 7, center_x - 2, center_y)
            painter.drawLine(center_x + 2, center_y + 7, center_x - 2, center_y)


class ExpTabWidget(QWidget):
    """Project an experiment snapshot into capability-driven forms and figure panes.

    The owned ExperimentFigureArea holds right-side presentation. Figure methods
    run synchronously on the Qt owner thread and borrow caller-owned Figures.
    Unsupported Analysis/Post operations raise RuntimeError; attach/draw failures
    propagate. Clearing presentation keeps retained Figures saveable.
    """

    def __init__(
        self,
        tab_id: str,
        ctrl: Controller,
        capabilities: AdapterCapabilities,
        parent: QWidget | None = None,
        *,
        dialog_presenter: DialogPresenter | None = None,
        preview_renderer: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.tab_id = tab_id
        self._ctrl = ctrl
        self._capabilities = capabilities
        self._has_analysis = capabilities.analysis is not AnalysisMode.NONE
        self._has_post = bool(capabilities.post_analysis)
        self._dialog_presenter = dialog_presenter or QtDialogPresenter()
        self._writable = True
        self._progress_control = ctrl.progress_control
        self._cfg = ctrl.cfg_resources.lookup(tab_id)
        # The action boundary is retained for Reset; button slots close over it.
        self._actions: TabActions | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(2)
        self._analysis_source_notice = QLabel()
        self._analysis_source_notice.setObjectName("history.analysis_source")
        self._analysis_source_notice.setWordWrap(True)
        make_label_selectable(self._analysis_source_notice)
        self._analysis_source_notice.hide()
        root_layout.addWidget(self._analysis_source_notice)

        # --- main content area: [splitter] ---
        self._content_widget = QWidget()
        content_row = QHBoxLayout(self._content_widget)
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)
        root_layout.addWidget(self._content_widget, stretch=1)

        # --- progress stack at bottom (zero height when idle) ---
        self.progress_stack = ProgressStack()
        root_layout.addWidget(self.progress_stack, stretch=0)

        # Subscribe once by our own tab_id
        self._progress_unsub = self._progress_control.attach_progress(
            self.tab_id, self._on_progress_changed
        )

        # splitter holds two panes: left (tab panel) | right (plot)
        # tolerated: QtPy promotes the binding enum members onto Qt at runtime.
        splitter = QSplitter(Qt.Horizontal)  # pyright: ignore[reportAttributeAccessIssue]

        content_row.addWidget(splitter, stretch=1)

        self._splitter = splitter
        self._splitter_left_saved = ctrl.get_left_panel_width()
        self._left_panel_collapsed = False
        self._splitter.splitterMoved.connect(self._on_splitter_moved)

        # ── Left pane: QTabWidget with capability-driven tabs ──────
        self._left_tabs = QTabWidget()

        self._left_edge_handle = _PanelEdgeHandle(self._content_widget)
        self._left_edge_handle.clicked.connect(self._toggle_left_panel)

        self._build_run_panel()
        self._build_analysis_panel()
        self._build_post_analysis_panel()
        self._build_data_panel()
        self._build_guide_panel()

        splitter.addWidget(self._left_tabs)

        self._figure_area = ExperimentFigureArea(
            tab_id,
            capabilities,
            lambda pane: self.get_current_figure_for_pane(pane),
            preview_renderer=preview_renderer,
        )
        splitter.addWidget(self._figure_area)

        splitter.setCollapsible(0, True)
        self._update_left_panel_controls()
        self._schedule_handle_layout()
        # Right pane follows left selection
        self._left_tabs.currentChanged.connect(self._on_left_tab_changed)
        # Initially show Run
        self._on_left_tab_changed(self._left_tabs.currentIndex())

    def _build_run_panel(self) -> None:
        # ── Tab: Run (always) ──────────────────────────────────────
        run_panel = QWidget()
        run_layout = QVBoxLayout(run_panel)
        run_layout.setContentsMargins(4, 4, 4, 4)
        run_layout.setSpacing(2)

        # The form editor borrows the resource's FormState; Reset installs a
        # new one, so the editor is remounted when the state changes.
        self._form_scroll = QScrollArea()
        self._form_scroll.setWidgetResizable(True)
        self._editor: FormEditor[FormModel] | None = None
        self._editor_state: FormState[FormModel] | None = None
        self._editing_enabled = True
        self._cfg_unsub: Callable[[], None] | None = None
        self._follow_cfg()
        run_layout.addWidget(self._form_scroll, stretch=1)
        self._run_cfg_notice = QLabel()
        self._run_cfg_notice.setObjectName("run.cfg_error")
        self._run_cfg_notice.setWordWrap(True)
        self._run_cfg_notice.hide()
        run_layout.addWidget(self._run_cfg_notice)

        # Explicit source actions share the idle row; running shows only Stop.
        self._run_action_bar = QFrame()
        self._run_action_bar.setObjectName("runActionBar")
        self._run_action_bar.setFrameShape(QFrame.Shape.StyledPanel)
        bar_layout = QHBoxLayout(self._run_action_bar)
        self._run_action_layout = bar_layout
        bar_layout.setContentsMargins(8, 6, 8, 6)
        bar_layout.setSpacing(6)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("refreshSourcesButton")
        self.refresh_btn.setToolTip(
            "Recapture sources and revalidate, keeping all input"
        )
        self.refresh_btn.setFixedHeight(30)
        self.refresh_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        bar_layout.addWidget(self.refresh_btn, stretch=15)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("resetButton")
        self.reset_btn.setToolTip("Recapture sources and restore adapter defaults")
        self.reset_btn.setFixedHeight(30)
        self.reset_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        bar_layout.addWidget(self.reset_btn, stretch=15)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setFixedHeight(30)
        self.run_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.run_btn.setStyleSheet(BLUE_PRIMARY_STYLESHEET)
        bar_layout.addWidget(self.run_btn, stretch=70)
        run_layout.addWidget(self._run_action_bar)
        self._run_panel = run_panel
        self._left_tabs.addTab(run_panel, "Run")

    def _build_analysis_panel(self) -> None:
        # ── Tab: Analysis (only when analysis capability present) ──
        if self._has_analysis:
            # Single-column 13 px ledger with whole-header folding; Analyze sits
            # immediately after params and before Writeback preview (A4).
            analysis_container, analysis_layout = ledger_column()

            self._analyze_section = LedgerSection(
                "Analysis parameters", collapsed=False
            )
            self.analyze_form = ledger_form()
            self._analyze_section.body_layout.addWidget(self.analyze_form)
            analysis_layout.addWidget(self._analyze_section)

            # Analyze immediately below params and before Writeback preview,
            # occupying 100% of available content width (A6). Stays enabled
            # per availability/busy and retains blue primary when idle.
            self.analyze_btn = primary_action_button("Analyze")
            analysis_layout.addWidget(self.analyze_btn)

            self.writeback_section = LedgerSection("Writeback preview", collapsed=False)
            self.writeback_widget = WritebackWidget(
                self._ctrl, tab_id=self.tab_id, pane="analysis"
            )
            self.writeback_section.body_layout.addWidget(self.writeback_widget)
            self.writeback_section.setVisible(False)
            analysis_layout.addWidget(self.writeback_section)
            analysis_layout.addStretch()

            self._analysis_panel = analysis_container
            self._analysis_tab_index = self._left_tabs.addTab(
                analysis_container, "Analysis"
            )

    def _build_post_analysis_panel(self) -> None:
        # ── Tab: Post-Analysis (only when post capability true) ────
        # Same ledger, form, button, and writeback presentation as Analysis.
        if self._has_post:
            post_container, post_layout = ledger_column()

            self._post_analyze_section = LedgerSection(
                "Post-analysis parameters", collapsed=False
            )
            self.post_analyze_form = ledger_form()
            self._post_analyze_section.body_layout.addWidget(self.post_analyze_form)
            post_layout.addWidget(self._post_analyze_section)

            # Gate hint shown until a primary analyze result exists (form/Run disabled).
            self._post_gate_label = QLabel("Run analyze first to enable post-analysis.")
            self._post_gate_label.setWordWrap(True)
            set_foreground(self._post_gate_label, secondary_colour(self))
            post_layout.addWidget(self._post_gate_label)

            self.post_analyze_btn = primary_action_button("Run Post-Analysis")
            post_layout.addWidget(self.post_analyze_btn)

            self.post_writeback_section = LedgerSection(
                "Writeback preview", collapsed=False
            )
            self.post_writeback_widget = WritebackWidget(
                self._ctrl, tab_id=self.tab_id, pane="post_analysis"
            )
            self.post_writeback_section.body_layout.addWidget(
                self.post_writeback_widget
            )
            self.post_writeback_section.setVisible(False)
            post_layout.addWidget(self.post_writeback_section)
            post_layout.addStretch()

            self._post_panel = post_container
            self._post_tab_index = self._left_tabs.addTab(
                post_container, "Post-Analysis"
            )

    def _build_data_panel(self) -> None:
        # ── Tab: Data (always) — save center ──────────────────────
        self._save_center = ArtifactSaveCenter(self._capabilities)
        self._save_center.bind_comment_changed(self._on_comment_changed)
        save_scroll = QScrollArea()
        save_scroll.setWidgetResizable(True)
        save_scroll.setWidget(self._save_center)
        self._save_panel = save_scroll
        self._left_tabs.addTab(save_scroll, "Data")

    def _build_guide_panel(self) -> None:
        # ── Tab: Guide (always) ──────────────────────────────────
        adapter_name = self._ctrl.tab_control.get_tab_adapter_name(self.tab_id)
        self._guide_panel = GuideView(self._ctrl.get_adapter_guide(adapter_name))
        self._left_tabs.addTab(self._guide_panel, "Guide")

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    def _require_analysis(self) -> None:
        if not self._has_analysis:
            raise RuntimeError(f"tab {self.tab_id!r} does not support analysis")

    def _require_post(self) -> None:
        if not self._has_post:
            raise RuntimeError(f"tab {self.tab_id!r} does not support post-analysis")

    def resizeEvent(self, a0) -> None:
        super().resizeEvent(a0)
        self._fix_splitter_on_resize()
        self._schedule_handle_layout()

    def _fix_splitter_on_resize(self) -> None:
        if self._left_panel_collapsed:
            return
        sizes = self._splitter.sizes()
        total = sizes[0] + sizes[1]
        if total <= 0:
            return
        max_left = int(total * 0.8)
        left = min(self._splitter_left_saved, max_left)
        right = total - left
        if sizes[0] != left:
            self._splitter.setSizes([left, right])

    def showEvent(self, a0) -> None:
        super().showEvent(a0)
        self._fix_splitter_on_resize()
        self._schedule_handle_layout()

    def _toggle_left_panel(self) -> None:
        if self._left_panel_collapsed:
            self._expand_left_panel()
        else:
            self._collapse_left_panel()

    def _collapse_left_panel(self) -> None:
        sizes = self._splitter.sizes()
        self._splitter_left_saved = max(1, sizes[0])
        sizes[1] += sizes[0]
        sizes[0] = 0
        self._splitter.setSizes(sizes)
        self._left_panel_collapsed = True
        self._update_left_panel_controls()

    def _expand_left_panel(self) -> None:
        sizes = self._splitter.sizes()
        saved = max(240, self._splitter_left_saved)
        sizes[0] = saved
        sizes[1] = max(0, sizes[1] - saved)
        self._splitter.setSizes(sizes)
        self._left_panel_collapsed = False
        self._update_left_panel_controls()

    def _update_left_panel_controls(self) -> None:
        self._left_edge_handle.set_collapsed(collapsed=self._left_panel_collapsed)
        self._left_edge_handle.setVisible(True)
        self._schedule_handle_layout()
        self._left_edge_handle.raise_()

    def _layout_collapsed_handle(self) -> None:
        host = self._content_widget.rect()
        splitter_x = self._splitter.geometry().x()
        if self._left_panel_collapsed:
            boundary_x = splitter_x
        else:
            boundary_x = splitter_x + self._left_tabs.geometry().right() + 1
        x = max(0, boundary_x - self._left_edge_handle.width() // 2)
        y = max(8, (host.height() - self._left_edge_handle.height()) // 2)
        self._left_edge_handle.move(x, y)

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        if not self._left_panel_collapsed:
            sizes = self._splitter.sizes()
            if sizes[0] > 0:
                self._splitter_left_saved = sizes[0]
        self._schedule_handle_layout()

    def _schedule_handle_layout(self) -> None:
        QTimer.singleShot(0, self._layout_collapsed_handle)

    # ── attach / detach (whole-tab, snapshot-driven) ──────────────────────

    def attach(self, snapshot: TabSnapshot, actions: TabActions) -> None:
        """Bring this tab widget to life from one snapshot."""
        if snapshot.capabilities is None:
            raise RuntimeError(
                f"render snapshot for tab {self.tab_id!r} has no capabilities"
            )
        if snapshot.capabilities != self._capabilities:
            raise RuntimeError(
                f"capability mismatch for tab {self.tab_id!r}: "
                f"widget {self._capabilities!r} vs snapshot {snapshot.capabilities!r}"
            )
        self._follow_cfg()
        if (
            self._has_analysis
            and snapshot.analysis is not None
            and snapshot.analysis.params is not None
            and self.has_analyze_params()
        ):
            self.analyze_form.populate_values(snapshot.analysis.params)
        if self._has_post:
            post_params = (
                snapshot.post_analysis.params
                if snapshot.post_analysis is not None
                else None
            )
            self.sync_post_analyze_params(post_params)
        else:
            if (
                snapshot.post_analysis is not None
                and snapshot.post_analysis.params is not None
            ):
                raise RuntimeError(
                    f"tab {self.tab_id!r} received post params but does not support post-analysis"
                )
        self.update_interaction_state(snapshot)
        self._bind_to_controller(actions)

    # ── populate / refresh helpers ────────────────────────────────────────

    def populate_analyze_params(self, instance: object) -> None:
        self._require_analysis()
        self.analyze_form.populate(instance)

    def read_analyze_params(self) -> object:
        self._require_analysis()
        return self.analyze_form.read_params()

    def has_analyze_params(self) -> bool:
        if not self._has_analysis:
            return False
        return self.analyze_form.has_params()

    def populate_post_analyze_params(self, instance: object) -> None:
        self._require_post()
        self.sync_post_analyze_params(instance)

    def sync_post_analyze_params(self, instance: object | None) -> None:
        self._require_post()
        self.post_analyze_form.sync(instance)

    def read_post_analyze_params(self) -> object:
        self._require_post()
        return self.post_analyze_form.read_params()

    def has_post_analyze_params(self) -> bool:
        if not self._has_post:
            return False
        return self.post_analyze_form.has_params()

    def update_writeback_items(self, items: list[WritebackItem]) -> None:
        """Show the primary analysis writeback preview for items.

        A preview failure inside its contract (expected, validation, or I/O
        error) shows its reason in place, above any unavailable reasons; any
        other failure propagates.
        """
        self._require_analysis()
        self._refresh_writeback("analysis", items)

    def update_post_writeback_items(self, items: list[WritebackItem]) -> None:
        """Show the post-analysis writeback preview, as update_writeback_items."""
        self._require_post()
        self._refresh_writeback("post_analysis", items)

    def _refresh_writeback(
        self, pane: WritebackPane, items: list[WritebackItem]
    ) -> None:
        snapshot = self._ctrl.tab_control.get_tab_snapshot(self.tab_id)
        if pane == "analysis":
            widget, section = self.writeback_widget, self.writeback_section
            facts = snapshot.analysis
        else:
            widget, section = self.post_writeback_widget, self.post_writeback_section
            facts = snapshot.post_analysis
        reasons = facts.unavailable_reasons if facts is not None else ()
        section.setVisible(bool(items or reasons))
        if not items:
            widget.populate(items, {}, unavailable_reasons=reasons)
            return
        try:
            try:
                values = self._ctrl.writeback_control.get_writeback_values_for_pane(
                    self.tab_id, pane
                )
            except LibraryEntryInvalid:
                widget.refresh_module_validity(items, unavailable_reasons=reasons)
                return
        except (ExpectedError, ValueError, OSError) as error:
            # Isolate the preview: the pane stays usable and names the failure.
            logger.exception("Writeback preview failed for %s %s", self.tab_id, pane)
            widget.show_preview_unavailable(str(error), unavailable_reasons=reasons)
            return
        widget.populate(items, values, unavailable_reasons=reasons)

    def get_comment(self) -> str:
        return self._save_center.get_comment()

    def _on_comment_changed(self, text: str) -> None:
        if self._actions is None:
            return
        self._ctrl.save_control.set_comment(self.tab_id, text)
        self.update_interaction_state(
            self._ctrl.tab_control.get_tab_snapshot(self.tab_id)
        )

    def has_unsaved_data(self) -> bool:
        """Return True if this tab contains unsaved measurement data."""
        return self._save_center.has_unsaved_data()

    def select_pane(
        self, pane: Literal["run", "analysis", "post_analysis", "data"]
    ) -> None:
        """Select an explicit available pane without changing domain state."""
        if pane == "run":
            panel = self._run_panel
        elif pane == "data":
            panel = self._save_panel
        elif pane == "analysis" and self._has_analysis:
            panel = self._analysis_panel
        elif pane == "post_analysis" and self._has_post:
            panel = self._post_panel
        else:
            raise ValueError(f"Unavailable tab pane: {pane!r}")
        self._left_tabs.setCurrentWidget(panel)

    def focus_result_panel(self) -> None:
        """Focus Analysis when supported, otherwise focus Save."""
        if self._has_analysis:
            self._left_tabs.setCurrentWidget(self._analysis_panel)
            return
        self._left_tabs.setCurrentWidget(self._save_panel)

    # ── Figure container helpers (stable per-subtab identity S2) ──────────

    def get_run_container(self) -> FigureContainer:
        """Return the stable Run container borrowed by the explicit plot host."""
        return self._figure_area.get_run_container()

    def get_analysis_container(self) -> FigureContainer:
        """Return the stable Analysis container; unsupported panes raise RuntimeError."""
        return self._figure_area.get_analysis_container()

    def get_post_container(self) -> FigureContainer:
        """Return the stable Post container; unsupported panes raise RuntimeError."""
        return self._figure_area.get_post_container()

    def prepare_run_container(self) -> FigureContainer:
        """Clear Run and every downstream presentation for a new run."""
        return self._figure_area.prepare_run_container()

    def prepare_analysis_container(self) -> FigureContainer:
        """Clear Analysis presentation and return its container."""
        return self._figure_area.prepare_analysis_container()

    def prepare_post_container(self) -> FigureContainer:
        """Clear Post presentation and return its container."""
        return self._figure_area.prepare_post_container()

    def clear_post_figure(self) -> None:
        """Clear only the invalidated Post-Analysis presentation."""
        self._figure_area.clear_post_figure()

    def mount_interactive_widget(self, widget: QWidget) -> None:
        """Mount an interactive analysis widget as the visible plot content (analysis pane)."""
        self._figure_area.mount_interactive_widget(widget)

    def interactive_frontend(self) -> InteractiveFrontend | None:
        """Return only the currently mounted interactive analysis presentation."""
        return self._figure_area.interactive_frontend()

    def unmount_interactive_widgets(self, widget_type: type[QWidget]) -> None:
        """Remove interactive widgets of ``widget_type`` and show placeholder in analysis pane."""
        self._figure_area.unmount_interactive_widgets(widget_type)

    def left_panel_width(self) -> int:
        """Return the latest expanded left-panel width for persistence."""
        return self._splitter_left_saved

    def get_current_figure_for_pane(self, pane: str) -> Figure | None:
        """Borrow the presented run/analysis/post_analysis Figure, or None if empty.

        Unknown panes raise ValueError; unsupported panes or mounted widgets
        without a Figure raise RuntimeError. Data previews use this overridable
        reader dynamically and isolate each pane's synchronous failure.
        """
        return self._figure_area.get_current_figure_for_pane(pane)

    def clear_all_figures(self) -> None:
        """Clear all figure panes (used for LoadData and Run start invalidation)."""
        self._figure_area.clear_all_figures()

    def present_figures(self, pane: str, figures: FigureCollection) -> None:
        """Borrow operation names after owner-thread canvas attachment.

        Clear/replacement retires presentation references, not the original
        Figures. Unknown panes raise ValueError; rendering errors propagate.
        """
        self._figure_area.present_figures(pane, figures)

    def show_run_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the Run pane."""
        self._figure_area.show_run_figure(fig)

    def show_analysis_figures(self, figures: FigureCollection | None) -> None:
        """Draw named Analysis Figures, or clear for None/empty; raise if unsupported.

        The same collection identity preserves the selected name. A replacement
        rebuilds the selector with signals blocked. Attach/draw failures propagate.
        """
        self._figure_area.show_analysis_figures(figures)

    def show_post_analysis_figures(self, figures: FigureCollection | None) -> None:
        """Draw named Post Figures using the same identity/draw contract as Analysis.

        None/empty clears presentation. Unsupported Post raises RuntimeError.
        """
        self._figure_area.show_post_analysis_figures(figures)

    def _on_refresh_sources_clicked(self) -> None:
        try:
            self._cfg.refresh_sources()
        except Exception as exc:
            # The Qt callback is the presentation boundary; the resource keeps
            # the previous form on capture/factory failure so users can retry.
            logger.exception("Source refresh failed for %s", self.tab_id)
            self._dialog_presenter.critical(self, "Refresh sources failed", str(exc))
            return
        assert self._actions is not None, "refresh clicked before bind"
        self._actions.refresh_interaction(self.tab_id)

    def _on_reset_cfg_clicked(self) -> None:
        confirmed = self._dialog_presenter.confirm(
            self,
            "Reset config",
            "Reset config to defaults? This discards the current configuration.",
            default=False,
        )
        if not confirmed:
            return
        assert self._actions is not None, "reset clicked before bind"
        try:
            self._cfg.reset()
        except Exception as exc:
            if self._cfg.unavailable_reason is None:
                raise
            self._dialog_presenter.critical(self, "Reset config failed", str(exc))
            return
        self._actions.refresh_interaction(self.tab_id)

    # ── cfg form editor over the tab's form resource ───────────────────────

    def cfg_invalid_reason(self) -> str | None:
        """Return the first message blocking Run on the displayed config, or None.

        Unparseable text and input the resource rejected count first, then the
        resource's model and lowering errors. Raises RuntimeError once detached.
        """
        if self._cfg.unavailable_reason is not None:
            return self._cfg.unavailable_reason
        errors = self._require_editor().errors()
        if errors is None:
            result = self._cfg.result()
            errors = result if isinstance(result, FormErrors) else None
        if errors is None:
            return None
        path, message = next(iter(errors.messages.items()))
        label = ".".join(str(part) for part in path)
        return f"{label}: {message}" if label else message

    def _follow_cfg(self) -> None:
        """Mount an editor over the resource's current state and watch for swaps."""
        if self._cfg_unsub is not None:
            return
        self._mount_editor()
        self._cfg_unsub = self._cfg.subscribe(self._on_cfg_published)

    def _unfollow_cfg(self) -> None:
        if self._cfg_unsub is not None:
            self._cfg_unsub()
            self._cfg_unsub = None
        if self._editor is not None:
            self._editor.detach()

    def _mount_editor(self) -> None:
        if self._editor is not None:
            self._editor.detach()
            self._form_scroll.takeWidget()
            self._editor.deleteLater()
            self._editor = None
        elif (previous := self._form_scroll.takeWidget()) is not None:
            previous.deleteLater()
        reason = self._cfg.unavailable_reason
        if reason is not None:
            message = QLabel(reason)
            message.setObjectName("history.form_error")
            message.setWordWrap(True)
            message.setTextFormat(Qt.TextFormat.PlainText)
            self._form_scroll.setWidget(message)
            self._editor_state = None
            return
        state = self._cfg.editor_state
        # Writes and option commands go to the resource unchanged, so its busy,
        # lifetime and notification guards and its subscriber publication
        # cover editor input; the editor never changes the borrowed state.
        editor = FormEditor(
            state,
            self._cfg.env,
            write=self._cfg.write,
            select=self._cfg.select,
            run_command=self._cfg.run_command,
        )
        editor.setContentsMargins(4, 4, 4, 4)
        editor.validity_changed.connect(self._on_cfg_validity_changed)
        editor.set_editing_enabled(enabled=self._editing_enabled)
        self._editor = editor
        self._editor_state = state
        self._form_scroll.setWidget(editor)

    def _on_cfg_published(self) -> None:
        if (
            self._cfg.unavailable_reason is not None
            or self._cfg.editor_state is not self._editor_state
        ):
            self._mount_editor()
        # Every committed write re-evaluates Run admission, like a snapshot
        # change: a valid edit must not leave a stale Run gate behind.
        if self._actions is not None:
            self._actions.refresh_interaction(self.tab_id)

    def _on_cfg_validity_changed(self) -> None:
        if self._actions is not None:
            self._actions.refresh_interaction(self.tab_id)

    def _require_editor(self) -> FormEditor[FormModel]:
        if self._editor is None:
            raise RuntimeError(f"tab {self.tab_id!r} has no form editor")
        return self._editor

    def _on_left_tab_changed(self, index: int) -> None:
        """Translate the left selection into the corresponding figure-area page."""
        widget = self._left_tabs.widget(index)
        if widget is self._run_panel:
            self._figure_area.select_pane("run")
        elif self._has_analysis and widget is self._analysis_panel:
            self._figure_area.select_pane("analysis")
        elif self._has_post and widget is self._post_panel:
            self._figure_area.select_pane("post_analysis")
        elif widget is self._save_panel:
            self._figure_area.select_pane("data")
        else:
            self._figure_area.select_pane("guide")

    def set_writable(self, *, writable: bool) -> None:
        """Receive Local write availability; domain and Stop gates remain separate."""
        self._writable = writable
        self._figure_area.set_writable(writable=writable)
        if self._actions is not None:
            self.update_interaction_state(
                self._ctrl.tab_control.get_tab_snapshot(self.tab_id)
            )

    def _update_run_action_row(
        self,
        state: TabInteractionState,
        capabilities: AdapterCapabilities,
        *,
        writable: bool,
        local_busy: bool,
    ) -> None:
        """Project idle source actions or running Stop without changing commands."""
        # Status-free action row (A5): no readiness text, only button enablement + tooltip.
        if state.is_running:
            self.refresh_btn.setVisible(False)
            self.reset_btn.setVisible(False)
            self._run_action_layout.setStretch(0, 0)
            self._run_action_layout.setStretch(1, 0)
            self._run_action_layout.setStretch(2, 100)
            self.run_btn.setText("Stop")
            self.run_btn.setEnabled(True)
            self.run_btn.setToolTip("Running")
            self.run_btn.setStyleSheet(_RED_STOP_STYLESHEET)
            self.run_btn.setObjectName("")
        else:
            self.refresh_btn.setVisible(True)
            self.reset_btn.setVisible(True)
            self._run_action_layout.setStretch(0, 15)
            self._run_action_layout.setStretch(1, 15)
            self._run_action_layout.setStretch(2, 70)
            self.run_btn.setText("Run")
            self.run_btn.setObjectName("primaryButton")
            self._update_run_admission(
                state, capabilities, writable=writable, local_busy=local_busy
            )
            # Apply explicit blue primary style (idle/disabled) — objectName alone has no global stylesheet
            self.run_btn.setStyleSheet(BLUE_PRIMARY_STYLESHEET)
            # Force style refresh for objectName/stylesheet change
            # tolerated: The initialized button has a live widget/application style after setStyleSheet.
            self.run_btn.style().unpolish(self.run_btn)  # pyright: ignore[reportOptionalMemberAccess]
            # tolerated: The initialized button has a live widget/application style after setStyleSheet.
            self.run_btn.style().polish(self.run_btn)  # pyright: ignore[reportOptionalMemberAccess]
            self.run_btn.update()

    def update_interaction_state(self, snapshot: TabSnapshot) -> None:
        """Refresh action, editing, and save gates on the Qt owner thread.

        The snapshot must contain interaction facts and the capabilities used
        to construct this widget. Missing or mismatched capabilities, or a
        missing Save pane, raise RuntimeError. Missing interaction is an invariant
        failure. The main window's pushed availability also gates local editing.
        """
        assert snapshot.interaction is not None
        if snapshot.capabilities is None:
            raise RuntimeError(
                f"render snapshot for tab {self.tab_id!r} has no capabilities"
            )
        if snapshot.capabilities != self._capabilities:
            raise RuntimeError(
                f"capability mismatch for tab {self.tab_id!r}: "
                f"widget {self._capabilities!r} vs snapshot {snapshot.capabilities!r}"
            )
        issue = snapshot.run.analysis_source_issue if snapshot.run is not None else None
        self._analysis_source_notice.setText(
            f"History loaded, no analysis source: {issue}" if issue else ""
        )
        self._analysis_source_notice.setVisible(issue is not None)
        state = snapshot.interaction
        capabilities = snapshot.capabilities
        writable = self._writable
        local_busy = state.is_running or state.is_analyzing
        self._update_run_action_row(
            state, capabilities, writable=writable, local_busy=local_busy
        )

        idle = writable and not local_busy
        self._editing_enabled = idle
        if self._editor is not None:
            self._editor.set_editing_enabled(enabled=idle)
        self.reset_btn.setEnabled(idle)
        self.refresh_btn.setEnabled(idle)

        self._update_analysis_actions(state, idle=idle)
        # The center renders the shared draft and State-owned artifact status.
        if snapshot.save is None:
            raise RuntimeError(
                f"render snapshot for tab {self.tab_id!r} has no save pane"
            )
        self._save_center.set_comment_text(snapshot.save.comment)
        self._save_center.update_interaction(snapshot, writable=writable)

    def _update_analysis_actions(
        self, state: TabInteractionState, *, idle: bool
    ) -> None:
        """Project analysis and writeback gates for the constructed panes.

        idle means the tab is writable and neither running nor analyzing.
        Form validity gates Analyze only when parameters are populated;
        readiness and context retain their existing action-specific gates.
        """
        if self._has_analysis:
            self.analyze_btn.setEnabled(
                idle
                and state.has_context
                and state.has_run_result
                and (not self.analyze_form.has_params() or self.analyze_form.is_valid())
            )
            self.analyze_form.setEnabled(idle)
            self.writeback_widget.setEnabled(
                idle and state.has_context and state.has_analyze_result
            )
            self.writeback_widget.setToolTip(
                "" if state.has_context else "Select or create a working point first"
            )
        if self._has_post:
            post_enabled = idle and state.has_analyze_result
            self.post_analyze_form.setEnabled(post_enabled)
            self.post_analyze_btn.setEnabled(
                post_enabled
                and (
                    not self.post_analyze_form.has_params()
                    or self.post_analyze_form.is_valid()
                )
            )
            self._post_gate_label.setVisible(not state.has_analyze_result)
            self.post_writeback_widget.setEnabled(
                idle and state.has_context and state.has_post_analyze_result
            )
            self.post_writeback_widget.setToolTip(
                "" if state.has_context else "Select or create a working point first"
            )

    def _update_run_admission(
        self,
        state: TabInteractionState,
        capabilities: AdapterCapabilities,
        *,
        writable: bool,
        local_busy: bool,
    ) -> None:
        """Enable idle Run and project the first blocking reason in UI priority order."""
        invalid_reason = self.cfg_invalid_reason()
        cfg_valid = invalid_reason is None
        self._run_cfg_notice.setText(
            "" if cfg_valid else f"Config invalid: {invalid_reason}"
        )
        self._run_cfg_notice.setVisible(
            not cfg_valid and self._cfg.unavailable_reason is None
        )
        can_run = (
            writable
            and not local_busy
            and not state.global_run_active
            and not state.startup_active
            and state.has_active_context
            and (not capabilities.requires_soc or state.has_soc)
            and cfg_valid
        )
        self.run_btn.setEnabled(can_run)
        if can_run:
            self.run_btn.setToolTip("")
        elif local_busy:
            self.run_btn.setToolTip("Tab is busy")
        elif state.startup_active:
            self.run_btn.setToolTip("Startup script is running")
        elif state.global_run_active:
            self.run_btn.setToolTip("Another tab is running")
        elif self._cfg.unavailable_reason is not None:
            self.run_btn.setToolTip(self._cfg.unavailable_reason)
        elif not state.has_active_context:
            self.run_btn.setToolTip("Select or create a working point first")
        elif capabilities.requires_soc and not state.has_soc:
            self.run_btn.setToolTip("No SoC connection")
        elif invalid_reason is not None:
            self.run_btn.setToolTip(f"Config invalid: {invalid_reason}")

    def _bind_to_controller(self, actions: TabActions) -> None:
        tab_id = self.tab_id
        self._actions = actions

        if self._has_analysis:
            self.analyze_form.validity_changed.connect(
                lambda _valid: actions.refresh_interaction(tab_id)
            )
            self.analyze_form.params_changed.connect(
                lambda instance: self._ctrl.update_tab_analyze_param_instance(
                    tab_id, instance
                )
            )
        if self._has_post:
            self.post_analyze_form.validity_changed.connect(
                lambda _valid: actions.refresh_interaction(tab_id)
            )
            self.post_analyze_form.params_changed.connect(
                lambda instance: self._ctrl.update_tab_post_analyze_param_instance(
                    tab_id, instance
                )
            )
        self.reset_btn.clicked.connect(self._on_reset_cfg_clicked)
        self.refresh_btn.clicked.connect(self._on_refresh_sources_clicked)
        self.run_btn.clicked.connect(lambda: actions.run_or_stop(tab_id))
        if self._capabilities.load_data:
            self._save_center.bind_load(lambda: actions.load_data(tab_id))
        self._save_center.bind_save_all(lambda: actions.save_all(tab_id))
        if self._has_analysis:
            self.analyze_btn.clicked.connect(lambda: actions.analyze(tab_id))
            self.writeback_widget.apply_requested.connect(
                lambda: actions.apply_writeback(tab_id)
            )
        if self._has_post:
            self.post_analyze_btn.clicked.connect(lambda: actions.post_analyze(tab_id))
            self.post_writeback_widget.apply_requested.connect(
                lambda: actions.apply_post_writeback(tab_id)
            )
        self._save_center.bind_artifact_save(
            lambda ref: actions.save_artifact(tab_id, ref)
        )

    def _on_progress_changed(self) -> None:
        models = tuple(m for _, m in self._progress_control.progress_bars(self.tab_id))
        self.progress_stack.render_models(models)

    def detach(self) -> None:
        """Tear this tab widget down."""
        if self._actions is None:
            raise RuntimeError(f"tab {self.tab_id!r} is not attached")
        self._progress_unsub()
        self._unfollow_cfg()
        self._actions = None
