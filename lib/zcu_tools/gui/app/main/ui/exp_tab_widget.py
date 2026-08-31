"""Per-experiment tab widget for the measure-gui main window - capability driven."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.events.completion import SaveDataFinishedPayload
from zcu_tools.gui.app.main.ui.artifact_save_center import (
    ArtifactKind,
    ArtifactSaveCenter,
)
from zcu_tools.gui.app.main.ui.cfg_binding import make_value_source_input_enhancer
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.plotting import FigureContainer, attach_existing_figure_to_container
from zcu_tools.gui.session.ui.progress_stack import ProgressStack
from zcu_tools.gui.widgets import DialogPresenter, QtDialogPresenter
from zcu_tools.gui.widgets.cfg import CfgFormWidget
from zcu_tools.gui.widgets.cfg.fields.containers import _CollapsibleSection

logger = logging.getLogger(__name__)

# Approved prototype blue primary treatment (A3)
_BLUE_PRIMARY_STYLESHEET = (
    "QPushButton#primaryButton { background-color: #286ac7; color: white; "
    "font-weight: 600; border: 1px solid #205aa9; border-radius: 4px; }"
    "QPushButton#primaryButton:disabled { background-color: #a0b8d9; color: #e6edf7; border-color: #8da6c9; }"
    "QPushButton#primaryButton:hover:!disabled { background-color: #2f76dc; }"
)
_RED_STOP_STYLESHEET = (
    "background-color: #f44336; color: white; font-weight: bold; "
    "border: 1px solid #d32f2f; border-radius: 4px;"
)

from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtGui import (  # type: ignore[attr-defined]
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analyze_form import AnalyzeFormWidget
from .data_figure_preview_gallery import DataFigurePreviewGallery
from .writeback_widget import WritebackWidget

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.main.adapter import WritebackItem
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.events.completion import SaveDataFinishedPayload
    from zcu_tools.gui.app.main.services import TabSnapshot


class TabActions(Protocol):
    """Tab-level actions supplied by the top-level window boundary."""

    def refresh_interaction(self, tab_id: str) -> None: ...

    def run_or_stop(self, tab_id: str) -> None: ...

    def load_data(self, tab_id: str) -> None: ...

    def analyze(self, tab_id: str) -> None: ...

    def post_analyze(self, tab_id: str) -> None: ...

    def apply_writeback(self, tab_id: str) -> None: ...

    def apply_post_writeback(self, tab_id: str) -> None: ...

    def save_data(self, tab_id: str) -> None: ...

    def save_image(self, tab_id: str) -> None: ...

    def save_post_image(self, tab_id: str) -> None: ...

    def save_all(self, tab_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Per-experiment tab widget
# ---------------------------------------------------------------------------


class _PanelEdgeHandle(QToolButton):
    """Boundary handle for collapsing/expanding the left panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(16, 42)
        self.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
        self.setAutoRaise(True)
        self._collapsed = False

    def set_collapsed(self, collapsed: bool) -> None:
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


class _LedgerSection(QWidget):
    """App-local single-column ledger section with whole-row folding.

    Header row (blank or text area) toggles the body; the action bar stays
    fixed outside the scroll area. Presentation is app-local; the shared tree
    adapter remains the only cross-module structural seam (S1).
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        collapsed: bool = False,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header — whole row is clickable (S1/A2).
        self._header = QWidget()
        self._header.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
        self._header.setObjectName("ledgerHeader")
        header_row = QHBoxLayout(self._header)
        header_row.setContentsMargins(4, 4, 4, 4)
        header_row.setSpacing(6)

        self._toggle_btn = QPushButton("▼" if not collapsed else "▶")
        self._toggle_btn.setFixedWidth(16)
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(not collapsed)
        header_row.addWidget(self._toggle_btn)

        self._title_label = QLabel(f"<b>{title}</b>")
        font = self._title_label.font()
        font.setPixelSize(13)
        self._title_label.setFont(font)
        header_row.addWidget(self._title_label, stretch=1)

        # Body
        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(8, 2, 0, 2)
        self.body_layout.setSpacing(2)

        outer.addWidget(self._header)
        outer.addWidget(self._body)

        self._collapsed = collapsed
        self._body.setVisible(not collapsed)

        # Whole header row toggles; button also toggles (no double toggle because
        # click on button goes to button widget, not header).
        self._toggle_btn.clicked.connect(self._toggle)
        # Install header click handler via event filter on header.
        self._header.mouseReleaseEvent = self._on_header_mouse_release  # type: ignore[method-assign]

    def _on_header_mouse_release(self, event) -> None:
        try:
            if event.button() == Qt.LeftButton:  # type: ignore[attr-defined]
                self._toggle()
                event.accept()
                return
        except Exception:
            pass
        # Fallback to default
        QWidget.mouseReleaseEvent(self._header, event)

    def _toggle(self, *_: object) -> None:
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self._toggle_btn.setText("▶" if self._collapsed else "▼")
        self._toggle_btn.setChecked(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        if bool(collapsed) == self._collapsed:
            return
        self._toggle()

    def is_collapsed(self) -> bool:
        return self._collapsed


class ExpTabWidget(QWidget):
    """A single experiment tab with capability-driven subtabs and independent figure panes."""

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
        if not isinstance(capabilities, AdapterCapabilities):
            raise TypeError(
                f"ExpTabWidget requires AdapterCapabilities, got {type(capabilities).__name__!r}"
            )
        self.tab_id = tab_id
        self._ctrl = ctrl
        self._capabilities = capabilities
        self._has_analysis = capabilities.analysis is not AnalysisMode.NONE
        self._has_post = bool(capabilities.post_analysis)
        self._dialog_presenter = dialog_presenter or QtDialogPresenter()
        self._progress_control = ctrl.progress_control
        # editor_id of this tab's shared cfg-editor session
        self._cfg_editor_id: str | None = None
        # The action boundary is retained for Reset; button slots close over it.
        self._actions: TabActions | None = None
        # Optional injected Figure->PNG renderer for Data preview (tests)
        self._preview_renderer = preview_renderer

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(2)

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
        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]

        content_row.addWidget(splitter, stretch=1)

        self._splitter = splitter
        self._splitter_left_saved = ctrl.get_persisted_startup().left_panel_width
        self._left_panel_collapsed = False
        self._splitter.splitterMoved.connect(self._on_splitter_moved)

        # ── Left pane: QTabWidget with capability-driven tabs ──────
        self._left_tabs = QTabWidget()

        self._left_edge_handle = _PanelEdgeHandle(self._content_widget)
        self._left_edge_handle.clicked.connect(self._toggle_left_panel)

        # ── Tab: Run (always) ──────────────────────────────────────
        run_panel = QWidget()
        run_layout = QVBoxLayout(run_panel)
        run_layout.setContentsMargins(4, 4, 4, 4)
        run_layout.setSpacing(2)

        self.cfg_form = CfgFormWidget(
            text_input_enhancer=make_value_source_input_enhancer(ctrl),
        )
        run_layout.addWidget(self.cfg_form, stretch=1)

        # Run action row: status-free, Reset 20% / Run 80% of available width
        # (A5). Both buttons expand proportionally; Run retains blue primary
        # and Stop retains red active semantics.
        self._run_action_bar = QFrame()
        self._run_action_bar.setObjectName("runActionBar")
        self._run_action_bar.setFrameShape(QFrame.Shape.StyledPanel)  # type: ignore[attr-defined]
        bar_layout = QHBoxLayout(self._run_action_bar)
        bar_layout.setContentsMargins(8, 6, 8, 6)
        bar_layout.setSpacing(6)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFlat(True)
        self.reset_btn.setToolTip("Discard current config and restore adapter defaults")
        reset_font = self.reset_btn.font()
        reset_font.setPointSize(max(reset_font.pointSize() - 1, 7))
        self.reset_btn.setFont(reset_font)
        self.reset_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )  # type: ignore[attr-defined]
        bar_layout.addWidget(self.reset_btn, stretch=20)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setFixedHeight(30)
        self.run_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )  # type: ignore[attr-defined]
        self.run_btn.setStyleSheet(_BLUE_PRIMARY_STYLESHEET)
        bar_layout.addWidget(self.run_btn, stretch=80)
        run_layout.addWidget(self._run_action_bar)
        self._run_panel = run_panel
        self._left_tabs.addTab(run_panel, "Run")

        # ── Tab: Analysis (only when analysis capability present) ──
        if self._has_analysis:
            # Single-column 13 px ledger with whole-header folding; Analyze sits
            # immediately after params and before Writeback preview (A4).
            analysis_container = QWidget()
            analysis_outer = QVBoxLayout(analysis_container)
            analysis_outer.setContentsMargins(0, 0, 0, 0)
            analysis_outer.setSpacing(0)

            analysis_scroll = QScrollArea()
            analysis_scroll.setWidgetResizable(True)
            analysis_scroll.setFrameShape(QFrame.Shape.NoFrame)  # type: ignore[attr-defined]
            analysis_inner = QWidget()
            analysis_layout = QVBoxLayout(analysis_inner)
            analysis_layout.setContentsMargins(4, 4, 4, 4)
            analysis_layout.setSpacing(4)
            analysis_layout.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]

            self._analyze_section = _LedgerSection(
                "Analysis parameters", collapsed=False
            )
            self.analyze_form = AnalyzeFormWidget()
            font = self.analyze_form.font()
            font.setPixelSize(13)
            self.analyze_form.setFont(font)
            self._analyze_section.body_layout.addWidget(self.analyze_form)
            analysis_layout.addWidget(self._analyze_section)

            # Analyze immediately below params and before Writeback preview,
            # occupying 100% of available content width (A6). Stays enabled
            # per availability/busy and retains blue primary when idle.
            self.analyze_btn = QPushButton("Analyze")
            self.analyze_btn.setObjectName("primaryButton")
            self.analyze_btn.setFixedHeight(30)
            self.analyze_btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )  # type: ignore[attr-defined]
            self.analyze_btn.setStyleSheet(_BLUE_PRIMARY_STYLESHEET)
            analysis_layout.addWidget(self.analyze_btn)

            self.writeback_section = _LedgerSection(
                "Writeback preview", collapsed=False
            )
            self.writeback_widget = WritebackWidget(
                self._ctrl, tab_id=self.tab_id, pane="analysis"
            )
            self.writeback_section.body_layout.addWidget(self.writeback_widget)
            self.writeback_section.setVisible(False)
            analysis_layout.addWidget(self.writeback_section)

            analysis_layout.addStretch()
            analysis_scroll.setWidget(analysis_inner)
            analysis_outer.addWidget(analysis_scroll, stretch=1)

            self._analysis_panel = analysis_container
            self._analysis_tab_index = self._left_tabs.addTab(
                analysis_container, "Analysis"
            )
        # ── Tab: Post-Analysis (only when post capability true) ────
        if self._has_post:
            post_scroll = QScrollArea()
            post_scroll.setWidgetResizable(True)
            post_inner = QWidget()
            post_layout = QVBoxLayout(post_inner)
            post_layout.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]

            self._post_analyze_section = _CollapsibleSection(
                "Post-Analysis", collapsible=True, collapsed=False
            )
            self.post_analyze_form = AnalyzeFormWidget()
            self._post_analyze_section.body_layout.addWidget(self.post_analyze_form)
            post_layout.addWidget(self._post_analyze_section)

            # Gate hint shown until a primary analyze result exists (form/Run disabled).
            self._post_gate_label = QLabel("Run analyze first to enable post-analysis.")
            self._post_gate_label.setWordWrap(True)
            self._post_gate_label.setStyleSheet("color: gray;")
            post_layout.addWidget(self._post_gate_label)

            self.post_analyze_btn = QPushButton("Run Post-Analysis")
            post_layout.addWidget(self.post_analyze_btn)

            # Post writeback
            self.post_writeback_section = _CollapsibleSection(
                "Writeback", collapsible=True, collapsed=False
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
            post_scroll.setWidget(post_inner)
            self._post_panel = post_scroll
            self._post_tab_index = self._left_tabs.addTab(post_scroll, "Post-Analysis")

        # ── Tab: Data (always) — save center ──────────────────────
        self._save_center = ArtifactSaveCenter(self.tab_id, capabilities)
        save_scroll = QScrollArea()
        save_scroll.setWidgetResizable(True)
        save_scroll.setWidget(self._save_center)
        self._save_panel = save_scroll
        self._left_tabs.addTab(save_scroll, "Data")

        # ── Tab: Guide (always) ──────────────────────────────────
        guide_scroll = QScrollArea()
        guide_scroll.setWidgetResizable(True)
        guide_label = QLabel()
        guide_label.setWordWrap(True)
        guide_label.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        guide_label.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        guide_label.setContentsMargins(8, 8, 8, 8)
        guide_label.setText(self._render_guide_html())
        guide_scroll.setWidget(guide_label)
        self._guide_panel = guide_scroll
        self._left_tabs.addTab(guide_scroll, "Guide")

        splitter.addWidget(self._left_tabs)

        # ── Right pane: per-pane figure containers ────────────────
        plot_panel = QWidget()
        self._plot_layout = QVBoxLayout(plot_panel)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._right_stack = QStackedWidget()

        # Run figure pane (always)
        self._run_stack = QStackedWidget()
        self._run_placeholder = QLabel("(no plot yet)")
        self._run_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._run_stack.addWidget(self._run_placeholder)
        self._run_container = FigureContainer(self._run_stack, self._run_placeholder)

        # Analysis figure pane (only when analysis present)
        if self._has_analysis:
            self._analysis_stack = QStackedWidget()
            self._analysis_placeholder = QLabel("(no plot yet)")
            self._analysis_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
            self._analysis_stack.addWidget(self._analysis_placeholder)
            self._analysis_container = FigureContainer(
                self._analysis_stack, self._analysis_placeholder
            )

        # Post figure pane (only when post present)
        if self._has_post:
            self._post_stack = QStackedWidget()
            self._post_placeholder = QLabel("(no plot yet)")
            self._post_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
            self._post_stack.addWidget(self._post_placeholder)
            self._post_container = FigureContainer(
                self._post_stack, self._post_placeholder
            )

        # Data preview gallery — Variant A stacked rail (S1, S3)
        self._data_gallery = DataFigurePreviewGallery(
            self._capabilities,
            renderer=self._preview_renderer,  # type: ignore[arg-type]
        )

        # Placeholder for Guide (no figure) — Data now shows gallery (S4)
        self._right_placeholder = QLabel("(no plot yet)")
        self._right_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]

        self._right_stack.addWidget(self._run_stack)
        if self._has_analysis:
            self._right_stack.addWidget(self._analysis_stack)
        if self._has_post:
            self._right_stack.addWidget(self._post_stack)
        self._right_stack.addWidget(self._data_gallery)
        self._right_stack.addWidget(self._right_placeholder)

        self._plot_layout.addWidget(self._right_stack, stretch=1)
        splitter.addWidget(plot_panel)

        splitter.setCollapsible(0, True)
        self._update_left_panel_controls()
        self._schedule_handle_layout()
        # Right pane follows left selection
        self._left_tabs.currentChanged.connect(self._on_left_tab_changed)
        # Initially show Run
        self._on_left_tab_changed(self._left_tabs.currentIndex())

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    def _require_analysis(self) -> None:
        if not self._has_analysis:
            raise RuntimeError(f"tab {self.tab_id!r} does not support analysis")

    def _require_post(self) -> None:
        if not self._has_post:
            raise RuntimeError(f"tab {self.tab_id!r} does not support post-analysis")

    # ------------------------------------------------------------------
    # Docked feedback panel host (ADR-0025 C3)
    # ------------------------------------------------------------------

    def mount_feedback_panel(self, panel: QWidget) -> None:
        """Dock the feedback panel directly below the figure (idempotent)."""
        if self._plot_layout.indexOf(panel) != -1:
            return
        self._plot_layout.insertWidget(1, panel)
        panel.show()

    def unmount_feedback_panel(self, panel: QWidget) -> None:
        """Remove the feedback panel from the plot column (idempotent)."""
        if self._plot_layout.indexOf(panel) == -1:
            return
        self._plot_layout.removeWidget(panel)
        panel.setParent(None)  # type: ignore[arg-type]

    def resizeEvent(self, a0) -> None:
        super().resizeEvent(a0)
        self._fix_splitter_on_resize()
        self._schedule_handle_layout()

    def _render_guide_html(self) -> str:
        """Render this adapter's static AdapterGuide as read-only rich text."""
        import html

        adapter_name = self._ctrl.get_tab_adapter_name(self.tab_id)
        guide = self._ctrl.get_adapter_guide(adapter_name)
        sections = [
            ("Behavior", guide.get("behavior", "")),
            ("Expects (MetaDict)", guide.get("expects_md", "")),
            ("Expects (ModuleLibrary)", guide.get("expects_ml", "")),
            ("Typical writeback", guide.get("typical_writeback", "")),
            ("Recommended", guide.get("recommended", "")),
        ]
        parts = [
            f"<p><b>{title}</b><br>{html.escape(body)}</p>"
            for title, body in sections
            if body
        ]
        if not parts:
            return "<p><i>No guide written for this adapter yet.</i></p>"
        return "".join(parts)

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
        self._left_edge_handle.set_collapsed(self._left_panel_collapsed)
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
        self._populate_cfg(snapshot.cfg_schema, self._ctrl)
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
        assert snapshot.paths is not None
        self.set_data_path(snapshot.paths.data.path or "")
        if self._has_analysis:
            self.set_analysis_image_path(snapshot.paths.analysis_image.path or "")
        if self._has_post:
            self.set_post_image_path(snapshot.paths.post_analysis_image.path or "")
        self.update_interaction_state(snapshot)
        self._bind_to_controller(actions)

    def _populate_cfg(self, schema: CfgSchema, ctrl: Controller) -> None:
        editor_id, _ = ctrl.open_seeded_cfg_editor(
            schema, gc=False, owner_key=self.tab_id
        )
        self._cfg_editor_id = editor_id
        self.cfg_form.attach(ctrl.get_cfg_editor_draft(editor_id))

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
        has_fields = instance is not None and bool(
            dataclasses.fields(cast(Any, instance))
        )
        self._post_analyze_section.setVisible(has_fields)

    def read_post_analyze_params(self) -> object:
        self._require_post()
        return self.post_analyze_form.read_params()

    def has_post_analyze_params(self) -> bool:
        if not self._has_post:
            return False
        return self.post_analyze_form.has_params()

    def update_writeback_items(self, items: list[WritebackItem]) -> None:
        self._require_analysis()
        self.writeback_widget.populate(items)
        self.writeback_section.setVisible(len(items) > 0)

    def update_post_writeback_items(self, items: list[WritebackItem]) -> None:
        self._require_post()
        self.post_writeback_widget.populate(items)
        self.post_writeback_section.setVisible(len(items) > 0)

    def set_data_path(self, data_path: str) -> None:
        self._save_center.set_data_path(data_path)

    def set_analysis_image_path(self, image_path: str) -> None:
        self._require_analysis()
        self._save_center.set_analysis_path(image_path)

    def set_post_image_path(self, image_path: str) -> None:
        self._require_post()
        self._save_center.set_post_analysis_path(image_path)

    def get_data_path(self) -> str:
        return self._save_center.get_data_path()

    def get_image_path(self) -> str:
        self._require_analysis()
        return self._save_center.get_analysis_path()

    def get_post_image_path(self) -> str:
        self._require_post()
        return self._save_center.get_post_analysis_path()

    def get_comment(self) -> str:
        return self._save_center.get_comment()

    # -- Data save center status delegation (S3) ------------------

    def notify_save_started(self, kind: ArtifactKind) -> None:
        """Capture pending signature for ``kind``."""
        self._save_center.notify_save_started(kind)

    def notify_save_succeeded(self, kind: ArtifactKind) -> None:
        self._save_center.notify_save_succeeded(kind)

    def notify_save_failed(self, kind: ArtifactKind) -> None:
        self._save_center.notify_save_failed(kind)

    def handle_save_data_finished(self, payload: SaveDataFinishedPayload) -> None:
        """Apply async data terminal outcome (error None => success)."""
        self._save_center.handle_data_finished(payload.error)

    def ordered_saveable_kinds(self, snapshot: TabSnapshot) -> list[ArtifactKind]:
        """Ordered saveable artifacts for Save All (snapshot single-fetch)."""
        return self._save_center.ordered_saveable_kinds(snapshot)

    def focus_result_panel(self) -> None:
        """Focus Analysis when supported, otherwise focus Save."""
        if self._has_analysis:
            self._left_tabs.setCurrentWidget(self._analysis_panel)
            return
        self._left_tabs.setCurrentWidget(self._save_panel)

    # ── Figure container helpers (stable per-subtab identity S2) ──────────

    def get_run_container(self) -> FigureContainer:
        return self._run_container

    def get_analysis_container(self) -> FigureContainer:
        self._require_analysis()
        return self._analysis_container

    def get_post_container(self) -> FigureContainer:
        self._require_post()
        return self._post_container

    def prepare_run_container(self) -> FigureContainer:
        """Clear Run and every downstream presentation for a new run."""
        self._run_container.clear_dynamic_canvases()
        if self._has_analysis:
            self._analysis_container.clear_dynamic_canvases()
        if self._has_post:
            self._post_container.clear_dynamic_canvases()
        self._refresh_data_gallery()
        return self._run_container

    def prepare_analysis_container(self) -> FigureContainer:
        """Clear Analysis presentation and return its container."""
        self._require_analysis()
        self._analysis_container.clear_dynamic_canvases()
        self._refresh_data_gallery()
        return self._analysis_container

    def prepare_post_container(self) -> FigureContainer:
        """Clear Post presentation and return its container."""
        self.clear_post_figure()
        return self._post_container

    def clear_post_figure(self) -> None:
        """Clear only the invalidated Post-Analysis presentation."""
        self._require_post()
        self._post_container.clear_dynamic_canvases()
        self._refresh_data_gallery()

    def mount_interactive_widget(self, widget: QWidget) -> None:
        """Mount an interactive analysis widget as the visible plot content (analysis pane)."""
        self._require_analysis()
        self._analysis_stack.addWidget(widget)
        self._analysis_stack.setCurrentWidget(widget)
        self._right_stack.setCurrentWidget(self._analysis_stack)

    def unmount_interactive_widgets(self, widget_type: type[QWidget]) -> None:
        """Remove interactive widgets of ``widget_type`` and show placeholder in analysis pane."""
        self._require_analysis()
        for index in reversed(range(self._analysis_stack.count())):
            widget = self._analysis_stack.widget(index)
            if isinstance(widget, widget_type):
                self._analysis_stack.removeWidget(widget)
                widget.deleteLater()
        self._analysis_stack.setCurrentWidget(self._analysis_placeholder)

    def left_panel_width(self) -> int:
        """Return the latest expanded left-panel width for persistence."""
        return self._splitter_left_saved

    def get_current_figure_for_pane(self, pane: str) -> Figure | None:
        """Pane-specific figure read (run|analysis|post_analysis)."""
        from matplotlib.figure import Figure

        if pane == "run":
            stack = self._run_stack
            placeholder = self._run_placeholder
        elif pane == "analysis":
            self._require_analysis()
            stack = self._analysis_stack  # type: ignore[attr-defined]
            placeholder = self._analysis_placeholder  # type: ignore[attr-defined]
        elif pane == "post_analysis":
            self._require_post()
            stack = self._post_stack  # type: ignore[attr-defined]
            placeholder = self._post_placeholder  # type: ignore[attr-defined]
        else:
            raise ValueError(f"unknown pane {pane!r}")
        canvas = stack.currentWidget()
        if canvas is None or canvas is placeholder:
            return None
        figure = getattr(canvas, "figure", None)
        if not isinstance(figure, Figure):
            raise RuntimeError(f"tab {self.tab_id!r} canvas has no matplotlib figure")
        return figure

    def clear_all_figures(self) -> None:
        """Clear all figure panes (used for LoadData and Run start invalidation)."""
        self._run_container.clear_dynamic_canvases()
        if self._has_analysis:
            self._analysis_container.clear_dynamic_canvases()
        if self._has_post:
            self._post_container.clear_dynamic_canvases()
        self._refresh_data_gallery()

    def show_run_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the Run pane."""
        canvas = attach_existing_figure_to_container(fig, self._run_container)
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached run canvas does not support draw()")
        draw()
        logger.debug("show_run_figure: tab_id=%r canvas set", self.tab_id)
        self._refresh_data_gallery()

    def show_analysis_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the Analysis pane and bring it to front."""
        self._require_analysis()
        canvas = attach_existing_figure_to_container(fig, self._analysis_container)
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached analysis canvas does not support draw()")
        draw()
        logger.debug("show_analysis_figure: tab_id=%r canvas set", self.tab_id)
        self._refresh_data_gallery()

    def show_post_analysis_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the Post-Analysis pane."""
        self._require_post()
        canvas = attach_existing_figure_to_container(fig, self._post_container)
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached post canvas does not support draw()")
        draw()
        logger.debug("show_post_analysis_figure: tab_id=%r canvas set", self.tab_id)
        self._refresh_data_gallery()

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
        schema = self._ctrl.reset_tab_cfg(self.tab_id)
        self._reseed_cfg(schema)
        self._actions.refresh_interaction(self.tab_id)

    def _reseed_cfg(self, schema: CfgSchema) -> None:
        self.cfg_form.detach()
        if self._cfg_editor_id is not None:
            self._ctrl.teardown_cfg_editor(self._cfg_editor_id)
            self._cfg_editor_id = None
        editor_id, _ = self._ctrl.open_seeded_cfg_editor(
            schema, gc=False, owner_key=self.tab_id
        )
        self._cfg_editor_id = editor_id
        self.cfg_form.attach(self._ctrl.get_cfg_editor_draft(editor_id))

    def _is_data_visible(self) -> bool:
        return self._left_tabs.currentWidget() is self._save_panel

    def _refresh_data_gallery(self) -> None:
        """Refresh Data gallery from current pane figures when Data is visible (S2)."""
        if not hasattr(self, "_data_gallery"):
            return
        if not self._is_data_visible():
            return
        try:
            run_fig = self.get_current_figure_for_pane("run")
        except Exception:
            run_fig = None
        if self._has_analysis:
            try:
                ana_fig = self.get_current_figure_for_pane("analysis")
            except Exception:
                ana_fig = None
        else:
            ana_fig = None
        if self._has_post:
            try:
                post_fig = self.get_current_figure_for_pane("post_analysis")
            except Exception:
                post_fig = None
        else:
            post_fig = None
        try:
            self._data_gallery.update_figures(run_fig, ana_fig, post_fig)
        except Exception:
            logger.exception("failed to refresh data gallery")

    def _on_left_tab_changed(self, index: int) -> None:
        """Switch right pane to match left subtab (S4).

        Data → gallery (with snapshot refresh), Guide → placeholder,
        Run/Analysis/Post → their source stacks.
        """
        widget = self._left_tabs.widget(index)
        if widget is self._run_panel:
            self._right_stack.setCurrentWidget(self._run_stack)
            return
        if self._has_analysis and widget is self._analysis_panel:
            self._right_stack.setCurrentWidget(self._analysis_stack)
            return
        if self._has_post and widget is self._post_panel:
            self._right_stack.setCurrentWidget(self._post_stack)
            return
        if widget is self._save_panel:
            self._refresh_data_gallery()
            self._right_stack.setCurrentWidget(self._data_gallery)
            # Viewport-driven mosaic must reflow by gallery's own width (S1);
            # QStackedWidget hides inactive pages so width changes while hidden
            # would otherwise defer resize/show until next event.
            try:
                self._data_gallery._arrange_cards()  # type: ignore[attr-defined]
            except Exception:
                pass
            return
        # Guide
        self._right_stack.setCurrentWidget(self._right_placeholder)

    def update_interaction_state(self, snapshot: TabSnapshot) -> None:
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
        state = snapshot.interaction
        capabilities = snapshot.capabilities
        local_busy = state.is_running or state.is_analyzing or state.is_saving_data
        # Status-free action row (A5): no readiness text, only button enablement + tooltip.
        if state.is_running:
            self.run_btn.setText("Stop")
            self.run_btn.setEnabled(True)
            self.run_btn.setToolTip("Running")
            self.run_btn.setStyleSheet(_RED_STOP_STYLESHEET)
            self.run_btn.setObjectName("")
        else:
            self.run_btn.setText("Run")
            self.run_btn.setObjectName("primaryButton")
            cfg_valid = self.cfg_form.is_valid()
            can_run = (
                not local_busy
                and not state.global_run_active
                and state.has_active_context
                and (not capabilities.requires_soc or state.has_soc)
                and cfg_valid
            )
            self.run_btn.setEnabled(can_run)
            if can_run:
                self.run_btn.setToolTip("")
            elif local_busy:
                self.run_btn.setToolTip("Tab is busy")
            elif state.global_run_active:
                self.run_btn.setToolTip("Another tab is running")
            elif not state.has_context:
                self.run_btn.setToolTip("No experiment context")
            elif not state.has_active_context:
                self.run_btn.setToolTip("Select or create a file-backed context")
            elif capabilities.requires_soc and not state.has_soc:
                self.run_btn.setToolTip("No SoC connection")
            elif not cfg_valid:
                reason = self.cfg_form.first_invalid_reason()
                self.run_btn.setToolTip(
                    f"Config invalid: {reason}" if reason else "Config invalid"
                )
            # Apply explicit blue primary style (idle/disabled) — objectName alone has no global stylesheet
            self.run_btn.setStyleSheet(_BLUE_PRIMARY_STYLESHEET)
            # Force style refresh for objectName/stylesheet change
            self.run_btn.style().unpolish(self.run_btn)  # type: ignore[attr-defined]
            self.run_btn.style().polish(self.run_btn)  # type: ignore[attr-defined]
            self.run_btn.update()

        idle = not local_busy
        self.cfg_form.set_editing_enabled(idle)
        self.reset_btn.setEnabled(idle)

        # Capability is already enforced by conditional construction; no tab visibility toggling.
        # Branch inner controls only for present panes.

        if self._has_analysis:
            self.analyze_btn.setEnabled(
                idle and state.has_context and state.has_run_result
            )
            self.analyze_form.setEnabled(idle)
            self.writeback_widget.setEnabled(
                idle and state.has_context and state.has_analyze_result
            )
        # Post pane controls (only if present)
        if self._has_post:
            post_enabled = idle and state.has_analyze_result
            self.post_analyze_form.setEnabled(post_enabled)
            self.post_analyze_btn.setEnabled(post_enabled)
            self._post_gate_label.setVisible(not state.has_analyze_result)
            self.post_writeback_widget.setEnabled(
                idle and state.has_context and state.has_post_analyze_result
            )
        # Data save center owns all save-row enablement and status.
        self._save_center.update_interaction(snapshot)

    def _bind_to_controller(self, actions: TabActions) -> None:
        tab_id = self.tab_id
        self._actions = actions

        def validity_cb(_valid: bool) -> None:
            actions.refresh_interaction(tab_id)

        def schema_cb(schema_obj: CfgSchema) -> None:
            self._ctrl.update_tab_cfg(tab_id, schema_obj)

        def data_path_cb(_text: str) -> None:
            data_path = self.get_data_path()
            self._ctrl.update_tab_data_path(tab_id, data_path if data_path else None)

        def analysis_image_cb(_text: str) -> None:
            image_path = self.get_image_path()
            self._ctrl.update_tab_analysis_image_path(
                tab_id, image_path if image_path else None
            )

        def post_image_cb(_text: str) -> None:
            image_path = self.get_post_image_path()
            self._ctrl.update_tab_post_analysis_image_path(
                tab_id, image_path if image_path else None
            )

        self.cfg_form.validity_changed.connect(validity_cb)
        self.cfg_form.schema_changed.connect(schema_cb)

        self._save_center.bind_data_path_changed(data_path_cb)
        if self._has_analysis:
            self.analyze_form.params_changed.connect(
                lambda instance: self._ctrl.update_tab_analyze_param_instance(
                    tab_id, instance
                )
            )
            self._save_center.bind_analysis_path_changed(analysis_image_cb)
        if self._has_post:
            self.post_analyze_form.params_changed.connect(
                lambda instance: self._ctrl.update_tab_post_analyze_param_instance(
                    tab_id, instance
                )
            )
            self._save_center.bind_post_path_changed(post_image_cb)

        self.reset_btn.clicked.connect(self._on_reset_cfg_clicked)
        self.run_btn.clicked.connect(lambda: actions.run_or_stop(tab_id))
        if self._capabilities.load_data:
            self._save_center.bind_load(lambda: actions.load_data(tab_id))
        self._save_center.bind_save_all(lambda: actions.save_all(tab_id))
        if self._has_analysis:
            self.analyze_btn.clicked.connect(lambda: actions.analyze(tab_id))
            self.writeback_widget.apply_requested.connect(
                lambda: actions.apply_writeback(tab_id)
            )
            self._save_center.bind_save(
                ArtifactKind.ANALYSIS, lambda: actions.save_image(tab_id)
            )
        if self._has_post:
            self.post_analyze_btn.clicked.connect(lambda: actions.post_analyze(tab_id))
            self.post_writeback_widget.apply_requested.connect(
                lambda: actions.apply_post_writeback(tab_id)
            )
            self._save_center.bind_save(
                ArtifactKind.POST_ANALYSIS, lambda: actions.save_post_image(tab_id)
            )
        self._save_center.bind_save(
            ArtifactKind.DATA, lambda: actions.save_data(tab_id)
        )

        self._validity_cb = validity_cb
        self._schema_cb = schema_cb
        self._data_path_cb = data_path_cb
        if self._has_analysis:
            self._analysis_image_cb = analysis_image_cb
        if self._has_post:
            self._post_image_cb = post_image_cb

    def _on_progress_changed(self) -> None:
        models = tuple(m for _, m in self._progress_control.progress_bars(self.tab_id))
        self.progress_stack.render_models(models)

    def detach(self) -> None:
        """Tear this tab widget down."""
        if self._actions is None:
            raise RuntimeError(f"tab {self.tab_id!r} is not attached")
        self.cfg_form.validity_changed.disconnect(self._validity_cb)
        self.cfg_form.schema_changed.disconnect(self._schema_cb)
        self._save_center.unbind_data_path_changed(self._data_path_cb)
        if self._has_analysis:
            self._save_center.unbind_analysis_path_changed(self._analysis_image_cb)
        if self._has_post:
            self._save_center.unbind_post_path_changed(self._post_image_cb)
        self._progress_unsub()
        self.cfg_form.detach()
        if self._cfg_editor_id is not None:
            self._ctrl.teardown_cfg_editor(self._cfg_editor_id)
            self._cfg_editor_id = None
        self._actions = None
