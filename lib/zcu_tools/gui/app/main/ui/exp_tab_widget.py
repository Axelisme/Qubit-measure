"""Per-experiment tab widget for the measure-gui main window - capability driven."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

from zcu_tools.gui.app.main.adapter import AnalysisMode
from zcu_tools.gui.app.main.ui.cfg_binding import make_value_source_input_enhancer
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.plotting import FigureContainer, attach_existing_figure_to_container
from zcu_tools.gui.session.ui.progress_stack import ProgressStack
from zcu_tools.gui.widgets import DialogPresenter, QtDialogPresenter
from zcu_tools.gui.widgets.cfg import CfgFormWidget
from zcu_tools.gui.widgets.cfg.fields import _CollapsibleSection

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analyze_form import AnalyzeFormWidget
from .writeback_widget import WritebackWidget

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.main.adapter import WritebackItem
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services import TabSnapshot


class TabActions(Protocol):
    def refresh_interaction(self, tab_id: str) -> None: ...
    def run_or_stop(self, tab_id: str) -> None: ...
    def load_data(self, tab_id: str) -> None: ...
    def analyze(self, tab_id: str) -> None: ...
    def post_analyze(self, tab_id: str) -> None: ...
    def apply_writeback(self, tab_id: str) -> None: ...
    def save_data(self, tab_id: str) -> None: ...
    def save_image(self, tab_id: str) -> None: ...
    def save_result(self, tab_id: str) -> None: ...
    def save_post_image(self, tab_id: str) -> None: ...


class _PanelEdgeHandle(QToolButton):
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


class ExpTabWidget(QWidget):
    """A single experiment tab: Run | Analysis? | Post? | Save | Guide with independent figure panes."""

    def __init__(
        self,
        tab_id: str,
        ctrl: Controller,
        parent: QWidget | None = None,
        *,
        dialog_presenter: DialogPresenter | None = None,
    ) -> None:
        super().__init__(parent)
        self.tab_id = tab_id
        self._ctrl = ctrl
        self._dialog_presenter = dialog_presenter or QtDialogPresenter()
        self._progress_control = ctrl.progress_control
        self._cfg_editor_id: str | None = None
        self._actions: TabActions | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(2)
        self._content_widget = QWidget()
        content_row = QHBoxLayout(self._content_widget)
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)
        root_layout.addWidget(self._content_widget, stretch=1)
        self.progress_stack = ProgressStack()
        root_layout.addWidget(self.progress_stack, stretch=0)
        self._progress_unsub = self._progress_control.attach_progress(
            self.tab_id, self._on_progress_changed
        )
        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        content_row.addWidget(splitter, stretch=1)
        self._splitter = splitter
        self._splitter_left_saved = ctrl.get_persisted_startup().left_panel_width
        self._left_panel_collapsed = False
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        self._left_tabs = QTabWidget()
        self._left_edge_handle = _PanelEdgeHandle(self._content_widget)
        self._left_edge_handle.clicked.connect(self._toggle_left_panel)

        # -- Run panel --
        run_panel = QWidget()
        run_layout = QVBoxLayout(run_panel)
        run_layout.setContentsMargins(4, 4, 4, 4)
        run_layout.setSpacing(2)
        cfg_top_strip = QHBoxLayout()
        cfg_top_strip.setContentsMargins(0, 0, 0, 0)
        cfg_top_strip.addStretch()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFlat(True)
        self.reset_btn.setToolTip("Discard current config and restore adapter defaults")
        reset_font = self.reset_btn.font()
        reset_font.setPointSize(max(reset_font.pointSize() - 1, 7))
        self.reset_btn.setFont(reset_font)
        cfg_top_strip.addWidget(self.reset_btn)
        run_layout.addLayout(cfg_top_strip)
        self.cfg_form = CfgFormWidget(
            text_input_enhancer=make_value_source_input_enhancer(ctrl)
        )
        run_layout.addWidget(self.cfg_form, stretch=1)
        self.run_btn = QPushButton("Run")
        self.run_btn.setFixedHeight(30)
        run_layout.addWidget(self.run_btn)
        self._run_panel = run_panel
        self._run_tab_index = self._left_tabs.addTab(run_panel, "Run")

        # -- Analysis panel --
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_inner = QWidget()
        analysis_layout = QVBoxLayout(analysis_inner)
        analysis_layout.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        self._analyze_section = _CollapsibleSection(
            "Analysis", collapsible=True, collapsed=False
        )
        self.analyze_form = AnalyzeFormWidget()
        self._analyze_section.body_layout.addWidget(self.analyze_form)
        analysis_layout.addWidget(self._analyze_section)
        self.analyze_btn = QPushButton("Analyze")
        analysis_layout.addWidget(self.analyze_btn)
        self.writeback_section = _CollapsibleSection(
            "Writeback", collapsible=True, collapsed=False
        )
        self.writeback_widget = WritebackWidget(self._ctrl, tab_id=self.tab_id)
        self.writeback_section.body_layout.addWidget(self.writeback_widget)
        self.writeback_section.setVisible(False)
        analysis_layout.addWidget(self.writeback_section)
        # Image path for analysis
        analysis_save_section = _CollapsibleSection(
            "Save", collapsible=True, collapsed=False
        )
        analysis_save_layout = analysis_save_section.form
        image_path_row = QHBoxLayout()
        self._image_path_edit = QLineEdit()
        self._image_path_edit.setPlaceholderText("/tmp/image.png")
        image_path_row.addWidget(self._image_path_edit)
        browse_image_btn = QPushButton("Browse…")
        browse_image_btn.clicked.connect(self._on_browse_image_path)
        image_path_row.addWidget(browse_image_btn)
        analysis_save_layout.addRow("Image path:", image_path_row)
        self.save_image_btn = QPushButton("Save Image")
        analysis_save_layout.addRow("", self.save_image_btn)
        analysis_layout.addWidget(analysis_save_section)
        analysis_layout.addStretch()
        analysis_scroll.setWidget(analysis_inner)
        self._analysis_panel = analysis_scroll
        self._analysis_tab_index = self._left_tabs.addTab(analysis_scroll, "Analysis")

        # -- Post-Analysis panel --
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
        self.post_writeback_widget = WritebackWidget(self._ctrl, tab_id=self.tab_id)
        self.post_writeback_section.body_layout.addWidget(self.post_writeback_widget)
        self.post_writeback_section.setVisible(False)
        post_layout.addWidget(self.post_writeback_section)
        # Post image save
        post_save_section = _CollapsibleSection(
            "Save", collapsible=True, collapsed=False
        )
        post_save_layout = post_save_section.form
        post_image_path_row = QHBoxLayout()
        self._post_image_path_edit = QLineEdit()
        self._post_image_path_edit.setPlaceholderText("/tmp/post_image.png")
        post_image_path_row.addWidget(self._post_image_path_edit)
        browse_post_image_btn = QPushButton("Browse…")
        browse_post_image_btn.clicked.connect(self._on_browse_post_image_path)
        post_image_path_row.addWidget(browse_post_image_btn)
        post_save_layout.addRow("Image path:", post_image_path_row)
        self.post_save_image_btn = QPushButton("Save Image")
        post_save_layout.addRow("", self.post_save_image_btn)
        post_layout.addWidget(post_save_section)
        post_layout.addStretch()
        post_scroll.setWidget(post_inner)
        self._post_panel = post_scroll
        self._post_tab_index = self._left_tabs.addTab(post_scroll, "Post-Analysis")

        # -- Save panel --
        save_scroll = QScrollArea()
        save_scroll.setWidgetResizable(True)
        save_inner = QWidget()
        save_layout = QVBoxLayout(save_inner)
        save_layout.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        self.load_data_btn = QPushButton("Load Data...")
        save_layout.addWidget(self.load_data_btn)
        save_section = _CollapsibleSection("Save", collapsible=True, collapsed=False)
        save_form = save_section.form
        data_path_row = QHBoxLayout()
        self._data_path_edit = QLineEdit()
        self._data_path_edit.setPlaceholderText("/tmp/data")
        data_path_row.addWidget(self._data_path_edit)
        browse_data_btn = QPushButton("Browse…")
        browse_data_btn.clicked.connect(self._on_browse_data_path)
        data_path_row.addWidget(browse_data_btn)
        save_form.addRow("Data path:", data_path_row)
        self._comment_edit = QTextEdit()
        self._comment_edit.setPlaceholderText("Optional comment…")
        self._comment_edit.setFixedHeight(60)
        save_form.addRow("Comment:", self._comment_edit)
        btn_row = QHBoxLayout()
        self.save_data_btn = QPushButton("Save Data")
        # Keep Save Result for transitional compatibility but hidden per S4 (data only Save)
        self.save_result_btn = QPushButton("Save Result")
        self.save_result_btn.setVisible(False)
        btn_row.addWidget(self.save_data_btn)
        btn_row.addWidget(self.save_result_btn)
        save_form.addRow("", btn_row)
        save_layout.addWidget(save_section)
        save_layout.addStretch()
        save_scroll.setWidget(save_inner)
        self._save_panel = save_scroll
        self._save_tab_index = self._left_tabs.addTab(save_scroll, "Save")

        # -- Guide panel --
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
        self._guide_tab_index = self._left_tabs.addTab(guide_scroll, "Guide")

        splitter.addWidget(self._left_tabs)

        # -- Right pane: per-pane figure containers --
        plot_panel = QWidget()
        self._plot_layout = QVBoxLayout(plot_panel)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._right_stack = QStackedWidget()

        # Run figure pane
        self._run_stack = QStackedWidget()
        self._run_placeholder = QLabel("(no plot yet)")
        self._run_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._run_stack.addWidget(self._run_placeholder)
        self._run_container = FigureContainer(self._run_stack, self._run_placeholder)

        # Analysis figure pane
        self._analysis_stack = QStackedWidget()
        self._analysis_placeholder = QLabel("(no plot yet)")
        self._analysis_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._analysis_stack.addWidget(self._analysis_placeholder)
        self._analysis_container = FigureContainer(
            self._analysis_stack, self._analysis_placeholder
        )

        # Post figure pane
        self._post_stack = QStackedWidget()
        self._post_placeholder = QLabel("(no plot yet)")
        self._post_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._post_stack.addWidget(self._post_placeholder)
        self._post_container = FigureContainer(self._post_stack, self._post_placeholder)

        # Placeholder for Save/Guide (no figure)
        self._right_placeholder = QLabel("(no plot yet)")
        self._right_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]

        self._right_stack.addWidget(self._run_stack)
        self._right_stack.addWidget(self._analysis_stack)
        self._right_stack.addWidget(self._post_stack)
        self._right_stack.addWidget(self._right_placeholder)

        # Legacy aliases for tests that access _plot_stack / _figure_container / _plot_placeholder
        # Legacy shared stack is now the analysis stack (most common figure); keep for compatibility
        self._plot_stack = self._analysis_stack
        self._plot_placeholder = self._analysis_placeholder
        self._figure_container = self._analysis_container

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
    # Docked feedback panel host (ADR-0025 C3)
    # ------------------------------------------------------------------

    def mount_feedback_panel(self, panel: QWidget) -> None:
        """Dock the feedback panel directly below the figure (idempotent).

        Inserts ``panel`` into the plot column at index 1 — right under the plot
        stack (index 0). Re-mounting the same panel is a no-op; mounting a
        different panel first unmounts the current one.
        """
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
        """Render this adapter's static AdapterGuide as read-only rich text.

        Pulled once at construction from the adapter (no tab/context needed).
        Empty sections are dropped; a guide with no content at all falls back to
        an honest 'not written yet' line.
        """
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
                # In-memory only — persisted to disk at close (the caretaker
                # captures the active tab's width via current_left_panel_width).
                self._splitter_left_saved = sizes[0]
        self._schedule_handle_layout()

    def _schedule_handle_layout(self) -> None:
        QTimer.singleShot(0, self._layout_collapsed_handle)

    # ── attach / detach (whole-tab, snapshot-driven) ──────────────────────

    def attach(self, snapshot: TabSnapshot, actions: TabActions) -> None:
        """Bring this tab widget to life from one snapshot (mirrors
        ``CfgFormWidget.attach`` at the whole-tab scale): seed every sub-view
        from the snapshot's live fields, then wire the controller signals.
        Paired with :meth:`detach`. The snapshot is always a render snapshot
        (live fields populated)."""
        self._populate_cfg(snapshot.cfg_schema, self._ctrl)
        if snapshot.analyze_params is not None and self.has_analyze_params():
            self.analyze_form.populate_values(snapshot.analyze_params)
        self.sync_post_analyze_params(snapshot.post_analyze_params)
        # Per-pane path resources (Ticket 02 pane snapshots). Prefer pane paths if present.
        if getattr(snapshot, 'paths', None) is not None:  # type: ignore[attr-defined]
            data_path = snapshot.paths.data.path or ""  # type: ignore[attr-defined]
            analysis_path = snapshot.paths.analysis_image.path or ""  # type: ignore[attr-defined]
            post_path = snapshot.paths.post_analysis_image.path or ""  # type: ignore[attr-defined]
            # Use per-pane setters without triggering intermediate combined updates
            if data_path:
                self._data_path_edit.blockSignals(True)
                self._data_path_edit.setText(data_path)
                self._data_path_edit.blockSignals(False)
            if analysis_path:
                self._image_path_edit.blockSignals(True)
                self._image_path_edit.setText(analysis_path)
                self._image_path_edit.blockSignals(False)
            if post_path:
                self._post_image_path_edit.blockSignals(True)
                self._post_image_path_edit.setText(post_path)
                self._post_image_path_edit.blockSignals(False)
            # Seed post from analysis if empty (legacy behavior for initial suggestion)
            if not self._post_image_path_edit.text() and analysis_path:
                self._post_image_path_edit.setText(analysis_path)
        elif snapshot.save_paths is not None:
            self.set_save_paths(
                snapshot.save_paths.data_path, snapshot.save_paths.image_path
            )
        self.update_interaction_state(snapshot)
        self._bind_to_controller(actions)

    def _populate_cfg(self, schema: CfgSchema, ctrl: Controller) -> None:
        # The cfg LiveModel is owned by the CfgEditorService (ADR-0008): open a
        # gc=False session seeded from the committed schema, then attach the
        # widget to the service-owned model. tab_id is the owner key so the
        # editor_id is discoverable (tab.snapshot) and the agent can drive it.
        editor_id, _ = ctrl.open_seeded_cfg_editor(
            schema, gc=False, owner_key=self.tab_id
        )
        self._cfg_editor_id = editor_id
        self.cfg_form.attach(ctrl.get_cfg_editor_draft(editor_id))

    # ── populate / refresh helpers ────────────────────────────────────────

    def populate_analyze_params(self, instance: object) -> None:
        self.analyze_form.populate(instance)

    def read_analyze_params(self) -> object:
        return self.analyze_form.read_params()

    def has_analyze_params(self) -> bool:
        return self.analyze_form.has_params()

    def populate_post_analyze_params(self, instance: object) -> None:
        self.sync_post_analyze_params(instance)

    def sync_post_analyze_params(self, instance: object | None) -> None:
        self.post_analyze_form.sync(instance)
        has_fields = instance is not None and bool(
            dataclasses.fields(cast(Any, instance))
        )
        self._post_analyze_section.setVisible(has_fields)

    def read_post_analyze_params(self) -> object:
        return self.post_analyze_form.read_params()

    def has_post_analyze_params(self) -> bool:
        return self.post_analyze_form.has_params()

    def update_writeback_items(self, items: list[WritebackItem]) -> None:
        self.writeback_widget.populate(items)
        self.writeback_section.setVisible(len(items) > 0)

    def update_post_writeback_items(self, items: list[WritebackItem]) -> None:
        self.post_writeback_widget.populate(items)
        self.post_writeback_section.setVisible(len(items) > 0)

    def _on_browse_data_path(self) -> None:
        # The GUI save path helper reserves .hdf5 destinations, so show that
        # here — a .h5 filter would mislead.
        path, _ = QFileDialog.getSaveFileName(
            self, "Save data file", "", "HDF5 files (*.hdf5);;All files (*)"
        )
        if path:
            self._data_path_edit.setText(path)

    def _on_browse_image_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image file", "", "PNG files (*.png);;All files (*)"
        )
        if path:
            self._image_path_edit.setText(path)

    def _on_browse_post_image_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save post-analysis image file",
            "",
            "PNG files (*.png);;All files (*)",
        )
        if path:
            self._post_image_path_edit.setText(path)

    def set_save_paths(self, data_path: str, image_path: str) -> None:
        if data_path:
            self._data_path_edit.blockSignals(True)
            self._data_path_edit.setText(data_path)
            self._data_path_edit.blockSignals(False)
        if image_path:
            self._image_path_edit.blockSignals(True)
            self._image_path_edit.setText(image_path)
            self._image_path_edit.blockSignals(False)
            # Seed the post image path from the same suggestion when the user has
            # not typed their own — the post layer saves to its own field, which
            # follows the tab's image path until overridden.
            if not self._post_image_path_edit.text():
                self._post_image_path_edit.setText(image_path)

    def set_data_path(self, data_path: str) -> None:
        if data_path:
            self._data_path_edit.blockSignals(True)
            self._data_path_edit.setText(data_path)
            self._data_path_edit.blockSignals(False)

    def set_analysis_image_path(self, image_path: str) -> None:
        if image_path:
            self._image_path_edit.blockSignals(True)
            self._image_path_edit.setText(image_path)
            self._image_path_edit.blockSignals(False)
            if not self._post_image_path_edit.text():
                self._post_image_path_edit.setText(image_path)

    def set_post_image_path(self, image_path: str) -> None:
        if image_path:
            self._post_image_path_edit.blockSignals(True)
            self._post_image_path_edit.setText(image_path)
            self._post_image_path_edit.blockSignals(False)

    def get_data_path(self) -> str:
        return self._data_path_edit.text()

    def get_image_path(self) -> str:
        return self._image_path_edit.text()

    def get_post_image_path(self) -> str:
        return self._post_image_path_edit.text()

    def get_comment(self) -> str:
        return self._comment_edit.toPlainText()

    def focus_result_panel(self) -> None:
        """Show the Analysis/Save panel for this tab."""
        # Post ticket 03, focus the Analysis pane if available, else Save.
        try:
            # Try analysis tab if visible
            if self._left_tabs.isTabVisible(self._analysis_tab_index):
                self._left_tabs.setCurrentWidget(self._analysis_panel)
                return
        except Exception:
            pass
        # Fallback to Save if analysis not available
        self._left_tabs.setCurrentWidget(self._save_panel)

    # ── Figure container helpers (stable per-subtab identity S2) ──────────

    def get_run_container(self) -> FigureContainer:
        return self._run_container

    def get_analysis_container(self) -> FigureContainer:
        return self._analysis_container

    def get_post_container(self) -> FigureContainer:
        return self._post_container

    def prepare_run_container(self) -> FigureContainer:
        """Clear Run presentation and return its container (S3: only affected pane)."""
        self._run_container.clear_dynamic_canvases()
        return self._run_container

    def prepare_analysis_container(self) -> FigureContainer:
        """Clear Analysis presentation and return its container."""
        self._analysis_container.clear_dynamic_canvases()
        # Also ensure the right stack will show this pane when figure arrives;
        # do not automatically switch left selection.
        return self._analysis_container

    def prepare_post_container(self) -> FigureContainer:
        """Clear Post presentation and return its container."""
        self._post_container.clear_dynamic_canvases()
        return self._post_container

    def prepare_live_container(self) -> FigureContainer:
        """Legacy entry for Run live plot (kept for transitional callers).

        Clears the Run pane's container and returns it. Transitional callers that
        still use the shared alias will clear analysis via legacy reset, but new
        pane-aware callers use prepare_run_container directly.
        """
        return self.prepare_run_container()

    def mount_interactive_widget(self, widget: QWidget) -> None:
        """Mount an interactive analysis widget as the visible plot content (analysis pane)."""
        self._analysis_stack.addWidget(widget)
        self._analysis_stack.setCurrentWidget(widget)
        # Ensure right shows analysis when interactive picker is active
        self._right_stack.setCurrentWidget(self._analysis_stack)

    def unmount_interactive_widgets(self, widget_type: type[QWidget]) -> None:
        """Remove interactive widgets of ``widget_type`` and show placeholder in analysis pane."""
        for index in reversed(range(self._analysis_stack.count())):
            widget = self._analysis_stack.widget(index)
            if isinstance(widget, widget_type):
                self._analysis_stack.removeWidget(widget)
                widget.deleteLater()
        self._analysis_stack.setCurrentWidget(self._analysis_placeholder)
        # Also handle legacy _plot_stack alias (same object)
        # No need to touch run/post stacks.

    def left_panel_width(self) -> int:
        """Return the latest expanded left-panel width for persistence."""
        return self._splitter_left_saved

    def current_figure(self) -> Figure | None:
        """Return the visible matplotlib figure, or ``None`` at placeholder.

        Searches the currently visible right pane's stack for a figure, falling back
        to analysis pane for legacy callers that expect a figure after show_analysis_figure.
        """
        from matplotlib.figure import Figure

        # Prefer the currently visible right stack page
        current_right = self._right_stack.currentWidget()
        stacks_to_check: list[QStackedWidget] = []
        if isinstance(current_right, QStackedWidget):
            stacks_to_check.append(current_right)
        # Always also check analysis stack for legacy current_figure expectations
        if self._analysis_stack not in stacks_to_check:
            stacks_to_check.append(self._analysis_stack)
        # Check run stack as well
        if self._run_stack not in stacks_to_check:
            stacks_to_check.append(self._run_stack)
        # Check post stack
        if self._post_stack not in stacks_to_check:
            stacks_to_check.append(self._post_stack)

        for stack in stacks_to_check:
            canvas = stack.currentWidget()
            if canvas is None:
                continue
            # Skip placeholder labels
            if canvas in (
                self._run_placeholder,
                self._analysis_placeholder,
                self._post_placeholder,
                self._right_placeholder,
            ):
                continue
            figure = getattr(canvas, "figure", None)
            if not isinstance(figure, Figure):
                raise RuntimeError(
                    f"tab {self.tab_id!r} canvas has no matplotlib figure"
                )
            return figure
        return None

    def get_current_figure_for_pane(self, pane: str) -> Figure | None:
        """Pane-specific figure read (run|analysis|post_analysis)."""
        from matplotlib.figure import Figure

        mapping = {
            "run": (self._run_stack, self._run_placeholder),
            "analysis": (self._analysis_stack, self._analysis_placeholder),
            "post_analysis": (self._post_stack, self._post_placeholder),
        }
        if pane not in mapping:
            raise ValueError(f"unknown pane {pane!r}")
        stack, placeholder = mapping[pane]
        canvas = stack.currentWidget()
        if canvas is None or canvas is placeholder:
            return None
        figure = getattr(canvas, "figure", None)
        if not isinstance(figure, Figure):
            raise RuntimeError(f"tab {self.tab_id!r} canvas has no matplotlib figure")
        return figure

    def reset_plot(self) -> None:
        """Legacy reset: clear the analysis pane (most common) and also run/post for test compatibility.

        Real invalidation now clears all figure panes before a new run (Ticket 02 matrix),
        but per-pane clearing is available via prepare_* methods for operation start.
        """
        # Clear all figure panes to satisfy legacy tests that expect single call clears stale figure regardless of pane
        self._run_container.clear_dynamic_canvases()
        self._analysis_container.clear_dynamic_canvases()
        self._post_container.clear_dynamic_canvases()
        # Legacy aliases already point to analysis, so extra clear is harmless

    def clear_all_figures(self) -> None:
        """Clear all figure panes (used for LoadData and Run start invalidation)."""
        self._run_container.clear_dynamic_canvases()
        self._analysis_container.clear_dynamic_canvases()
        self._post_container.clear_dynamic_canvases()

    def show_run_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the Run pane."""
        canvas = attach_existing_figure_to_container(fig, self._run_container)
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached run canvas does not support draw()")
        draw()
        logger.debug("show_run_figure: tab_id=%r canvas set", self.tab_id)

    def show_analysis_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the Analysis pane and bring it to front.

        The run figure lives in its own container; analysis and post each have
        distinct containers. Attaching only switches that pane's stack.
        """
        canvas = attach_existing_figure_to_container(fig, self._analysis_container)
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached analysis canvas does not support draw()")
        draw()
        logger.debug("show_analysis_figure: tab_id=%r canvas set", self.tab_id)

    def show_post_analysis_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the Post-Analysis pane."""
        canvas = attach_existing_figure_to_container(fig, self._post_container)
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached post canvas does not support draw()")
        draw()
        logger.debug("show_post_analysis_figure: tab_id=%r canvas set", self.tab_id)

    def show_post_figure(self, fig: Figure) -> None:
        """Alias for show_post_analysis_figure (legacy)."""
        self.show_post_analysis_figure(fig)

    def _on_reset_cfg_clicked(self) -> None:
        # Guard: ask before discarding — Reset is destructive (drops entire cfg).
        confirmed = self._dialog_presenter.confirm(
            self,
            "Reset config",
            "Reset config to defaults? This discards the current configuration.",
            default=False,
        )
        if not confirmed:
            return
        # Controller regenerates + commits the adapter-default cfg (and gates a
        # running tab); we just re-seed the form over the new committed schema.
        assert self._actions is not None, "reset clicked before bind"
        schema = self._ctrl.reset_tab_cfg(self.tab_id)
        self._reseed_cfg(schema)
        self._actions.refresh_interaction(self.tab_id)

    def _reseed_cfg(self, schema: CfgSchema) -> None:
        """Swap the cfg form onto a fresh service-owned session for ``schema``.

        The cfg_form widget itself is unchanged — only the LiveModel it views is
        replaced — so the widget→controller bindings (``schema_changed`` →
        ``_schema_cb``, ``validity_changed`` → ``_validity_cb`` set in
        ``_bind_to_controller``) stay connected exactly once and must NOT be
        re-connected here (that would double-fire ``update_tab_cfg``). Only the
        model↔widget binding is rebuilt: ``detach`` drops the old one, ``attach``
        wires the new model. ``attach`` re-emits only ``validity_changed`` (not
        ``schema_changed``), so re-seeding does not write the default cfg back —
        ``reset_tab_cfg`` already committed it.
        """
        self.cfg_form.detach()
        if self._cfg_editor_id is not None:
            self._ctrl.teardown_cfg_editor(self._cfg_editor_id)
            self._cfg_editor_id = None
        editor_id, _ = self._ctrl.open_seeded_cfg_editor(
            schema, gc=False, owner_key=self.tab_id
        )
        self._cfg_editor_id = editor_id
        self.cfg_form.attach(self._ctrl.get_cfg_editor_draft(editor_id))

    def _on_left_tab_changed(self, index: int) -> None:
        """Switch right pane to match left subtab (Save/Guide → placeholder)."""
        widget = self._left_tabs.widget(index)
        if widget is self._run_panel:
            self._right_stack.setCurrentWidget(self._run_stack)
        elif widget is self._analysis_panel:
            self._right_stack.setCurrentWidget(self._analysis_stack)
        elif widget is self._post_panel:
            self._right_stack.setCurrentWidget(self._post_stack)
        else:
            # Save or Guide
            self._right_stack.setCurrentWidget(self._right_placeholder)

    def update_interaction_state(self, snapshot: TabSnapshot) -> None:
        # A render snapshot (get_tab_snapshot) always fills the live fields; only
        # the persist/restore form leaves them None, and that never reaches here.
        assert snapshot.interaction is not None
        assert snapshot.capabilities is not None
        state = snapshot.interaction
        capabilities = snapshot.capabilities
        local_busy = state.is_running or state.is_analyzing or state.is_saving_data
        if state.is_running:
            self.run_btn.setText("Stop")
            self.run_btn.setEnabled(True)
            self.run_btn.setToolTip("Running")
            self.run_btn.setStyleSheet(
                "background-color: #f44336; color: white; font-weight: bold;"
            )
        else:
            self.run_btn.setText("Run")
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
            self.run_btn.setStyleSheet("")

        idle = not local_busy
        self.cfg_form.set_editing_enabled(idle)
        self.reset_btn.setEnabled(idle)

        # Capability-driven left tab visibility (S1)
        has_analysis = capabilities.analysis is not AnalysisMode.NONE
        has_post = capabilities.post_analysis
        # load_data capability controls Load Data button; default to False if missing
        has_load = bool(getattr(capabilities, "load_data", False))
        self._left_tabs.setTabVisible(self._analysis_tab_index, has_analysis)
        self._left_tabs.setTabVisible(self._post_tab_index, has_post)
        # If current tab is now hidden, switch to Run
        current = self._left_tabs.currentWidget()
        if current is not None and not self._left_tabs.isTabVisible(
            self._left_tabs.indexOf(current)
        ):
            self._left_tabs.setCurrentWidget(self._run_panel)

        # Analysis pane controls
        # When analysis capability missing, keep its inner widgets disabled/hidden but pane itself hidden
        self._analyze_section.setVisible(has_analysis)
        self.analyze_btn.setVisible(has_analysis)
        if not has_analysis:
            self.writeback_section.setVisible(False)
        self.analyze_btn.setEnabled(
            idle and has_analysis and state.has_context and state.has_run_result
        )
        self.analyze_form.setEnabled(idle and has_analysis)
        self.writeback_widget.setEnabled(
            idle and has_analysis and state.has_context and state.has_analyze_result
        )
        # Load Data button lives in Save pane, visibility driven by load_data
        self.load_data_btn.setVisible(has_load)
        self.load_data_btn.setEnabled(idle and has_load and state.has_context)
        # Save pane data controls
        self.save_data_btn.setEnabled(
            idle and state.has_active_context and state.has_run_result
        )
        # Save Result hidden per S4 (data only)
        # Analysis image save
        self.save_image_btn.setEnabled(
            idle and state.has_active_context and state.has_figure
        )
        # Post pane controls
        if has_post:
            post_enabled = idle and state.has_analyze_result
            self.post_analyze_form.setEnabled(post_enabled)
            self.post_analyze_btn.setEnabled(post_enabled)
            self._post_gate_label.setVisible(not state.has_analyze_result)
            self.post_save_image_btn.setEnabled(
                idle and state.has_active_context and state.has_post_analyze_result
            )
            self.post_writeback_widget.setEnabled(
                idle and state.has_context and state.has_post_analyze_result
            )
        else:
            # Ensure post gate hidden when pane hidden
            self._post_gate_label.setVisible(False)

        # Save pane controls are always visible (data path)
        # Guide has no controls

    def _bind_to_controller(self, actions: TabActions) -> None:
        tab_id = self.tab_id
        # Held so the Reset handler can refresh interaction state after re-seeding
        # (the only post-bind path that needs the actions off a button slot).
        self._actions = actions

        def validity_cb(_valid: bool) -> None:
            actions.refresh_interaction(tab_id)

        def schema_cb(schema_obj: CfgSchema) -> None:
            self._ctrl.update_tab_cfg(tab_id, schema_obj)

        def data_path_cb(_text: str) -> None:
            # Per-pane data path independent; update only data override
            data_path = self.get_data_path()
            # Use per-pane controller method if available, else fallback to combined
            update_data = getattr(self._ctrl, "update_tab_data_path", None)
            if callable(update_data):
                try:
                    update_data(tab_id, data_path if data_path else None)
                    return
                except Exception:
                    pass
            # Fallback to legacy combined (requires image path too)
            image_path = self.get_image_path()
            if bool(data_path) != bool(image_path):
                return
            self._ctrl.update_tab_save_paths(tab_id, data_path, image_path)

        def analysis_image_cb(_text: str) -> None:
            image_path = self.get_image_path()
            update_img = getattr(self._ctrl, "update_tab_analysis_image_path", None)
            if callable(update_img):
                try:
                    update_img(tab_id, image_path if image_path else None)
                    return
                except Exception:
                    pass
            # fallback
            data_path = self.get_data_path()
            if bool(data_path) != bool(image_path):
                return
            self._ctrl.update_tab_save_paths(tab_id, data_path, image_path)

        def post_image_cb(_text: str) -> None:
            image_path = self.get_post_image_path()
            update_post = getattr(
                self._ctrl, "update_tab_post_analysis_image_path", None
            )
            if callable(update_post):
                try:
                    update_post(tab_id, image_path if image_path else None)
                    return
                except Exception:
                    pass

        self.cfg_form.validity_changed.connect(validity_cb)
        self.cfg_form.schema_changed.connect(schema_cb)

        # The cfg editor session + widget attach were already set up in
        # populate_cfg (the service owns the model — ADR-0008). The agent reaches
        # it via the tab's editor_id (exposed on tab.snapshot).
        self.analyze_form.params_changed.connect(
            lambda instance: self._ctrl.update_tab_analyze_param_instance(
                tab_id, instance
            )
        )
        self.post_analyze_form.params_changed.connect(
            lambda instance: self._ctrl.update_tab_post_analyze_param_instance(
                tab_id, instance
            )
        )
        self._data_path_edit.textChanged.connect(data_path_cb)
        self._image_path_edit.textChanged.connect(analysis_image_cb)
        self._post_image_path_edit.textChanged.connect(post_image_cb)
        self.reset_btn.clicked.connect(self._on_reset_cfg_clicked)
        self.run_btn.clicked.connect(lambda: actions.run_or_stop(tab_id))
        self.load_data_btn.clicked.connect(lambda: actions.load_data(tab_id))
        self.analyze_btn.clicked.connect(lambda: actions.analyze(tab_id))
        self.post_analyze_btn.clicked.connect(lambda: actions.post_analyze(tab_id))
        self.writeback_widget.apply_requested.connect(
            lambda: actions.apply_writeback(tab_id)
        )
        # Post writeback apply currently uses same tab writeback apply (primary); for post we need separate?
        # For now, post writeback apply also goes through same handler but will apply post draft via controller's pane-aware writeback?
        # Keep as same for backward compat; future tickets will split.
        self.post_writeback_widget.apply_requested.connect(
            lambda: actions.apply_writeback(tab_id)
        )
        self.save_data_btn.clicked.connect(lambda: actions.save_data(tab_id))
        self.save_image_btn.clicked.connect(lambda: actions.save_image(tab_id))
        self.save_result_btn.clicked.connect(lambda: actions.save_result(tab_id))
        self.post_save_image_btn.clicked.connect(
            lambda: actions.save_post_image(tab_id)
        )

        self._validity_cb = validity_cb
        self._schema_cb = schema_cb
        self._data_path_cb = data_path_cb
        self._analysis_image_cb = analysis_image_cb
        self._post_image_cb = post_image_cb

    def _on_progress_changed(self) -> None:
        # Main-thread callback from ProgressService; re-render the live bars of
        # this tab's current run (empty when no run is live).
        models = tuple(m for _, m in self._progress_control.progress_bars(self.tab_id))
        self.progress_stack.render_models(models)

    def detach(self) -> None:
        """Tear this tab widget down (mirrors ``CfgFormWidget.detach`` at the
        whole-tab scale): drop the controller signal bindings, detach the cfg
        widget, and tell the service to tear down the model it owns (ADR-0008).
        Paired with :meth:`attach`."""
        if hasattr(self, "_validity_cb"):
            self.cfg_form.validity_changed.disconnect(self._validity_cb)
        if hasattr(self, "_schema_cb"):
            self.cfg_form.schema_changed.disconnect(self._schema_cb)
        # Disconnect path callbacks if they exist
        if hasattr(self, "_data_path_cb"):
            try:
                self._data_path_edit.textChanged.disconnect(self._data_path_cb)
            except Exception:
                pass
        if hasattr(self, "_analysis_image_cb"):
            try:
                self._image_path_edit.textChanged.disconnect(self._analysis_image_cb)
            except Exception:
                pass
        if hasattr(self, "_post_image_cb"):
            try:
                self._post_image_path_edit.textChanged.disconnect(self._post_image_cb)
            except Exception:
                pass
        self._progress_unsub()
        # Detach the widget first (drop its signal bindings + widget tree), then
        # tell the service to tear down the model it owns (ADR-0008).
        self.cfg_form.detach()
        if self._cfg_editor_id is not None:
            self._ctrl.teardown_cfg_editor(self._cfg_editor_id)
            self._cfg_editor_id = None
