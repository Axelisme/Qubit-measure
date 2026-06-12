"""MainWindow — the top-level View for the v2_gui framework.

Implements ViewProtocol; all state lives in Controller/State, never here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from zcu_tools.gui.app.main.adapter import AnalysisMode, CfgSchema
from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.plotting import (
    FigureContainer,
    attach_existing_figure_to_container,
    remove_canvas,
    set_shutting_down,
)
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    MlChangedPayload,
    PredictorChangedPayload,
    SocChangedPayload,
)

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtGui import (  # type: ignore[attr-defined]
    QCloseEvent,  # type: ignore[attr-defined]
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.session.ui.progress_stack import ProgressStack

from .analyze_form import AnalyzeFormWidget
from .cfg_form import (
    CfgFormWidget,
)
from .fields import (
    _CollapsibleSection,
)
from .writeback_widget import WritebackWidget

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QDialog  # type: ignore[attr-defined]

    from zcu_tools.gui.app.main.adapter import CfgSchema, WritebackItem
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services import TabSnapshot


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


class ExpTabWidget(QWidget):
    """A single experiment tab: Config | Plot | Result areas."""

    def __init__(
        self, tab_id: str, ctrl: Controller, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.tab_id = tab_id
        self._ctrl = ctrl
        self._writeback_count: int = 0
        # editor_id of this tab's shared cfg-editor session (set on bind, when
        # the cfg_form's live model exists). Exposed to agents via tab.snapshot.
        self._cfg_editor_id: str | None = None
        # The owning MainWindow, captured in _bind_to_controller (button slots
        # otherwise close over it; the Reset handler needs it off-slot).
        self._main_window: MainWindow | None = None

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

        # Subscribe once by our own tab_id (the run operation's owner); the
        # listener re-reads the live bars on every change and follows the tab
        # across successive runs. Disposed in teardown.
        self._progress_unsub = self._ctrl.attach_progress(
            self.tab_id, self._on_progress_changed
        )

        # splitter holds two panes: left (tab panel) | right (plot)
        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]

        content_row.addWidget(splitter, stretch=1)

        self._splitter = splitter
        self._splitter_left_saved = ctrl.get_persisted_startup().left_panel_width
        self._left_panel_collapsed = False
        self._splitter.splitterMoved.connect(self._on_splitter_moved)

        # ── Left pane: QTabWidget with Config tab and Analysis tab ───────
        self._left_tabs = QTabWidget()

        self._left_edge_handle = _PanelEdgeHandle(self._content_widget)
        self._left_edge_handle.clicked.connect(self._toggle_left_panel)

        # ── Tab 0: Config ────────────────────────────────────────────────
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(4, 4, 4, 4)
        config_layout.setSpacing(2)

        # Thin top strip: Reset sits right-aligned at the top of the cfg area,
        # visually de-emphasised (flat, small font) to reduce accidental clicks.
        # A spacer pushes it to the right; the strip adds only minimal height.
        cfg_top_strip = QHBoxLayout()
        cfg_top_strip.setContentsMargins(0, 0, 0, 0)
        cfg_top_strip.addStretch()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFlat(True)
        self.reset_btn.setToolTip("Discard current config and restore adapter defaults")
        # Smaller font signals secondary action (Run is the primary action below).
        reset_font = self.reset_btn.font()
        reset_font.setPointSize(max(reset_font.pointSize() - 1, 7))
        self.reset_btn.setFont(reset_font)
        cfg_top_strip.addWidget(self.reset_btn)
        config_layout.addLayout(cfg_top_strip)

        self.cfg_form = CfgFormWidget()
        self.cfg_form.validity_changed.connect(self._on_cfg_validity_changed)
        config_layout.addWidget(self.cfg_form, stretch=1)

        self.run_btn = QPushButton("Run")
        self.run_btn.setFixedHeight(30)
        config_layout.addWidget(self.run_btn)
        self._left_tabs.addTab(config_panel, "Config")

        # ── Tab 1: Analysis ──────────────────────────────────────────────
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_inner = QWidget()
        analysis_layout = QVBoxLayout(analysis_inner)
        analysis_layout.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]

        # Analyze params group
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
        self.writeback_widget = WritebackWidget(self._ctrl)
        self.writeback_section.body_layout.addWidget(self.writeback_widget)
        self.writeback_section.setVisible(False)
        analysis_layout.addWidget(self.writeback_section)

        # Save group
        save_section = _CollapsibleSection("Save", collapsible=True, collapsed=False)
        save_layout = save_section.form

        data_path_row = QHBoxLayout()
        self._data_path_edit = QLineEdit()
        self._data_path_edit.setPlaceholderText("/tmp/data")
        data_path_row.addWidget(self._data_path_edit)
        browse_data_btn = QPushButton("Browse…")
        browse_data_btn.clicked.connect(self._on_browse_data_path)
        data_path_row.addWidget(browse_data_btn)
        save_layout.addRow("Data path:", data_path_row)

        image_path_row = QHBoxLayout()
        self._image_path_edit = QLineEdit()
        self._image_path_edit.setPlaceholderText("/tmp/image.png")
        image_path_row.addWidget(self._image_path_edit)
        browse_image_btn = QPushButton("Browse…")
        browse_image_btn.clicked.connect(self._on_browse_image_path)
        image_path_row.addWidget(browse_image_btn)
        save_layout.addRow("Image path:", image_path_row)

        self._comment_edit = QTextEdit()
        self._comment_edit.setPlaceholderText("Optional comment…")
        self._comment_edit.setFixedHeight(60)
        save_layout.addRow("Comment:", self._comment_edit)

        btn_row = QHBoxLayout()
        self.save_data_btn = QPushButton("Save Data")
        self.save_image_btn = QPushButton("Save Image")
        self.save_result_btn = QPushButton("Save Result")
        btn_row.addWidget(self.save_data_btn)
        btn_row.addWidget(self.save_image_btn)
        btn_row.addWidget(self.save_result_btn)
        save_layout.addRow("", btn_row)

        analysis_layout.addWidget(save_section)
        analysis_layout.addStretch()

        analysis_scroll.setWidget(analysis_inner)
        self._left_tabs.addTab(analysis_scroll, "Analysis")

        # ── Tab 2: Post-Analysis ─────────────────────────────────────────
        # A second analysis layer that runs on top of the primary analyze result
        # (e.g. single-shot multi-backend discrimination). Only adapters declaring
        # ``capabilities.post_analysis`` enable it; for the rest the whole sub-tab
        # is hidden. The post figure renders into the *shared* right-pane container
        # (the same one run/analyze use) — the container shows the most recently
        # produced figure, so the post layer never gets a private plot stack.
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

        # Save group for the post layer — mirrors the primary Save section but
        # image-only: the post figure (``tab.post_figure``) is the thing to save;
        # there is no separate post data file (it shares the run result's data).
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
        self._post_tab_index = self._left_tabs.addTab(post_scroll, "Post-Analysis")

        # ── Tab 3: Guide ─────────────────────────────────────────────────
        # Read-only orientation for this adapter (behavior / expects / writeback
        # / recommended). Static content — filled once at tab creation from the
        # adapter's AdapterGuide; no subscription/refresh needed.
        guide_scroll = QScrollArea()
        guide_scroll.setWidgetResizable(True)
        guide_label = QLabel()
        guide_label.setWordWrap(True)
        guide_label.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        guide_label.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        guide_label.setContentsMargins(8, 8, 8, 8)
        guide_label.setText(self._render_guide_html())
        guide_scroll.setWidget(guide_label)
        self._left_tabs.addTab(guide_scroll, "Guide")

        splitter.addWidget(self._left_tabs)

        # ── Right pane: Plot ─────────────────────────────────────────────
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._plot_stack = QStackedWidget()

        self._plot_placeholder = QLabel("(no plot yet)")
        self._plot_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._plot_stack.addWidget(self._plot_placeholder)
        self._figure_container = FigureContainer(
            self._plot_stack, self._plot_placeholder
        )

        self._canvas_widget: QWidget | None = None

        plot_layout.addWidget(self._plot_stack, stretch=1)
        splitter.addWidget(plot_panel)

        splitter.setCollapsible(0, True)
        self._update_left_panel_controls()
        self._schedule_handle_layout()

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

    def attach(self, snapshot: TabSnapshot, main_window: MainWindow) -> None:
        """Bring this tab widget to life from one snapshot (mirrors
        ``CfgFormWidget.attach`` at the whole-tab scale): seed every sub-view
        from the snapshot's live fields, then wire the controller signals.
        Paired with :meth:`detach`. The snapshot is always a render snapshot
        (live fields populated)."""
        self._populate_cfg(snapshot.cfg_schema, self._ctrl)
        if snapshot.analyze_params is not None and self.has_analyze_params():
            self.analyze_form.populate_values(snapshot.analyze_params)
        if snapshot.post_analyze_params is not None and self.has_post_analyze_params():
            self.post_analyze_form.populate_values(snapshot.post_analyze_params)
        if snapshot.save_paths is not None:
            self.set_save_paths(
                snapshot.save_paths.data_path, snapshot.save_paths.image_path
            )
        self.update_interaction_state(snapshot)
        self._bind_to_controller(main_window)

    def _populate_cfg(self, schema: CfgSchema, ctrl: Controller) -> None:
        # The cfg LiveModel is owned by the CfgEditorService (ADR-0008): open a
        # gc=False session seeded from the committed schema, then attach the
        # widget to the service-owned model. tab_id is the owner key so the
        # editor_id is discoverable (tab.snapshot) and the agent can drive it.
        editor_id, _ = ctrl.open_seeded_cfg_editor(
            schema, gc=False, owner_key=self.tab_id
        )
        self._cfg_editor_id = editor_id
        self.cfg_form.attach(ctrl.get_cfg_editor_root(editor_id))

    def read_schema(self) -> CfgSchema:
        return self.cfg_form.read_schema()

    # ── populate / refresh helpers ────────────────────────────────────────

    def populate_analyze_params(self, instance: object) -> None:
        self.analyze_form.populate(instance)

    def read_analyze_params(self) -> object:
        return self.analyze_form.read_params()

    def has_analyze_params(self) -> bool:
        return self.analyze_form.has_params()

    def populate_post_analyze_params(self, instance: object) -> None:
        self.post_analyze_form.populate(instance)

    def read_post_analyze_params(self) -> object:
        return self.post_analyze_form.read_params()

    def has_post_analyze_params(self) -> bool:
        return self.post_analyze_form.has_params()

    def update_writeback_items(self, items: list[WritebackItem]) -> None:
        self._writeback_count = sum(1 for item in items if item.selected)
        self.writeback_widget.populate(items)
        self.writeback_section.setVisible(len(items) > 0)

    def _on_browse_data_path(self) -> None:
        # The data saver forces the .hdf5 extension (save_local_data /
        # safe_labber_filepath), so show that here — a .h5 filter would mislead:
        # the file actually lands as .hdf5.
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

    def get_data_path(self) -> str:
        return self._data_path_edit.text()

    def get_image_path(self) -> str:
        return self._image_path_edit.text()

    def get_post_image_path(self) -> str:
        return self._post_image_path_edit.text()

    def get_comment(self) -> str:
        return self._comment_edit.toPlainText()

    def reset_plot(self) -> None:
        """Remove all canvases from plot_stack, revert to placeholder."""
        self._figure_container.clear_dynamic_canvases()
        self._canvas_widget = None

    def show_analysis_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the plot area (replaces prior analysis canvas)."""
        canvas = attach_existing_figure_to_container(fig, self._figure_container)
        if self._canvas_widget is not None and self._canvas_widget is not canvas:
            remove_canvas(self._canvas_widget)
        self._canvas_widget = canvas
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached analysis canvas does not support draw()")
        draw()
        logger.debug("show_analysis_figure: tab_id=%r canvas set", self.tab_id)

    def _on_cfg_validity_changed(self, valid: bool) -> None:
        del valid

    def _on_reset_cfg_clicked(self) -> None:
        # Guard: ask before discarding — Reset is destructive (drops entire cfg).
        reply = QMessageBox.question(
            self,
            "Reset config",
            "Reset config to defaults? This discards the current configuration.",
            QMessageBox.Yes | QMessageBox.No,  # type: ignore[attr-defined]
            QMessageBox.No,  # type: ignore[attr-defined]
        )
        if reply != QMessageBox.Yes:  # type: ignore[attr-defined]
            return
        # Controller regenerates + commits the adapter-default cfg (and gates a
        # running tab); we just re-seed the form over the new committed schema.
        assert self._main_window is not None, "reset clicked before bind"
        schema = self._ctrl.reset_tab_cfg(self.tab_id)
        self._reseed_cfg(schema)
        self._main_window.refresh_tab_interaction(self.tab_id)

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
        self.cfg_form.attach(self._ctrl.get_cfg_editor_root(editor_id))

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
        self.cfg_form.setEnabled(idle)
        self.reset_btn.setEnabled(idle)

        # Non-analysis adapters (flux_dep / power_dep) hide only the analysis
        # widgets, NOT the whole tab — the Save section lives in this same tab and
        # must stay reachable so any run can be saved. (Writeback is already
        # gated by item count in update_writeback_items; hide it here too so it
        # never lingers from a previous analysis adapter on the same tab.)
        has_analysis = capabilities.analysis is not AnalysisMode.NONE
        self._analyze_section.setVisible(has_analysis)
        self.analyze_btn.setVisible(has_analysis)
        if not has_analysis:
            self.writeback_section.setVisible(False)
        # The second tab carries analysis widgets + Save; when analysis is hidden
        # only Save remains, so label it accordingly instead of "Analysis".
        self._left_tabs.setTabText(1, "Analysis" if has_analysis else "Save")
        self.analyze_form.setEnabled(idle and has_analysis)
        self.analyze_btn.setEnabled(
            idle and has_analysis and state.has_context and state.has_run_result
        )

        self.save_data_btn.setEnabled(
            idle and state.has_active_context and state.has_run_result
        )
        self.save_image_btn.setEnabled(
            idle and state.has_active_context and state.has_figure
        )
        self.save_result_btn.setEnabled(
            idle and state.has_active_context and state.has_figure
        )
        self.writeback_widget.setEnabled(
            idle and state.has_context and state.has_analyze_result
        )

        # Post-analysis sub-tab: shown only for adapters that declare it. The form
        # + Run are gated on a *primary* analyze result existing (the post layer
        # builds on it); a hint label is shown while that gate is closed.
        has_post = capabilities.post_analysis
        self._left_tabs.setTabVisible(self._post_tab_index, has_post)
        if has_post:
            post_enabled = idle and state.has_analyze_result
            self.post_analyze_form.setEnabled(post_enabled)
            self.post_analyze_btn.setEnabled(post_enabled)
            self._post_gate_label.setVisible(not state.has_analyze_result)
            # Post Save Image gates on a post result existing (its figure is the
            # thing saved), mirroring the primary Save Image gate on has_figure.
            self.post_save_image_btn.setEnabled(
                idle and state.has_active_context and state.has_post_analyze_result
            )

    def _bind_to_controller(self, main_window: MainWindow) -> None:
        tab_id = self.tab_id
        # Held so the Reset handler can refresh interaction state after re-seeding
        # (the only post-bind path that needs the MainWindow off a button slot).
        self._main_window = main_window

        def validity_cb(_valid: bool) -> None:
            main_window.refresh_tab_interaction(tab_id)

        def schema_cb(schema_obj: CfgSchema) -> None:
            self._ctrl.update_tab_cfg(tab_id, schema_obj)

        def save_paths_cb(_text: str) -> None:
            data_path = self.get_data_path()
            image_path = self.get_image_path()
            if bool(data_path) != bool(image_path):
                return
            self._ctrl.update_tab_save_paths(tab_id, data_path, image_path)

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
        self._data_path_edit.textChanged.connect(save_paths_cb)
        self._image_path_edit.textChanged.connect(save_paths_cb)
        self.reset_btn.clicked.connect(self._on_reset_cfg_clicked)
        self.run_btn.clicked.connect(lambda: main_window._on_run_stop_clicked(tab_id))
        self.analyze_btn.clicked.connect(
            lambda: main_window._on_analyze_clicked(tab_id)
        )
        self.post_analyze_btn.clicked.connect(
            lambda: main_window._on_post_analyze_clicked(tab_id)
        )
        self.writeback_widget.apply_requested.connect(
            lambda: main_window._on_writeback_inline_apply(tab_id)
        )
        self.save_data_btn.clicked.connect(
            lambda: main_window._on_save_data_clicked(tab_id)
        )
        self.save_image_btn.clicked.connect(
            lambda: main_window._on_save_image_clicked(tab_id)
        )
        self.save_result_btn.clicked.connect(
            lambda: main_window._on_save_result_clicked(tab_id)
        )
        self.post_save_image_btn.clicked.connect(
            lambda: main_window._on_post_save_image_clicked(tab_id)
        )

        self._validity_cb = validity_cb
        self._schema_cb = schema_cb

    def _on_progress_changed(self) -> None:
        # Main-thread callback from ProgressService; re-render the live bars of
        # this tab's current run (empty when no run is live).
        models = tuple(m for _, m in self._ctrl.progress_bars(self.tab_id))
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
        self._progress_unsub()
        # Detach the widget first (drop its signal bindings + widget tree), then
        # tell the service to tear down the model it owns (ADR-0008).
        self.cfg_form.detach()
        if self._cfg_editor_id is not None:
            self._ctrl.teardown_cfg_editor(self._cfg_editor_id)
            self._cfg_editor_id = None


# ---------------------------------------------------------------------------
# MainWindow — implements ViewProtocol
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Top-level window; implements ViewProtocol for Controller callbacks."""

    def __init__(self, controller: Controller) -> None:
        super().__init__()
        self._ctrl = controller
        self._tab_widgets: dict[str, ExpTabWidget] = {}
        # ``DialogName -> live QDialog`` registry. ``InspectDialog`` is also
        # tracked through this dict (the legacy ``_inspect_dialog`` attribute
        # is gone — there is now exactly one entry point).
        self._open_dialogs: dict[DialogName, QDialog] = {}
        # True once _perform_close has begun the actual teardown, so the second
        # closeEvent (triggered by _perform_close's self.close()) passes straight
        # through instead of re-entering the cancel-and-wait coordination.
        self._closing = False

        self.setWindowTitle("ZCU Qubit Measure — v2 GUI")
        self.resize(1280, 750)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- toolbar ---
        toolbar = QHBoxLayout()
        self._new_tab_btn = QPushButton("New Tab ▾")
        self._new_tab_btn.clicked.connect(self._on_new_tab_requested)
        toolbar.addWidget(self._new_tab_btn)
        toolbar.addStretch()

        setup_btn = QPushButton("Setup…")
        setup_btn.clicked.connect(self._on_setup_clicked)
        toolbar.addWidget(setup_btn)

        devices_btn = QPushButton("Devices…")
        devices_btn.clicked.connect(self._on_devices_clicked)
        toolbar.addWidget(devices_btn)

        predictor_btn = QPushButton("Predictor…")
        predictor_btn.clicked.connect(self._on_predictor_clicked)
        toolbar.addWidget(predictor_btn)

        inspect_btn = QPushButton("Inspect…")
        inspect_btn.clicked.connect(self._on_inspect_clicked)
        toolbar.addWidget(inspect_btn)

        main_layout.addLayout(toolbar)

        # --- context / predictor status bar ---
        ctx_bar = QHBoxLayout()
        ctx_bar.addWidget(QLabel("Context:"))
        self._ctx_label = QLabel("(none)")
        ctx_bar.addWidget(self._ctx_label)
        ctx_bar.addSpacing(24)
        ctx_bar.addWidget(QLabel("Predictor:"))
        self._predictor_label = QLabel("none")
        ctx_bar.addWidget(self._predictor_label)
        ctx_bar.addStretch()
        main_layout.addLayout(ctx_bar)

        # --- tab widget ---
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tabs.currentChanged.connect(self._on_current_tab_changed)
        main_layout.addWidget(self._tabs, stretch=1)

        # --- status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

        # EventBus subscriptions
        bus = self._ctrl.get_bus()
        bus.subscribe(
            TabInteractionChangedPayload, self._on_bus_tab_interaction_changed
        )
        bus.subscribe(RunStartedPayload, self._on_bus_run_started)
        bus.subscribe(RunFinishedPayload, self._on_bus_run_finished)
        bus.subscribe(ContextSwitchedPayload, self._on_bus_context_switched)
        bus.subscribe(MlChangedPayload, self._on_bus_ml_changed)
        bus.subscribe(TabAddedPayload, self._on_bus_tab_added)
        bus.subscribe(TabClosedPayload, self._on_bus_tab_closed)
        bus.subscribe(TabContentChangedPayload, self._on_bus_tab_content_changed)
        bus.subscribe(PredictorChangedPayload, self._on_bus_predictor_changed)
        bus.subscribe(SocChangedPayload, self._on_bus_soc_changed)

        # Cleanup on destroy
        self.destroyed.connect(self._cleanup_bus_subscriptions)

    def _cleanup_bus_subscriptions(self) -> None:
        bus = self._ctrl.get_bus()
        bus.unsubscribe(
            TabInteractionChangedPayload, self._on_bus_tab_interaction_changed
        )
        bus.unsubscribe(RunStartedPayload, self._on_bus_run_started)
        bus.unsubscribe(RunFinishedPayload, self._on_bus_run_finished)
        bus.unsubscribe(ContextSwitchedPayload, self._on_bus_context_switched)
        bus.unsubscribe(MlChangedPayload, self._on_bus_ml_changed)
        bus.unsubscribe(TabAddedPayload, self._on_bus_tab_added)
        bus.unsubscribe(TabClosedPayload, self._on_bus_tab_closed)
        bus.unsubscribe(TabContentChangedPayload, self._on_bus_tab_content_changed)
        bus.unsubscribe(PredictorChangedPayload, self._on_bus_predictor_changed)
        bus.unsubscribe(SocChangedPayload, self._on_bus_soc_changed)

    def _on_bus_tab_interaction_changed(
        self, payload: TabInteractionChangedPayload
    ) -> None:
        snapshot = self._ctrl.get_tab_snapshot(payload.tab_id)
        self.refresh_tab_writeback(payload.tab_id, snapshot)
        self.refresh_tab_interaction(payload.tab_id, snapshot)

    def _on_bus_run_started(self, payload: RunStartedPayload) -> None:
        # Run lock now held by this tab.
        self.refresh_run_lock(payload.tab_id)

    def _on_bus_run_finished(self, payload: RunFinishedPayload) -> None:
        # Run lock released.
        self.refresh_run_lock(None)
        # Auto-switch to the second tab on a normal finish — RUN_FINISHED carries
        # the outcome directly, so the decision lives here. A stopped run
        # (outcome=cancelled) may leave a partial result, but the user interrupted
        # it on purpose; don't yank them away. RunService writes the result to
        # State before emitting RUN_FINISHED, so has_run_result is already set.
        # The second tab holds analysis widgets AND the Save section; for a
        # non-analysis adapter (flux_dep / power_dep) the analysis widgets are
        # hidden but Save stays — switching there lands the user on Save, which is
        # exactly what they want after a 2D sweep.
        if payload.outcome != "finished":
            return
        tab_w = self._tab_widgets.get(payload.tab_id)
        if tab_w is None:
            return
        snapshot = self._ctrl.get_tab_snapshot(payload.tab_id)
        assert snapshot.interaction is not None  # render snapshot fills live fields
        if snapshot.interaction.has_run_result:
            tab_w._left_tabs.setCurrentIndex(1)

    def _on_bus_context_switched(self, payload: ContextSwitchedPayload) -> None:
        del payload
        # cfg EvalValue refresh is the CfgEditorService's job now (it owns the
        # models — ADR-0008); here we only refresh the surrounding tab panels.
        self.refresh_context_panel()
        for tab_id in list(self._tab_widgets):
            snapshot = self._ctrl.get_tab_snapshot(tab_id)
            self.refresh_tab_writeback(tab_id, snapshot)
            self.refresh_tab_save_paths(tab_id, snapshot)
            self.refresh_tab_interaction(tab_id, snapshot)

    def _on_bus_ml_changed(self, payload: MlChangedPayload) -> None:
        del payload
        # cfg EvalValue refresh is the CfgEditorService's job now (ADR-0008);
        # here we only refresh the surrounding tab panels.
        for tab_id in list(self._tab_widgets):
            snapshot = self._ctrl.get_tab_snapshot(tab_id)
            self.refresh_tab_writeback(tab_id, snapshot)
            self.refresh_tab_interaction(tab_id, snapshot)

    def _on_bus_tab_added(self, payload: TabAddedPayload) -> None:
        tab_id = payload.tab_id
        adapter_name = payload.adapter_name
        logger.info("_on_bus_tab_added: tab_id=%r adapter=%r", tab_id, adapter_name)
        if tab_id in self._tab_widgets:
            return

        tab_label = adapter_name
        tab_w = ExpTabWidget(tab_id, self._ctrl)
        self._tab_widgets[tab_id] = tab_w
        self._tabs.addTab(tab_w, tab_label)
        self._tabs.setCurrentWidget(tab_w)

        # Bring the whole tab widget to life from one render snapshot (seed every
        # sub-view + wire controller signals) — the whole-tab analogue of
        # CfgFormWidget.attach.
        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        self._new_tab_btn.setEnabled(self._ctrl.get_running_tab_id() is None)
        tab_w.attach(snapshot, self)

    def _on_bus_tab_closed(self, payload: TabClosedPayload) -> None:
        tab_id = payload.tab_id
        logger.info("_on_bus_tab_closed: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.pop(tab_id, None)
        if tab_w is not None:
            tab_w.detach()
            index = self._tabs.indexOf(tab_w)
            if index >= 0:
                self._tabs.removeTab(index)
            tab_w.deleteLater()

        self.refresh_run_lock(self._ctrl.get_running_tab_id())

    def _on_bus_tab_content_changed(self, payload: TabContentChangedPayload) -> None:
        tab_id = payload.tab_id
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        self.refresh_tab_analyze_form(tab_id, snapshot)
        self.refresh_tab_post_analyze_form(tab_id, snapshot)
        self.refresh_tab_writeback(tab_id, snapshot)
        self.refresh_tab_save_paths(tab_id, snapshot)
        self.refresh_tab_figure(tab_id, snapshot)
        self.refresh_tab_post_figure(tab_id, snapshot)
        # The auto-switch to Analysis lives in _on_bus_run_finished (it needs the
        # run outcome); content refresh here is outcome-agnostic.
        self.refresh_tab_interaction(tab_id, snapshot)

    def _on_bus_predictor_changed(self, payload: PredictorChangedPayload) -> None:
        del payload
        self.refresh_predictor_panel()

    def _on_bus_soc_changed(self, payload: SocChangedPayload) -> None:
        del payload
        self.refresh_run_lock(self._ctrl.get_running_tab_id())

    # ------------------------------------------------------------------
    # ViewProtocol implementation
    # ------------------------------------------------------------------

    def _set_tab_running(
        self,
        tab_w: ExpTabWidget,
        snapshot: TabSnapshot,
    ) -> None:
        tab_w.update_interaction_state(snapshot)

    def refresh_tab_analyze_form(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        assert current.interaction is not None  # render snapshot fills live fields
        assert current.capabilities is not None  # render snapshot fills live fields
        # Non-analysis adapters (flux_dep / power_dep 2D sweeps) intentionally
        # have no analyze params after a run — there is no analyze form to fill,
        # so skip before the Fast-Fail below (which guards the *analysis* adapter
        # contract: a run result must carry initialized params).
        if current.capabilities.analysis is AnalysisMode.NONE:
            return
        if not current.interaction.has_run_result:
            return
        if current.analyze_params is None:
            raise RuntimeError("Run result has no initialized analyze parameters")
        tab_w.populate_analyze_params(current.analyze_params)
        tab_w.analyze_form.populate_values(current.analyze_params)

    def refresh_tab_post_analyze_form(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        assert current.capabilities is not None  # render snapshot fills live fields
        # Only post-analysis adapters have a post form; for the rest there is
        # nothing to fill. When the primary analyze result is invalidated the post
        # params are cleared (State), so there is no instance to populate — the
        # gate (update_interaction_state) disables the empty form.
        if not current.capabilities.post_analysis:
            return
        if current.post_analyze_params is None:
            return
        tab_w.populate_post_analyze_params(current.post_analyze_params)
        tab_w.post_analyze_form.populate_values(current.post_analyze_params)

    def refresh_tab_writeback(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        tab_w.update_writeback_items(list(current.writeback_items))

    def refresh_tab_save_paths(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        save_paths = current.save_paths
        if save_paths is not None:
            tab_w.set_save_paths(save_paths.data_path, save_paths.image_path)

    def refresh_tab_figure(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        figure = current.figure
        if figure is not None:
            self.show_analysis_image(tab_id, figure)

    def refresh_tab_post_figure(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        post_figure = current.post_figure
        if post_figure is not None:
            # Post + analyze share one container; ``refresh_tab_figure`` runs just
            # before this and renders ``tab.figure``, so when a post figure exists
            # it is drawn last and the shared container shows the most recent
            # (post) figure. On invalidation (post_figure is None) there is nothing
            # to do: the shared container already shows the primary figure.
            self.show_post_analysis_image(tab_id, post_figure)

    def refresh_run_lock(self, running_tab_id: str | None) -> None:
        logger.debug("refresh_run_lock: running_tab_id=%r", running_tab_id)
        self._new_tab_btn.setEnabled(running_tab_id is None)
        for tab_id, tab_w in self._tab_widgets.items():
            if self._ctrl.has_tab(tab_id):
                self._set_tab_running(tab_w, self._ctrl.get_tab_snapshot(tab_id))
        # Progress no longer cleared here — ProgressService.discard_operation on
        # the run's terminal path drops the container and notifies the tab's
        # listener, which re-renders to empty.

    def refresh_tab_interaction(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None or not self._ctrl.has_tab(tab_id):
            return
        self._set_tab_running(tab_w, snapshot or self._ctrl.get_tab_snapshot(tab_id))

    def refresh_context_panel(self) -> None:
        label = self._ctrl.get_active_context_label()
        if label is not None:
            # file-backed flux context is active
            self._ctx_label.setText(label)
            self._ctx_label.setStyleSheet("")
        elif self._ctrl.has_startup_context():
            # startup context (in-memory, no file sync)
            self._ctx_label.setText(
                "Startup context (in-memory) — set up project for persistence"
            )
            self._ctx_label.setStyleSheet("color: blue;")
        elif self._ctrl.has_project():
            self._ctx_label.setText(
                "Project set — select a context to enable Run/Analyze/Save"
            )
            self._ctx_label.setStyleSheet("color: orange;")
        else:
            self._ctx_label.setText("No project set — use Project… to configure")
            self._ctx_label.setStyleSheet("color: gray;")
        for tab_id, tab_w in self._tab_widgets.items():
            if self._ctrl.has_tab(tab_id):
                self._set_tab_running(tab_w, self._ctrl.get_tab_snapshot(tab_id))

    def refresh_inspect_panel(self) -> None:
        inspect = self._open_dialogs.get(DialogName.INSPECT)
        if inspect is not None and inspect.isVisible():
            # InspectDialog defines ``refresh``; cast through ``Any`` to avoid
            # importing the concrete class in the hot signature surface.
            from typing import cast

            cast(Any, inspect).refresh()

    def refresh_predictor_panel(self) -> None:
        info = self._ctrl.get_predictor_info()
        if info is None:
            self._predictor_label.setText("none")
            self._predictor_label.setStyleSheet("")
        else:
            flux_bias = info["flux_bias"]
            self._predictor_label.setText(f"loaded (flux_bias={flux_bias:.4g})")
            self._predictor_label.setStyleSheet("color: green;")

    def make_live_container(self, tab_id: str) -> Any:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return None
        tab_w.reset_plot()  # clear prior liveplot before new run/analyze
        return tab_w._figure_container

    def mount_interactive_analysis(
        self,
        tab_id: str,
        session_factory: Callable[[Any], Any],
        on_finish: Callable[[Any], None],
    ) -> None:
        """RenderHost impl: mount an INTERACTIVE adapter's live picker in the tab's
        plot stack. Build the host widget, hand it to the adapter to set up the
        session, render the session's actions, and wire Done -> on_finish(session).
        The View knows nothing of the interaction (lines / flux); it just forwards
        events and shows session.info_text()."""
        from zcu_tools.gui.app.main.ui.interactive_analysis import (
            InteractiveAnalysisWidget,
        )

        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        tab_w.reset_plot()
        # The Controller satisfies InteractiveHostEnv (run_background via bg's
        # pool); the widget pulls only that one capability through the port.
        widget = InteractiveAnalysisWidget(self._ctrl)
        session = session_factory(widget)  # the widget IS the InteractiveHost
        widget.bind(session, on_done=lambda: on_finish(session))
        tab_w._plot_stack.addWidget(widget)
        tab_w._plot_stack.setCurrentWidget(widget)

    def current_left_panel_width(self) -> int:
        """RenderHost impl: the active tab's left-panel width (the single
        persistence value sourced from the View). Falls back to the default when
        no tab is open."""
        from zcu_tools.gui.app.main.state import DEFAULT_LEFT_PANEL_WIDTH

        current = self._tabs.currentWidget()
        if isinstance(current, ExpTabWidget):
            return current._splitter_left_saved
        return DEFAULT_LEFT_PANEL_WIDTH

    def notify_diagnostic(self, severity: str, title: str, message: str) -> None:
        """DiagnosticSink impl (ADR-0013): render a Controller diagnostic the Qt
        way — error pops a modal dialog, info goes to the status bar."""
        if severity == "error":
            self.show_error_dialog(title or "Error", message)
        else:
            self.show_status_message(message)

    def show_status_message(self, message: str) -> None:
        logger.info("status: %s", message)
        self._status_bar.showMessage(message)

    def show_error_dialog(self, title: str, message: str) -> None:
        from qtpy.QtWidgets import QMessageBox

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()

    def show_plot(self, tab_id: str, fig: Any) -> None:  # Phase 11
        logger.debug("show_plot: tab_id=%r fig=%s", tab_id, type(fig).__name__)

    def show_analysis_image(self, tab_id: str, fig: Any) -> None:
        logger.debug("show_analysis_image: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        tab_w.show_analysis_figure(fig)

    def show_post_analysis_image(self, tab_id: str, fig: Any) -> None:
        # Post figures share the primary right-pane container (the container shows
        # the most recently produced figure), so this routes through the same
        # render path as the analyze figure.
        logger.debug("show_post_analysis_image: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        tab_w.show_analysis_figure(fig)

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    def _on_new_tab_requested(self) -> None:
        menu = QMenu(self)
        submenus: dict[tuple[str, ...], QMenu] = {}

        def _get_or_create_submenu(path: tuple[str, ...]) -> QMenu:
            cached = submenus.get(path)
            if cached is not None:
                return cached
            if len(path) == 1:
                parent_menu = menu
            else:
                parent_menu = _get_or_create_submenu(path[:-1])
            sub_menu = parent_menu.addMenu(path[-1])
            if sub_menu is None:
                raise RuntimeError(f"Failed to create submenu: {'/'.join(path)}")
            submenus[path] = sub_menu
            return sub_menu

        for name in self._ctrl.get_adapter_names():
            parts = tuple(name.split("/"))
            if len(parts) == 1:
                action = menu.addAction(parts[0])
                action.setData(name)  # type: ignore[union-attr]
                continue
            parent_menu = _get_or_create_submenu(parts[:-1])
            action = parent_menu.addAction(parts[-1])
            action.setData(name)  # type: ignore[union-attr]

        action = menu.exec(
            self._new_tab_btn.mapToGlobal(  # type: ignore[assignment]
                self._new_tab_btn.rect().bottomLeft()
            )
        )
        if action is None:
            return
        adapter_name = action.data()
        if not adapter_name:
            return

        self._ctrl.new_tab(adapter_name)

    def _on_tab_close_requested(self, index: int) -> None:
        tab_w = self._tabs.widget(index)
        if not isinstance(tab_w, ExpTabWidget):
            return
        tab_id = tab_w.tab_id
        logger.info("_on_tab_close_requested: tab_id=%r", tab_id)
        self._ctrl.close_tab(tab_id)

    def _on_current_tab_changed(self, index: int) -> None:
        widget = self._tabs.widget(index)
        if not isinstance(widget, ExpTabWidget):
            return
        if not self._ctrl.has_tab(widget.tab_id):
            return
        self._ctrl.set_active_tab(widget.tab_id)

    def _resolve_tab_widget(self, tab_id: str, action: str) -> ExpTabWidget | None:
        """Look up the widget; log + bail if tab_id is unknown to the controller."""
        if not self._ctrl.has_tab(tab_id):
            logger.warning("%s: unknown tab_id=%r — ignoring", action, tab_id)
            return None
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            logger.warning(
                "%s: tab_id=%r missing in view registry — ignoring", action, tab_id
            )
            return None
        return tab_w

    def _on_run_stop_clicked(self, tab_id: str) -> None:
        tab_w = self._resolve_tab_widget(tab_id, "_on_run_stop_clicked")
        if tab_w is None:
            return
        interaction = self._ctrl.get_tab_snapshot(tab_id).interaction
        assert interaction is not None  # render snapshot fills live fields
        if interaction.is_running:
            logger.info("_on_run_stop_clicked: stop requested tab_id=%r", tab_id)
            self._ctrl.cancel_run()
            return
        logger.info("_on_run_stop_clicked: run requested tab_id=%r", tab_id)
        if not tab_w.cfg_form.is_valid():
            reason = tab_w.cfg_form.first_invalid_reason()
            if reason:
                msg = f"Config invalid: {reason}"
            else:
                msg = "Config has unset fields — fill required values before running"
            logger.warning("_on_run_stop_clicked: blocked — %s", msg)
            self.show_status_message(msg)
            return
        self._ctrl.start_run(tab_id)

    def _on_analyze_clicked(self, tab_id: str) -> None:
        logger.info("_on_analyze_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_analyze_clicked")
        if tab_w is None:
            return
        self._ctrl.analyze(tab_id, tab_w.read_analyze_params())

    def _on_post_analyze_clicked(self, tab_id: str) -> None:
        logger.info("_on_post_analyze_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_post_analyze_clicked")
        if tab_w is None:
            return
        self._ctrl.start_post_analyze(tab_id, tab_w.read_post_analyze_params())

    def _on_writeback_inline_apply(self, tab_id: str) -> None:
        logger.info("_on_writeback_inline_apply: tab_id=%r", tab_id)
        if not self._ctrl.has_tab(tab_id):
            logger.warning(
                "_on_writeback_inline_apply: unknown tab_id=%r — ignoring", tab_id
            )
            return
        applied_ids = self._ctrl.apply_writeback(tab_id)
        if applied_ids:
            self.show_status_message(f"Writeback applied: {', '.join(applied_ids)}")

    def _on_save_data_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_data_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_data_clicked")
        if tab_w is None:
            return
        path = tab_w.get_data_path()
        self._ctrl.save_data(tab_id, path, comment=tab_w.get_comment())

    def _on_save_image_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_image_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_image_clicked")
        if tab_w is None:
            return
        path = tab_w.get_image_path()
        self._ctrl.save_image(tab_id, path)

    def _on_post_save_image_clicked(self, tab_id: str) -> None:
        logger.info("_on_post_save_image_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_post_save_image_clicked")
        if tab_w is None:
            return
        path = tab_w.get_post_image_path()
        self._ctrl.save_post_image(tab_id, path)

    def _on_save_result_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_result_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_result_clicked")
        if tab_w is None:
            return
        data_path = tab_w.get_data_path()
        image_path = tab_w.get_image_path()
        self._ctrl.save_result(
            tab_id, data_path, image_path, comment=tab_w.get_comment()
        )

    def _on_setup_clicked(self) -> None:
        self.open_dialog(DialogName.SETUP)

    def _on_devices_clicked(self) -> None:
        self.open_dialog(DialogName.DEVICE)

    def _on_predictor_clicked(self) -> None:
        self.open_dialog(DialogName.PREDICTOR)

    def _on_inspect_clicked(self) -> None:
        self.open_dialog(DialogName.INSPECT)

    # ------------------------------------------------------------------
    # Dialog API — single entry point shared by UI clicks and remote control
    # ------------------------------------------------------------------

    def _build_dialog(self, name: DialogName) -> QDialog:
        """Construct a fresh QDialog for ``name``.

        Per-name imports are deferred to avoid heavy front-loading and to
        keep test fixtures from pulling in unused dialog modules.
        """
        if name is DialogName.SETUP:
            from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

            return SetupDialog(self._ctrl, parent=self)
        if name is DialogName.DEVICE:
            from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

            return DeviceDialog(self._ctrl, parent=self)
        if name is DialogName.PREDICTOR:
            from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog

            return PredictorDialog(self._ctrl, parent=self)
        if name is DialogName.INSPECT:
            from .inspect_dialog import InspectDialog

            return InspectDialog(self._ctrl, bus=self._ctrl.get_bus(), parent=self)
        if name is DialogName.STARTUP:
            # STARTUP dialog needs ``startup_mode=True`` and is normally opened
            # by the application bootstrap, not by this generic factory. We
            # still build one here so a remote ``dialog.open STARTUP`` works
            # after the initial bootstrap has dismissed the original.
            from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

            return SetupDialog(self._ctrl, parent=self, startup_mode=True)
        raise ValueError(f"Unknown DialogName: {name!r}")  # pragma: no cover

    def open_dialog(self, name: DialogName) -> None:
        """Open the named dialog non-modally, or raise it if already open.

        Idempotent: a second ``open_dialog(name)`` while the dialog is
        visible just brings it to the front (``raise_`` + ``activateWindow``).
        """
        existing = self._open_dialogs.get(name)
        if existing is not None:
            try:
                existing.raise_()
                existing.activateWindow()
                if not existing.isVisible():
                    existing.show()
                return
            except RuntimeError:
                # Underlying Qt object was destroyed but registry was not
                # cleaned up — drop the stale entry and fall through to
                # rebuild.
                self._open_dialogs.pop(name, None)

        dlg = self._build_dialog(name)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._open_dialogs[name] = dlg
        # ``finished`` fires for both accept and reject; the lambda must
        # tolerate signal payload (status code) being passed through.
        dlg.finished.connect(lambda _status, n=name: self._open_dialogs.pop(n, None))
        dlg.open()

    def close_dialog(self, name: DialogName) -> None:
        """Close the named dialog if it is currently open."""
        existing = self._open_dialogs.get(name)
        if existing is None:
            return
        try:
            existing.reject()
        except RuntimeError:
            # Dialog already destroyed; tidy the registry.
            self._open_dialogs.pop(name, None)

    def list_open_dialogs(self) -> list[DialogName]:
        return list(self._open_dialogs.keys())

    def register_dialog(self, name: DialogName, dialog: QDialog) -> None:
        """Register a dialog that was constructed outside ``open_dialog``.

        ``app.py`` uses this for the bootstrap startup dialog so the remote
        ``dialog.list_open`` query and ``dialog.close STARTUP`` work
        uniformly. The caller is responsible for ``setAttribute
        (WA_DeleteOnClose)`` and for ``open()`` / ``show()`` — this helper
        only wires the registry cleanup on ``finished``.
        """
        self._open_dialogs[name] = dialog
        dialog.finished.connect(lambda _status, n=name: self._open_dialogs.pop(n, None))

    # ------------------------------------------------------------------
    # Remote view query helpers
    # ------------------------------------------------------------------

    def get_view_snapshot(self) -> dict[str, object]:
        """Capture the visible window state as a JSON-friendly dict."""
        active_id: str | None = None
        if self._tabs.count() > 0:
            current = self._tabs.currentWidget()
            for tid, tab_w in self._tab_widgets.items():
                if tab_w is current:
                    active_id = tid
                    break
        return {
            "active_tab_id": active_id,
            "tab_ids": list(self._tab_widgets.keys()),
            "context_label": self._ctx_label.text() if self._ctx_label else "",
            "predictor_label": (
                self._predictor_label.text() if self._predictor_label else ""
            ),
            "status": self._status_bar.currentMessage() if self._status_bar else "",
            "open_dialogs": [n.value for n in self._open_dialogs.keys()],
        }

    def take_screenshot(self, tab_id: str | None = None) -> bytes:
        """Grab the window (or a single tab) and return raw PNG bytes.

        ``tab_id`` must be the active tab; off-screen tabs grab as blank,
        so we refuse them explicitly. Returning bytes (not base64) keeps
        this helper Qt-only; the dispatcher base64-encodes for the wire.
        """
        from qtpy.QtCore import QBuffer, QIODevice  # type: ignore[attr-defined]

        target: QWidget
        if tab_id is None:
            target = self
        else:
            tab_w = self._tab_widgets.get(tab_id)
            if tab_w is None:
                raise RuntimeError(f"unknown tab_id: {tab_id!r}")
            if self._tabs.currentWidget() is not tab_w:
                raise RuntimeError(
                    f"tab {tab_id!r} is not the active tab; "
                    "remote screenshot requires it to be active first"
                )
            target = tab_w
        pixmap = target.grab()
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        if not pixmap.save(buf, "PNG"):
            raise RuntimeError("Qt failed to encode the grabbed pixmap as PNG")
        # ``QBuffer.data()`` returns ``QByteArray``; its ``.data()`` method
        # produces native ``bytes``.
        ba = buf.data()
        return bytes(ba.data())  # type: ignore[arg-type]

    def take_figure_screenshot(self, tab_id: str) -> bytes:
        """Render a tab's figure to PNG bytes at the fixed export size.

        Renders the live figure via savefig (not ``canvas.grab()``) so the
        screenshot has the same window-independent geometry as a saved image,
        rather than tracking the current widget pixel size.
        """
        from matplotlib.figure import Figure

        from zcu_tools.gui.app.main.figure_export import render_figure_png

        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            raise RuntimeError(f"unknown tab_id: {tab_id!r}")
        canvas = tab_w._plot_stack.currentWidget()
        if canvas is None or canvas is tab_w._plot_placeholder:
            raise RuntimeError(f"tab {tab_id!r} has no figure yet")
        figure = getattr(canvas, "figure", None)
        if not isinstance(figure, Figure):
            raise RuntimeError(f"tab {tab_id!r} canvas has no matplotlib figure")
        return render_figure_png(figure)

    def take_dialog_screenshot(self, dialog_name: DialogName) -> bytes:
        """Grab a currently-open dialog and return raw PNG bytes."""
        from qtpy.QtCore import QBuffer, QIODevice  # type: ignore[attr-defined]

        dlg = self._open_dialogs.get(dialog_name)
        if dlg is None:
            raise RuntimeError(f"dialog {dialog_name.value!r} is not currently open")
        pixmap = dlg.grab()
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        if not pixmap.save(buf, "PNG"):
            raise RuntimeError(
                f"Qt failed to encode {dialog_name.value!r} dialog as PNG"
            )
        return bytes(buf.data().data())  # type: ignore[arg-type]

    def request_shutdown(self) -> None:
        """Programmatic close (the app.shutdown RPC). Runs on the Qt main thread
        via the remote dispatch marshal. Does the same work as a user's
        window-close — cancel every live operation, wait for them to stop,
        persist session, tear down remote, quit — but without the interactive
        confirmation (no user to answer it).

        The cancel-and-wait is deferred to the next event-loop turn so the
        triggering RPC's reply is written back before the remote service tears
        down (else the agent's app.shutdown would race the socket teardown)."""
        from qtpy.QtCore import QTimer  # type: ignore[attr-defined]

        QTimer.singleShot(0, lambda: self._ctrl.begin_shutdown(self._perform_close))

    def _perform_close(self, a0: QCloseEvent | None = None) -> None:
        """The actual teardown: persist session, stop remote, accept the close.
        Runs once every cancelled operation has settled (or timed out), driven by
        the Controller's shutdown coordinator. Shared by closeEvent (user) and
        request_shutdown (RPC)."""
        self._closing = True
        self._ctrl.persist_all()
        set_shutting_down(True)
        # Tear down remote control before the Qt main loop exits so any in-flight
        # RPC sees a clean shutdown (timeout / EPIPE) rather than a dead Controller.
        remote = getattr(self, "remote_control_service", None)
        if remote is not None:
            remote.stop()
            self.remote_control_service = None
        if a0 is not None:
            super().closeEvent(a0)
        else:
            self.close()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        # Second pass: _perform_close → self.close() re-enters here once teardown
        # has begun; accept it straight through.
        if self._closing:
            if a0 is not None:
                super().closeEvent(a0)
            return
        # A user window-close cancels every live operation, then closes once they
        # stop (or a timeout forces it). Confirm first if work is in progress —
        # closing will interrupt it. The wait is asynchronous, so ignore this
        # event now; the coordinator drives _perform_close when ready.
        active = self._ctrl.active_operation_count()
        if active > 0:
            if a0 is None:
                return
            from qtpy.QtWidgets import QMessageBox

            answer = QMessageBox.question(
                self,
                "Operations in progress",
                f"Cancel {active} operation(s) in progress and close once they stop?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                a0.ignore()
                return
            a0.ignore()
        elif a0 is not None:
            a0.ignore()
        # Defer begin_shutdown to the next event-loop turn (mirrors
        # request_shutdown). When idle, the shutdown coordinator settles
        # synchronously and calls _perform_close → self.close(), which would
        # otherwise re-enter this closeEvent within its own stack — Qt does not
        # honour a self.close() issued from inside a closeEvent handler, so the
        # first click would appear to do nothing. The singleShot breaks out of
        # this stack first.
        from qtpy.QtCore import QTimer  # type: ignore[attr-defined]

        QTimer.singleShot(0, lambda: self._ctrl.begin_shutdown(self._perform_close))
