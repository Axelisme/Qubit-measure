"""Per-experiment tab widget for the measure-gui main window."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.main.adapter import AnalysisMode, CfgSchema
from zcu_tools.gui.plotting import FigureContainer, attach_existing_figure_to_container
from zcu_tools.gui.session.ui.progress_stack import ProgressStack
from zcu_tools.gui.widgets import DialogPresenter, QtDialogPresenter

logger = logging.getLogger(__name__)


from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtGui import (  # type: ignore[attr-defined]
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
from .cfg_form import CfgFormWidget
from .fields import _CollapsibleSection
from .writeback_widget import WritebackWidget

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.main.adapter import WritebackItem
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services import TabSnapshot


class TabActions(Protocol):
    """Tab-level actions supplied by the top-level window boundary."""

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
        self._writeback_count: int = 0
        # editor_id of this tab's shared cfg-editor session (set on bind, when
        # the cfg_form's live model exists). Exposed to agents via tab.snapshot.
        self._cfg_editor_id: str | None = None
        # The action boundary is retained for Reset; button slots close over it.
        self._actions: TabActions | None = None

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

        self.load_data_btn = QPushButton("Load Data...")
        analysis_layout.addWidget(self.load_data_btn)

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
        # Kept as an attribute so the feedback dock host can insert the panel
        # below the figure (mount_feedback_panel inserts it at index 1, directly
        # under the plot stack).
        self._plot_layout = QVBoxLayout(plot_panel)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)

        self._plot_stack = QStackedWidget()

        self._plot_placeholder = QLabel("(no plot yet)")
        self._plot_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._plot_stack.addWidget(self._plot_placeholder)
        self._figure_container = FigureContainer(
            self._plot_stack, self._plot_placeholder
        )

        self._plot_layout.addWidget(self._plot_stack, stretch=1)
        splitter.addWidget(plot_panel)

        splitter.setCollapsible(0, True)
        self._update_left_panel_controls()
        self._schedule_handle_layout()

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
        if snapshot.post_analyze_params is not None and self.has_post_analyze_params():
            self.post_analyze_form.populate_values(snapshot.post_analyze_params)
        if snapshot.save_paths is not None:
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

    def get_data_path(self) -> str:
        return self._data_path_edit.text()

    def get_image_path(self) -> str:
        return self._image_path_edit.text()

    def get_post_image_path(self) -> str:
        return self._post_image_path_edit.text()

    def get_comment(self) -> str:
        return self._comment_edit.toPlainText()

    def reset_plot(self) -> None:
        """Remove all canvases from plot_stack, revert to placeholder.

        This is the genuine-invalidation teardown for BOTH the analyze and post
        figures (they coexist in the same stack); it runs before each new
        run/analyze so stale canvases never linger.
        """
        self._figure_container.clear_dynamic_canvases()

    def show_analysis_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the plot area and bring it to front.

        The run/analyze figure (``tab.figure``) and the post-analysis figure
        (``tab.post_figure``) are two distinct Figure objects sharing this one
        container's QStackedWidget. They coexist as separate canvases; attaching
        a figure only switches the stack to it (``attach_canvas`` setsCurrent),
        it must NOT evict the other figure's canvas — doing so deletes a canvas
        still owned by a live figure and the next attach reuses the dead wrapper.
        Genuine teardown of both canvases happens in ``reset_plot`` (before a new
        run/analyze) and on tab close.
        """
        canvas = attach_existing_figure_to_container(fig, self._figure_container)
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached analysis canvas does not support draw()")
        draw()
        logger.debug("show_analysis_figure: tab_id=%r canvas set", self.tab_id)

    def _on_cfg_validity_changed(self, valid: bool) -> None:
        del valid

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
        self.cfg_form.set_editing_enabled(idle)
        self.reset_btn.setEnabled(idle)

        # Non-analysis adapters (flux_dep / power_dep) hide only the analysis
        # widgets, NOT the whole tab — the Save section lives in this same tab and
        # must stay reachable so any run can be saved. (Writeback is already
        # gated by item count in update_writeback_items; hide it here too so it
        # never lingers from a previous analysis adapter on the same tab.)
        has_analysis = capabilities.analysis is not AnalysisMode.NONE
        self._analyze_section.setVisible(has_analysis)
        self.load_data_btn.setVisible(has_analysis)
        self.analyze_btn.setVisible(has_analysis)
        if not has_analysis:
            self.writeback_section.setVisible(False)
        # The second tab carries analysis widgets + Save; when analysis is hidden
        # only Save remains, so label it accordingly instead of "Analysis".
        self._left_tabs.setTabText(1, "Analysis" if has_analysis else "Save")
        self.load_data_btn.setEnabled(idle and has_analysis and state.has_context)
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

    def _bind_to_controller(self, actions: TabActions) -> None:
        tab_id = self.tab_id
        # Held so the Reset handler can refresh interaction state after re-seeding
        # (the only post-bind path that needs the actions off a button slot).
        self._actions = actions

        def validity_cb(_valid: bool) -> None:
            actions.refresh_interaction(tab_id)

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
        self.run_btn.clicked.connect(lambda: actions.run_or_stop(tab_id))
        self.load_data_btn.clicked.connect(lambda: actions.load_data(tab_id))
        self.analyze_btn.clicked.connect(lambda: actions.analyze(tab_id))
        self.post_analyze_btn.clicked.connect(lambda: actions.post_analyze(tab_id))
        self.writeback_widget.apply_requested.connect(
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
        self._progress_unsub()
        # Detach the widget first (drop its signal bindings + widget tree), then
        # tell the service to tear down the model it owns (ADR-0008).
        self.cfg_form.detach()
        if self._cfg_editor_id is not None:
            self._ctrl.teardown_cfg_editor(self._cfg_editor_id)
            self._cfg_editor_id = None
