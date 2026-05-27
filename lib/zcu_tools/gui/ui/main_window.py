"""MainWindow — the top-level View for the v2_gui framework.

Implements ViewProtocol; all state lives in Controller/State, never here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from zcu_tools.gui.adapter import CfgSchema
from zcu_tools.gui.event_bus import (
    ContextSwitchedPayload,
    DeviceSetupChangedPayload,
    GuiEvent,
    MlChangedPayload,
    PredictorChangedPayload,
    RunLockChangedPayload,
    SocChangedPayload,
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.plot_host import (
    FigureContainer,
    attach_existing_figure_to_container,
    remove_canvas,
    set_shutting_down,
)
from zcu_tools.gui.state import TabInteractionState

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
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analyze_form import AnalyzeFormWidget
from .cfg_form import (
    CfgFormWidget,
)
from .fields import (
    _CollapsibleSection,
)
from .progress_stack import ProgressStack
from .writeback_widget import WritebackWidget

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.adapter import CfgSchema, WritebackItem
    from zcu_tools.gui.controller import Controller
    from zcu_tools.gui.services import TabViewSnapshot
    from zcu_tools.meta_tool import ModuleLibrary


# ---------------------------------------------------------------------------
# Per-experiment tab widget
# ---------------------------------------------------------------------------


class _PanelEdgeHandle(QToolButton):
    """Boundary handle for collapsing/expanding the left panel."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
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
        self, tab_id: str, ctrl: "Controller", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.tab_id = tab_id
        self._ctrl = ctrl
        self._writeback_count: int = 0

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

        # splitter holds two panes: left (tab panel) | right (plot)
        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]

        content_row.addWidget(splitter, stretch=1)

        self._splitter = splitter
        self._splitter_left_saved = ctrl.get_persisted_left_panel_width()
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

        btn_row = QHBoxLayout()
        self.save_data_btn = QPushButton("Save Data")
        self.save_image_btn = QPushButton("Save Image")
        self.save_both_btn = QPushButton("Save Both")
        btn_row.addWidget(self.save_data_btn)
        btn_row.addWidget(self.save_image_btn)
        btn_row.addWidget(self.save_both_btn)
        save_layout.addRow("", btn_row)

        analysis_layout.addWidget(save_section)
        analysis_layout.addStretch()

        analysis_scroll.setWidget(analysis_inner)
        self._left_tabs.addTab(analysis_scroll, "Analysis")

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

        self._canvas_widget: Optional[QWidget] = None

        plot_layout.addWidget(self._plot_stack, stretch=1)
        splitter.addWidget(plot_panel)

        splitter.setCollapsible(0, True)
        self._update_left_panel_controls()
        self._schedule_handle_layout()

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
                self._ctrl.save_left_panel_width(sizes[0])
        self._schedule_handle_layout()

    def _schedule_handle_layout(self) -> None:
        QTimer.singleShot(0, self._layout_collapsed_handle)

    # ── cfg helpers ───────────────────────────────────────────────────────

    def populate_cfg(self, schema: "CfgSchema", ctrl: "Controller") -> None:
        self.cfg_form.populate(schema, ctrl)

    def read_schema(self) -> "CfgSchema":
        return self.cfg_form.read_schema()

    # ── populate / refresh helpers ────────────────────────────────────────

    def populate_analyze_params(self, instance: object) -> None:
        self.analyze_form.populate(instance)

    def read_analyze_params(self) -> object:
        return self.analyze_form.read_params()

    def has_analyze_params(self) -> bool:
        return self.analyze_form.has_params()

    def update_writeback_items(self, items: list["WritebackItem"]) -> None:
        self._writeback_count = sum(1 for item in items if item.selected)
        self.writeback_widget.populate(items)
        self.writeback_section.setVisible(len(items) > 0)

    def _on_browse_data_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save data file", "", "HDF5 files (*.h5);;All files (*)"
        )
        if path:
            self._data_path_edit.setText(path)

    def _on_browse_image_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image file", "", "PNG files (*.png);;All files (*)"
        )
        if path:
            self._image_path_edit.setText(path)

    def set_save_paths(self, data_path: str, image_path: str) -> None:
        if data_path:
            self._data_path_edit.blockSignals(True)
            self._data_path_edit.setText(data_path)
            self._data_path_edit.blockSignals(False)
        if image_path:
            self._image_path_edit.blockSignals(True)
            self._image_path_edit.setText(image_path)
            self._image_path_edit.blockSignals(False)

    def get_data_path(self) -> str:
        return self._data_path_edit.text()

    def get_image_path(self) -> str:
        return self._image_path_edit.text()

    def reset_plot(self) -> None:
        """Remove all canvases from plot_stack, revert to placeholder."""
        self._figure_container.clear_dynamic_canvases()
        self._canvas_widget = None

    def show_analysis_figure(self, fig: "Figure") -> None:
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

    def update_interaction_state(self, snapshot: TabViewSnapshot) -> None:
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

        has_analysis = capabilities.supports_analysis
        self._left_tabs.setTabVisible(1, has_analysis)
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
        self.save_both_btn.setEnabled(
            idle and state.has_active_context and state.has_figure
        )
        self.writeback_widget.setEnabled(
            idle and state.has_context and state.has_analyze_result
        )

    def bind_to_controller(self, main_window: "MainWindow") -> None:
        tab_id = self.tab_id

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
        self.analyze_form.params_changed.connect(
            lambda instance: self._ctrl.update_tab_analyze_param_instance(
                tab_id, instance
            )
        )
        self._data_path_edit.textChanged.connect(save_paths_cb)
        self._image_path_edit.textChanged.connect(save_paths_cb)
        self.run_btn.clicked.connect(lambda: main_window._on_run_stop_clicked(tab_id))
        self.analyze_btn.clicked.connect(
            lambda: main_window._on_analyze_clicked(tab_id)
        )
        self.writeback_widget.apply_requested.connect(
            lambda items: main_window._on_writeback_inline_apply(tab_id, items)
        )
        self.save_data_btn.clicked.connect(
            lambda: main_window._on_save_data_clicked(tab_id)
        )
        self.save_image_btn.clicked.connect(
            lambda: main_window._on_save_image_clicked(tab_id)
        )
        self.save_both_btn.clicked.connect(
            lambda: main_window._on_save_both_clicked(tab_id)
        )

        self._validity_cb = validity_cb
        self._schema_cb = schema_cb

    def unbind_from_controller(self) -> None:
        if hasattr(self, "_validity_cb"):
            self.cfg_form.validity_changed.disconnect(self._validity_cb)
        if hasattr(self, "_schema_cb"):
            self.cfg_form.schema_changed.disconnect(self._schema_cb)
        self.cfg_form.clear()


# ---------------------------------------------------------------------------
# MainWindow — implements ViewProtocol
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Top-level window; implements ViewProtocol for Controller callbacks."""

    def __init__(self, controller: "Controller") -> None:
        super().__init__()
        self._ctrl = controller
        self._tab_widgets: dict[str, ExpTabWidget] = {}
        self._inspect_dialog: Any = None
        self._shutdown_waiting_for_device_setup = False

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
            GuiEvent.TAB_INTERACTION_CHANGED, self._on_bus_tab_interaction_changed
        )
        bus.subscribe(GuiEvent.RUN_LOCK_CHANGED, self._on_bus_run_lock_changed)
        bus.subscribe(GuiEvent.CONTEXT_SWITCHED, self._on_bus_context_switched)
        bus.subscribe(GuiEvent.ML_CHANGED, self._on_bus_ml_changed)
        bus.subscribe(GuiEvent.TAB_ADDED, self._on_bus_tab_added)
        bus.subscribe(GuiEvent.TAB_CLOSED, self._on_bus_tab_closed)
        bus.subscribe(GuiEvent.TAB_CONTENT_CHANGED, self._on_bus_tab_content_changed)
        bus.subscribe(GuiEvent.PREDICTOR_CHANGED, self._on_bus_predictor_changed)
        bus.subscribe(GuiEvent.SOC_CHANGED, self._on_bus_soc_changed)
        bus.subscribe(GuiEvent.DEVICE_SETUP_CHANGED, self._on_bus_device_setup_changed)

        # Cleanup on destroy
        self.destroyed.connect(self._cleanup_bus_subscriptions)

    def _cleanup_bus_subscriptions(self) -> None:
        bus = self._ctrl.get_bus()
        bus.unsubscribe(
            GuiEvent.TAB_INTERACTION_CHANGED, self._on_bus_tab_interaction_changed
        )
        bus.unsubscribe(GuiEvent.RUN_LOCK_CHANGED, self._on_bus_run_lock_changed)
        bus.unsubscribe(GuiEvent.CONTEXT_SWITCHED, self._on_bus_context_switched)
        bus.unsubscribe(GuiEvent.ML_CHANGED, self._on_bus_ml_changed)
        bus.unsubscribe(GuiEvent.TAB_ADDED, self._on_bus_tab_added)
        bus.unsubscribe(GuiEvent.TAB_CLOSED, self._on_bus_tab_closed)
        bus.unsubscribe(GuiEvent.TAB_CONTENT_CHANGED, self._on_bus_tab_content_changed)
        bus.unsubscribe(GuiEvent.PREDICTOR_CHANGED, self._on_bus_predictor_changed)
        bus.unsubscribe(GuiEvent.SOC_CHANGED, self._on_bus_soc_changed)
        bus.unsubscribe(
            GuiEvent.DEVICE_SETUP_CHANGED, self._on_bus_device_setup_changed
        )

    def _on_bus_tab_interaction_changed(
        self, payload: TabInteractionChangedPayload
    ) -> None:
        self.refresh_tab_interaction(payload.tab_id)

    def _on_bus_run_lock_changed(self, payload: RunLockChangedPayload) -> None:
        self.refresh_run_lock(payload.running_tab_id)

    def _on_bus_device_setup_changed(self, payload: DeviceSetupChangedPayload) -> None:
        if self._shutdown_waiting_for_device_setup and payload.active_setup is None:
            QTimer.singleShot(0, self.close)

    def _on_bus_context_switched(self, payload: ContextSwitchedPayload) -> None:
        del payload
        self.refresh_context_panel()
        for tab_id in list(self._tab_widgets):
            tab_w = self._tab_widgets.get(tab_id)
            if tab_w is not None:
                tab_w.cfg_form.refresh_external(GuiEvent.CONTEXT_SWITCHED)
            snapshot = self._ctrl.get_tab_snapshot(tab_id)
            self.refresh_tab_writeback(tab_id, snapshot)
            self.refresh_tab_save_paths(tab_id, snapshot)
            self.refresh_tab_interaction(tab_id, snapshot)

    def _on_bus_ml_changed(self, payload: MlChangedPayload) -> None:
        del payload
        for tab_id in list(self._tab_widgets):
            tab_w = self._tab_widgets.get(tab_id)
            if tab_w is not None:
                tab_w.cfg_form.refresh_external(GuiEvent.ML_CHANGED)
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

        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        tab_w.populate_cfg(snapshot.cfg_schema, self._ctrl)
        if snapshot.analyze_params is not None and tab_w.has_analyze_params():
            tab_w.analyze_form.populate_values(snapshot.analyze_params)
        if snapshot.save_paths is not None:
            tab_w.set_save_paths(
                snapshot.save_paths.data_path,
                snapshot.save_paths.image_path,
            )

        # refresh state (enables/disables buttons based on context)
        self._new_tab_btn.setEnabled(self._ctrl.get_running_tab_id() is None)
        tab_w.update_interaction_state(snapshot)

        # wire all signals/buttons for this tab
        tab_w.bind_to_controller(self)

    def _on_bus_tab_closed(self, payload: TabClosedPayload) -> None:
        tab_id = payload.tab_id
        logger.info("_on_bus_tab_closed: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.pop(tab_id, None)
        if tab_w is not None:
            tab_w.unbind_from_controller()
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
        self.refresh_tab_writeback(tab_id, snapshot)
        self.refresh_tab_save_paths(tab_id, snapshot)
        self.refresh_tab_figure(tab_id, snapshot)
        # auto-switch to Analysis tab when a new run result first arrives
        if snapshot.interaction.has_run_result:
            tab_w._left_tabs.setCurrentIndex(1)
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
        tab_w: "ExpTabWidget",
        snapshot: "TabViewSnapshot",
    ) -> None:
        tab_w.update_interaction_state(snapshot)

    def refresh_tab_analyze_form(
        self, tab_id: str, snapshot: Optional["TabViewSnapshot"] = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        if not current.interaction.has_run_result:
            return
        if current.analyze_params is None:
            raise RuntimeError("Run result has no initialized analyze parameters")
        tab_w.populate_analyze_params(current.analyze_params)
        tab_w.analyze_form.populate_values(current.analyze_params)

    def refresh_tab_writeback(
        self, tab_id: str, snapshot: Optional["TabViewSnapshot"] = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        tab_w.update_writeback_items(list(current.writeback_items))

    def refresh_tab_save_paths(
        self, tab_id: str, snapshot: Optional["TabViewSnapshot"] = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        save_paths = current.save_paths
        if save_paths is not None:
            tab_w.set_save_paths(save_paths.data_path, save_paths.image_path)

    def refresh_tab_figure(
        self, tab_id: str, snapshot: Optional["TabViewSnapshot"] = None
    ) -> None:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        current = snapshot or self._ctrl.get_tab_snapshot(tab_id)
        figure = current.figure
        if figure is not None:
            self.show_analysis_image(tab_id, figure)

    def refresh_run_lock(self, running_tab_id: Optional[str]) -> None:
        logger.debug("refresh_run_lock: running_tab_id=%r", running_tab_id)
        self._new_tab_btn.setEnabled(running_tab_id is None)
        for tab_id, tab_w in self._tab_widgets.items():
            if self._ctrl.has_tab(tab_id):
                self._set_tab_running(tab_w, self._ctrl.get_tab_snapshot(tab_id))
        if running_tab_id is None:
            for tab_w in self._tab_widgets.values():
                tab_w.progress_stack.reset_all()

    def refresh_tab_interaction(
        self, tab_id: str, snapshot: Optional["TabViewSnapshot"] = None
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
        if self._inspect_dialog is not None and self._inspect_dialog.isVisible():
            self._inspect_dialog.refresh()

    def refresh_predictor_panel(self) -> None:
        info = self._ctrl.get_predictor_info()
        if info is None:
            self._predictor_label.setText("none")
            self._predictor_label.setStyleSheet("")
        else:
            flux_bias = info["flux_bias"]
            self._predictor_label.setText(f"loaded (flux_bias={flux_bias:.4g})")
            self._predictor_label.setStyleSheet("color: green;")

    def make_pbar_factory(self, tab_id: str) -> Any:
        from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return None
        return QtProgressBarFactory(tab_w.progress_stack)

    def make_live_container(self, tab_id: str) -> Any:
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return None
        tab_w.reset_plot()  # clear prior liveplot before new run/analyze
        return tab_w._figure_container

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

    def _resolve_tab_widget(self, tab_id: str, action: str) -> Optional[ExpTabWidget]:
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
        if self._ctrl.get_tab_snapshot(tab_id).interaction.is_running:
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

    def _on_writeback_inline_apply(
        self, tab_id: str, items: list["WritebackItem"]
    ) -> None:
        logger.info("_on_writeback_inline_apply: tab_id=%r", tab_id)
        if not items:
            return
        if not self._ctrl.has_tab(tab_id):
            logger.warning(
                "_on_writeback_inline_apply: unknown tab_id=%r — ignoring", tab_id
            )
            return
        applied_keys = self._ctrl.apply_writeback_items(tab_id, items)
        if applied_keys:
            self.show_status_message(f"Writeback applied: {', '.join(applied_keys)}")

    def _on_save_data_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_data_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_data_clicked")
        if tab_w is None:
            return
        path = tab_w.get_data_path()
        self._ctrl.save_data(tab_id, path)

    def _on_save_image_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_image_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_image_clicked")
        if tab_w is None:
            return
        path = tab_w.get_image_path()
        self._ctrl.save_image(tab_id, path)

    def _on_save_both_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_both_clicked: tab_id=%r", tab_id)
        tab_w = self._resolve_tab_widget(tab_id, "_on_save_both_clicked")
        if tab_w is None:
            return
        data_path = tab_w.get_data_path()
        image_path = tab_w.get_image_path()
        self._ctrl.save_both(tab_id, data_path, image_path)

    def _on_setup_clicked(self) -> None:
        from .setup_dialog import SetupDialog

        dlg = SetupDialog(self._ctrl, parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.open()

    def _on_devices_clicked(self) -> None:
        from .device_dialog import DeviceDialog

        dlg = DeviceDialog(self._ctrl, parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.open()

    def _on_predictor_clicked(self) -> None:
        from .predictor_dialog import PredictorDialog

        dlg = PredictorDialog(self._ctrl, parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.open()

    def _on_inspect_clicked(self) -> None:
        from .inspect_dialog import InspectDialog

        if self._inspect_dialog is None:
            self._inspect_dialog = InspectDialog(
                self._ctrl, bus=self._ctrl.get_bus(), parent=None
            )
        self._inspect_dialog.show()
        self._inspect_dialog.raise_()
        self._inspect_dialog.activateWindow()

    def closeEvent(self, a0: Optional[QCloseEvent]) -> None:
        active_setup = self._ctrl.get_active_device_setup()
        if active_setup is not None:
            if a0 is None:
                return
            if self._shutdown_waiting_for_device_setup:
                a0.ignore()
                return
            from qtpy.QtWidgets import QMessageBox

            answer = QMessageBox.question(
                self,
                "Device setup is running",
                "Cancel the active device setup and close after it stops?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                a0.ignore()
                return
            self._shutdown_waiting_for_device_setup = True
            self._ctrl.cancel_device_operation(active_setup.device_name)
            a0.ignore()
            return
        self._ctrl.persist_tabs_session()
        set_shutting_down(True)
        super().closeEvent(a0)
