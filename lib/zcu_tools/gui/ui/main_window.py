"""MainWindow — the top-level View for the v2_gui framework.

Implements ViewProtocol; all state lives in Controller/State, never here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.adapter import ParamSpec, WritebackItem
    from zcu_tools.gui.controller import Controller


# ---------------------------------------------------------------------------
# Progress bar stack panel (max 4 visible layers, innermost on top)
# ---------------------------------------------------------------------------


class _ProgressStack(QWidget):
    """Displays up to MAX_LAYERS progress bars stacked innermost-on-top."""

    MAX_LAYERS = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self._bars: list[QProgressBar] = []

    def push(self, label: str = "", total: int = 0) -> QProgressBar:
        bar = QProgressBar()
        bar.setFormat(f"{label} %v/%m" if label else "%v/%m")
        bar.setMaximum(total)
        bar.setValue(0)
        self._bars.append(bar)
        self._refresh_visibility()
        self._layout.insertWidget(0, bar)  # newest on top
        return bar

    def pop(self, bar: QProgressBar) -> None:
        if bar in self._bars:
            self._bars.remove(bar)
            self._layout.removeWidget(bar)
            bar.deleteLater()
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        for i, bar in enumerate(reversed(self._bars)):
            bar.setVisible(i < self.MAX_LAYERS)


# ---------------------------------------------------------------------------
# Helpers: dynamic param widgets
# ---------------------------------------------------------------------------


def _make_param_widget(spec: "ParamSpec") -> QWidget:
    """Build an input widget from a ParamSpec."""
    if spec.choices:
        w = QComboBox()
        w.addItems([str(c) for c in spec.choices])
        default_str = str(spec.default)
        idx = w.findText(default_str)
        if idx >= 0:
            w.setCurrentIndex(idx)
        return w
    if spec.type is bool:
        w = QCheckBox()
        w.setChecked(bool(spec.default))
        return w
    if spec.type is int:
        w = QSpinBox()
        w.setRange(-(2**31), 2**31 - 1)
        w.setValue(int(spec.default))
        return w
    if spec.type is float:
        w = QDoubleSpinBox()
        w.setRange(-1e12, 1e12)
        w.setDecimals(6)
        w.setValue(float(spec.default))
        return w
    # fallback: text
    w = QLineEdit(str(spec.default))
    return w


def _read_param_widget(w: QWidget, spec: "ParamSpec") -> Any:
    """Read current value from a widget created by _make_param_widget."""
    if isinstance(w, QComboBox):
        txt = w.currentText()
        return spec.type(txt) if spec.type not in (str,) else txt
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    if isinstance(w, QDoubleSpinBox):
        return w.value()
    if isinstance(w, QLineEdit):
        return spec.type(w.text())
    return None


# ---------------------------------------------------------------------------
# Per-experiment tab widget
# ---------------------------------------------------------------------------


class ExpTabWidget(QWidget):
    """A single experiment tab: Config | Plot | Result areas."""

    def __init__(self, tab_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.tab_id = tab_id
        self._param_widgets: dict[str, QWidget] = {}  # analyze param key → widget
        self._writeback_checks: dict[str, QCheckBox] = {}  # wb key → checkbox
        self._applied_writeback_keys: set[str] = set()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)

        # --- progress stack at top ---
        self.progress_stack = _ProgressStack()
        root_layout.addWidget(self.progress_stack)

        # --- main content splitter ---
        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        root_layout.addWidget(splitter, stretch=1)

        # ── Config area (left pane) ──────────────────────────────────────
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.addWidget(QLabel("<b>Config</b>"))
        self.cfg_editor = QTextEdit()
        self.cfg_editor.setPlaceholderText("(cfg schema shown here)")
        config_layout.addWidget(self.cfg_editor)

        run_btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        run_btn_row.addWidget(self.run_btn)
        run_btn_row.addWidget(self.cancel_btn)
        config_layout.addLayout(run_btn_row)
        splitter.addWidget(config_panel)

        # ── Plot area (centre pane) ──────────────────────────────────────
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._plot_stack = QStackedWidget()

        # page 0: placeholder label
        self._plot_placeholder = QLabel("(no plot yet)")
        self._plot_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._plot_stack.addWidget(self._plot_placeholder)  # index 0

        # page 1: matplotlib canvas (inserted on first show_analysis_image call)
        self._canvas_widget: Optional[QWidget] = None

        plot_layout.addWidget(self._plot_stack, stretch=1)
        splitter.addWidget(plot_panel)

        # ── Result area (right pane) ─────────────────────────────────────
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_inner = QWidget()
        result_layout = QVBoxLayout(result_inner)
        result_layout.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]

        # Analyze params group
        self._analyze_group = QGroupBox("Analyze Params")
        self._analyze_form = QFormLayout(self._analyze_group)
        result_layout.addWidget(self._analyze_group)
        self.analyze_btn = QPushButton("Analyze")
        result_layout.addWidget(self.analyze_btn)

        # Writeback group
        self._writeback_group = QGroupBox("Writeback")
        self._writeback_layout = QVBoxLayout(self._writeback_group)
        self._writeback_group.setVisible(False)
        result_layout.addWidget(self._writeback_group)
        self.apply_writeback_btn = QPushButton("Apply Writeback")
        self.apply_writeback_btn.setVisible(False)
        result_layout.addWidget(self.apply_writeback_btn)

        # Save group
        save_group = QGroupBox("Save")
        save_layout = QFormLayout(save_group)

        self._data_path_edit = QLineEdit()
        self._data_path_edit.setPlaceholderText("/tmp/data")
        save_layout.addRow("Data path:", self._data_path_edit)
        self.save_data_btn = QPushButton("Save Data")
        save_layout.addRow("", self.save_data_btn)

        self._image_path_edit = QLineEdit()
        self._image_path_edit.setPlaceholderText("/tmp/image.png")
        save_layout.addRow("Image path:", self._image_path_edit)
        self.save_image_btn = QPushButton("Save Image")
        save_layout.addRow("", self.save_image_btn)

        result_layout.addWidget(save_group)
        result_layout.addStretch()

        result_scroll.setWidget(result_inner)
        result_scroll.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding,  # type: ignore[attr-defined]
        )
        splitter.addWidget(result_scroll)
        splitter.setSizes([250, 450, 300])

    # ── populate / refresh helpers ────────────────────────────────────────

    def populate_analyze_params(self, param_specs: dict[str, "ParamSpec"]) -> None:
        """Rebuild the Analyze Params form from the adapter's ParamSpec dict."""
        # clear old widgets
        while self._analyze_form.rowCount():
            self._analyze_form.removeRow(0)
        self._param_widgets.clear()

        for key, spec in param_specs.items():
            w = _make_param_widget(spec)
            self._analyze_form.addRow(spec.label + ":", w)
            self._param_widgets[key] = w

    def show_writeback_spec(self, items: list["WritebackItem"]) -> None:
        """Rebuild the writeback checkbox list."""
        while self._writeback_layout.count():
            child = self._writeback_layout.takeAt(0)
            w = child.widget() if child is not None else None
            if w is not None:
                w.deleteLater()
        self._writeback_checks.clear()
        self._applied_writeback_keys: set[str] = set()

        for item in items:
            cb = QCheckBox(
                f"{item.key}  ({item.current_value!r} → {item.new_value!r})\n"
                f"  {item.description}"
            )
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_writeback_btn)
            self._writeback_layout.addWidget(cb)
            self._writeback_checks[item.key] = cb

        has_items = bool(items)
        self._writeback_group.setVisible(has_items)
        self.apply_writeback_btn.setVisible(has_items)
        self.apply_writeback_btn.setText("Apply Writeback")
        self._refresh_writeback_btn()

    def _refresh_writeback_btn(self, *_: int) -> None:
        has_selected = any(cb.isChecked() for cb in self._writeback_checks.values())
        self.apply_writeback_btn.setEnabled(has_selected)

    def mark_writeback_applied(self, applied_keys: list[str]) -> None:
        """Hide checkboxes for already-applied keys; lock button when all done."""
        self._applied_writeback_keys.update(applied_keys)
        for key in applied_keys:
            cb = self._writeback_checks.get(key)
            if cb is not None:
                cb.setVisible(False)
        all_applied = all(
            k in self._applied_writeback_keys for k in self._writeback_checks
        )
        if all_applied:
            self.apply_writeback_btn.setEnabled(False)
            self.apply_writeback_btn.setText("Writeback Applied")

    def get_selected_writeback_keys(self) -> list[str]:
        return [k for k, cb in self._writeback_checks.items() if cb.isChecked()]

    def set_save_paths(self, data_path: str, image_path: str) -> None:
        if data_path:
            self._data_path_edit.setText(data_path)
        if image_path:
            self._image_path_edit.setText(image_path)

    def get_data_path(self) -> str:
        return self._data_path_edit.text()

    def get_image_path(self) -> str:
        return self._image_path_edit.text()

    def show_analysis_figure(self, fig: "Figure") -> None:
        """Embed a matplotlib Figure in the plot area (replaces placeholder)."""
        from matplotlib.backends.backend_qtagg import (  # type: ignore[import-untyped]
            FigureCanvasQTAgg,
        )

        if self._canvas_widget is not None:
            self._plot_stack.removeWidget(self._canvas_widget)
            self._canvas_widget.deleteLater()

        canvas = FigureCanvasQTAgg(fig)
        self._canvas_widget = canvas
        self._plot_stack.addWidget(canvas)  # index 1 (or replaces old)
        self._plot_stack.setCurrentWidget(canvas)
        logger.debug("show_analysis_figure: tab_id=%r canvas set", self.tab_id)

    def set_running(self, is_running: bool) -> None:
        self.run_btn.setEnabled(not is_running)
        self.cancel_btn.setEnabled(is_running)
        self.analyze_btn.setEnabled(not is_running)
        self.save_data_btn.setEnabled(not is_running)
        self.save_image_btn.setEnabled(not is_running)
        self.apply_writeback_btn.setEnabled(not is_running)


# ---------------------------------------------------------------------------
# MainWindow — implements ViewProtocol
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Top-level window; implements ViewProtocol for Controller callbacks."""

    def __init__(self, controller: "Controller") -> None:
        super().__init__()
        self._ctrl = controller
        self._tab_widgets: dict[str, ExpTabWidget] = {}

        self.setWindowTitle("ZCU Qubit Measure — v2 GUI")
        self.resize(1280, 750)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- toolbar ---
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Experiment:"))
        self._adapter_combo = QComboBox()
        self._adapter_combo.addItems(controller.get_adapter_names())
        toolbar.addWidget(self._adapter_combo)
        self._new_tab_btn = QPushButton("New Tab")
        self._new_tab_btn.clicked.connect(self._on_new_tab_requested)
        toolbar.addWidget(self._new_tab_btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # --- tab widget ---
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        main_layout.addWidget(self._tabs, stretch=1)

        # --- status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    # ------------------------------------------------------------------
    # ViewProtocol implementation
    # ------------------------------------------------------------------

    def refresh_tab(self, tab_id: str) -> None:
        logger.debug("refresh_tab: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return

        # populate analyze params on first result (adapter is now known)
        if not tab_w._param_widgets:
            params = self._ctrl.get_tab_analyze_params(tab_id)
            tab_w.populate_analyze_params(params)
            tab_w._analyze_specs = params  # store for value-read

        # update writeback list (may have new analyze result)
        spec = self._ctrl.get_tab_writeback_spec(tab_id)
        tab_w.show_writeback_spec(spec)

        # populate save paths
        try:
            save_paths = self._ctrl.get_tab_save_paths(tab_id)
            tab_w.set_save_paths(save_paths.data_path, save_paths.image_path)
        except Exception:
            pass

        # show analysis figure if available
        figure = self._ctrl.get_tab_figure(tab_id)
        if figure is not None:
            self.show_analysis_image(tab_id, figure)

    def refresh_run_state(self, is_running: bool) -> None:
        logger.debug("refresh_run_state: is_running=%s", is_running)
        self._new_tab_btn.setEnabled(not is_running)
        for tab_w in self._tab_widgets.values():
            tab_w.set_running(is_running)

    def refresh_context_panel(self) -> None:
        pass  # Phase 10

    def refresh_config_panels(self) -> None:
        pass  # Phase 10

    def show_status_message(self, message: str) -> None:
        logger.info("status: %s", message)
        self._status_bar.showMessage(message)

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
        adapter_name = self._adapter_combo.currentText()
        if not adapter_name:
            return
        logger.info("_on_new_tab_requested: adapter=%r", adapter_name)
        tab_id = self._ctrl.new_tab(adapter_name)
        tab_w = ExpTabWidget(tab_id)
        self._tab_widgets[tab_id] = tab_w
        self._tabs.addTab(tab_w, adapter_name)
        self._tabs.setCurrentWidget(tab_w)

        # wire buttons
        tab_w.run_btn.clicked.connect(lambda: self._on_run_clicked(tab_id))
        tab_w.cancel_btn.clicked.connect(self._on_cancel_clicked)
        tab_w.analyze_btn.clicked.connect(lambda: self._on_analyze_clicked(tab_id))
        tab_w.apply_writeback_btn.clicked.connect(
            lambda: self._on_apply_writeback_clicked(tab_id)
        )
        tab_w.save_data_btn.clicked.connect(lambda: self._on_save_data_clicked(tab_id))
        tab_w.save_image_btn.clicked.connect(
            lambda: self._on_save_image_clicked(tab_id)
        )

    def _on_tab_close_requested(self, index: int) -> None:
        tab_w = self._tabs.widget(index)
        if not isinstance(tab_w, ExpTabWidget):
            return
        tab_id = tab_w.tab_id
        logger.info("_on_tab_close_requested: tab_id=%r", tab_id)
        self._ctrl.close_tab(tab_id)
        self._tab_widgets.pop(tab_id, None)
        self._tabs.removeTab(index)

    def _on_run_clicked(self, tab_id: str) -> None:
        from zcu_tools.gui.adapter import CfgSchema, CfgSection

        logger.info("_on_run_clicked: tab_id=%r", tab_id)
        schema = CfgSchema(root=CfgSection(fields={}))
        try:
            self._ctrl.start_run(tab_id, schema, {})
        except RuntimeError as exc:
            logger.warning("_on_run_clicked: blocked — %s", exc)
            self.show_status_message(str(exc))

    def _on_cancel_clicked(self) -> None:
        logger.info("_on_cancel_clicked")
        self._ctrl.cancel_run()

    def _on_analyze_clicked(self, tab_id: str) -> None:
        logger.info("_on_analyze_clicked: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        # collect current param values
        user_params: dict[str, Any] = {}
        specs: dict = getattr(tab_w, "_analyze_specs", {})
        for key, w in tab_w._param_widgets.items():
            spec = specs.get(key)
            if spec is not None:
                user_params[key] = _read_param_widget(w, spec)
        try:
            self._ctrl.analyze(tab_id, user_params)
        except RuntimeError as exc:
            logger.warning("_on_analyze_clicked: blocked — %s", exc)
            self.show_status_message(str(exc))

    def _on_apply_writeback_clicked(self, tab_id: str) -> None:
        logger.info("_on_apply_writeback_clicked: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        keys = tab_w.get_selected_writeback_keys()
        try:
            self._ctrl.apply_writeback(tab_id, keys)
            tab_w.mark_writeback_applied(keys)
            self.show_status_message(f"Writeback applied: {', '.join(keys)}")
        except RuntimeError as exc:
            logger.warning("_on_apply_writeback_clicked: blocked — %s", exc)
            self.show_status_message(str(exc))

    def _on_save_data_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_data_clicked: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        path = tab_w.get_data_path()
        try:
            self._ctrl.save_data(tab_id, path)
        except RuntimeError as exc:
            logger.warning("_on_save_data_clicked: blocked — %s", exc)
            self.show_status_message(str(exc))

    def _on_save_image_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_image_clicked: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        path = tab_w.get_image_path()
        try:
            self._ctrl.save_image(tab_id, path)
        except RuntimeError as exc:
            logger.warning("_on_save_image_clicked: blocked — %s", exc)
            self.show_status_message(str(exc))

    def _on_context_selected(self, label: str) -> None:
        _ = label  # Phase 10
