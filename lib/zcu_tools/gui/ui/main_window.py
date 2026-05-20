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
    QFileDialog,
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
    QVBoxLayout,
    QWidget,
)

from .cfg_form import CfgFormWidget, _CollapsibleSection

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.adapter import ParamSpec, WritebackItem
    from zcu_tools.gui.controller import Controller


# ---------------------------------------------------------------------------
# Progress bar stack panel (max 4 visible layers, innermost on top)
# ---------------------------------------------------------------------------


class _ProgressStack(QWidget):
    """Compact progress bar panel that only occupies space for active bars.

    Bars are added to the layout on push() and removed on pop()/reset_all(),
    so the widget has zero height when idle and grows only as bars are pushed.
    The anti-jitter strategy: bars are reused from a pool so Qt does not
    repeatedly allocate/free widgets; only the layout insertion/removal happens.
    """

    MAX_LAYERS = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        # Pool of recycled bars (not currently in layout)
        self._pool: list[QProgressBar] = [
            QProgressBar() for _ in range(self.MAX_LAYERS)
        ]
        # Bars currently inserted into the layout, bottom-to-top order
        self._active: list[QProgressBar] = []

    def push(self, label: str = "", total: int = 0) -> QProgressBar:
        if self._pool:
            bar = self._pool.pop()
        else:
            bar = self._active[-1]  # reuse innermost when all slots busy
            return bar
        bar.setFormat(f"{label} %v/%m" if label else "%v/%m")
        bar.setMaximum(total)
        bar.setValue(0)
        # Insert at position 0 so the newest bar appears at the top
        self._layout.insertWidget(0, bar)
        self._active.append(bar)
        return bar

    def pop(self, bar: QProgressBar) -> None:
        if bar in self._active:
            self._active.remove(bar)
            self._layout.removeWidget(bar)
            bar.setParent(None)  # type: ignore[call-overload]
            bar.setValue(0)
            bar.setFormat("%v/%m")
            self._pool.append(bar)

    def reset_all(self) -> None:
        """Remove all active bars (called when a run ends)."""
        for bar in list(self._active):
            self._layout.removeWidget(bar)
            bar.setParent(None)  # type: ignore[call-overload]
            bar.setValue(0)
            bar.setFormat("%v/%m")
            self._pool.append(bar)
        self._active.clear()


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
        self._writeback_rows: dict[str, QWidget] = {}  # wb key → full row widget
        self._applied_writeback_keys: set[str] = set()
        self._writeback_overrides: dict[str, Any] = {}  # key → parsed JSON override
        self._ml: Optional[Any] = None  # ModuleLibrary; set by show_writeback_spec

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(2)

        # --- main content area: [collapse-btn | splitter | collapse-btn] ---
        content_widget = QWidget()
        content_row = QHBoxLayout(content_widget)
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)
        root_layout.addWidget(content_widget, stretch=1)

        # --- progress stack at bottom (zero height when idle) ---
        self.progress_stack = _ProgressStack()
        root_layout.addWidget(self.progress_stack, stretch=0)

        # splitter holds the three panes
        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]

        # left collapse button (collapses/restores config pane)
        _left_collapse_btn = QPushButton("◀")
        _left_collapse_btn.setFixedWidth(16)
        _left_collapse_btn.setToolTip("Collapse/expand config panel")
        _left_collapse_btn.setCheckable(True)
        _left_collapse_btn.setChecked(False)
        content_row.addWidget(_left_collapse_btn)
        content_row.addWidget(splitter, stretch=1)

        # right collapse button (collapses/restores result pane)
        _right_collapse_btn = QPushButton("▶")
        _right_collapse_btn.setFixedWidth(16)
        _right_collapse_btn.setToolTip("Collapse/expand result panel")
        _right_collapse_btn.setCheckable(True)
        _right_collapse_btn.setChecked(False)
        content_row.addWidget(_right_collapse_btn)

        # store default sizes for restore; updated lazily on first collapse
        self._splitter = splitter
        self._left_collapse_btn = _left_collapse_btn
        self._right_collapse_btn = _right_collapse_btn

        def _on_left_collapse(checked: bool) -> None:
            sizes = self._splitter.sizes()
            if checked:
                self._splitter_left_saved = sizes[0]
                sizes[1] += sizes[0]
                sizes[0] = 0
            else:
                saved = getattr(self, "_splitter_left_saved", 250)
                sizes[1] = max(0, sizes[1] - saved)
                sizes[0] = saved
            self._splitter.setSizes(sizes)
            _left_collapse_btn.setText("▶" if checked else "◀")

        def _on_right_collapse(checked: bool) -> None:
            sizes = self._splitter.sizes()
            if checked:
                self._splitter_right_saved = sizes[2]
                sizes[1] += sizes[2]
                sizes[2] = 0
            else:
                saved = getattr(self, "_splitter_right_saved", 300)
                sizes[1] = max(0, sizes[1] - saved)
                sizes[2] = saved
            self._splitter.setSizes(sizes)
            _right_collapse_btn.setText("◀" if checked else "▶")

        _left_collapse_btn.clicked.connect(_on_left_collapse)
        _right_collapse_btn.clicked.connect(_on_right_collapse)

        # ── Config area (left pane) ──────────────────────────────────────
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(2)

        config_layout.addWidget(QLabel("<b>Config</b>"))
        self.cfg_form = CfgFormWidget()
        config_layout.addWidget(self.cfg_form, stretch=1)

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
        self._analyze_section = _CollapsibleSection(
            "Analyze Params", collapsible=True, collapsed=False
        )
        self._analyze_form = self._analyze_section.form
        result_layout.addWidget(self._analyze_section)
        self.analyze_btn = QPushButton("Analyze")
        result_layout.addWidget(self.analyze_btn)

        # Writeback group
        self._writeback_section = _CollapsibleSection(
            "Writeback", collapsible=True, collapsed=False
        )
        self._writeback_layout = QVBoxLayout()
        self._writeback_section.form.addRow(self._writeback_layout)
        self._writeback_section.setVisible(False)
        result_layout.addWidget(self._writeback_section)
        self.apply_writeback_btn = QPushButton("Apply Writeback")
        self.apply_writeback_btn.setVisible(False)
        result_layout.addWidget(self.apply_writeback_btn)

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

        result_layout.addWidget(save_section)
        result_layout.addStretch()

        result_scroll.setWidget(result_inner)
        result_scroll.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding,  # type: ignore[attr-defined]
        )
        splitter.addWidget(result_scroll)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(2, True)
        splitter.setSizes([250, 450, 300])

    # ── cfg helpers ───────────────────────────────────────────────────────

    def populate_cfg(self, schema: Any) -> None:
        self.cfg_form.populate(schema)

    def read_schema(self) -> Any:
        return self.cfg_form.read_schema()

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

    def show_writeback_spec(self, items: list["WritebackItem"], ml: Any = None) -> None:
        """Rebuild the writeback checkbox list."""
        from qtpy.QtWidgets import QWidget, QHBoxLayout  # type: ignore[attr-defined]

        self._ml = ml  # stored for Edit Config dialog

        while self._writeback_layout.count():
            child = self._writeback_layout.takeAt(0)
            w = child.widget() if child is not None else None
            if w is not None:
                w.deleteLater()
        self._writeback_checks.clear()
        self._writeback_rows.clear()
        self._writeback_overrides.clear()
        self._applied_writeback_keys: set[str] = set()

        for item in items:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            # Show Edit Config button only when the Adapter provides a CfgSchema template
            show_edit_btn = item.edit_template is not None

            if show_edit_btn:
                label_text = f"{item.key}  (Config modified)\n  {item.description}"
            else:
                label_text = f"{item.key}  ({item.current_value!r} → {item.new_value!r})\n  {item.description}"

            cb = QCheckBox(label_text)
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_writeback_btn)
            row_layout.addWidget(cb, 1)

            self._writeback_checks[item.key] = cb
            self._writeback_rows[item.key] = row_widget

            if show_edit_btn:
                edit_btn = QPushButton("Edit Config")
                edit_btn.clicked.connect(self._make_edit_cb(item, cb))
                row_layout.addWidget(edit_btn)

            self._writeback_layout.addWidget(row_widget)

        has_items = bool(items)
        self._writeback_section.setVisible(has_items)
        self.apply_writeback_btn.setVisible(has_items)
        self.apply_writeback_btn.setText("Apply Writeback")
        self._refresh_writeback_btn()

    def _make_edit_cb(self, item: "WritebackItem", cb: QCheckBox):
        return lambda: self._on_edit_config_clicked(item, cb)

    def _on_edit_config_clicked(self, item: "WritebackItem", cb: QCheckBox) -> None:
        from qtpy.QtWidgets import (  # type: ignore[attr-defined]
            QDialog,
            QVBoxLayout,
            QHBoxLayout,
            QMessageBox,
            QLabel,
            QScrollArea,
        )
        import copy

        from zcu_tools.gui.adapter import schema_to_dict
        from .cfg_form import CfgFormWidget

        # Determine which schema to show in the form.
        # Priority: previously saved override dict (reconstruct schema by overriding
        # edit_template) > edit_template schema > module_cfg_to_section fallback
        schema = None
        if item.edit_template is not None:
            schema = copy.deepcopy(item.edit_template)

        if schema is None:
            return  # no editable template — button should not be shown

        ml = self._ml

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Config: {item.key}")
        dialog.setMinimumSize(560, 460)

        layout = QVBoxLayout(dialog)
        hint = QLabel("Edit the configuration below. Click Save to confirm.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = CfgFormWidget()
        form_widget.populate(schema, ml=ml)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        cancel_btn.clicked.connect(dialog.reject)

        def save() -> None:
            try:
                updated_schema = form_widget.read_schema()
                parsed = schema_to_dict(updated_schema, ml)
                self._writeback_overrides[item.key] = parsed
                cb.setText(
                    f"{item.key}  (Config modified & edited)\n  {item.description}"
                )
                dialog.accept()
            except Exception as e:
                QMessageBox.critical(
                    dialog, "Validation Error", f"Failed to read config:\n{e}"
                )

        save_btn.clicked.connect(save)
        dialog.exec_()

    def _refresh_writeback_btn(self, *_: int) -> None:
        has_selected = any(cb.isChecked() for cb in self._writeback_checks.values())
        self.apply_writeback_btn.setEnabled(has_selected)

    def mark_writeback_applied(self, applied_keys: list[str]) -> None:
        """Hide entire rows for already-applied keys; lock button when all done."""
        self._applied_writeback_keys.update(applied_keys)
        for key in applied_keys:
            row = self._writeback_rows.get(key)
            if row is not None:
                row.setVisible(False)
        all_applied = all(
            k in self._applied_writeback_keys for k in self._writeback_checks
        )
        if all_applied:
            self.apply_writeback_btn.setEnabled(False)
            self.apply_writeback_btn.setText("Writeback Applied")

    def get_selected_writeback_keys(self) -> list[str]:
        return [k for k, cb in self._writeback_checks.items() if cb.isChecked()]

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
            self._data_path_edit.setText(data_path)
        if image_path:
            self._image_path_edit.setText(image_path)

    def get_data_path(self) -> str:
        return self._data_path_edit.text()

    def get_image_path(self) -> str:
        return self._image_path_edit.text()

    def reset_plot(self) -> None:
        """Remove all canvases from plot_stack, revert to placeholder."""
        while self._plot_stack.count() > 1:
            w = self._plot_stack.widget(self._plot_stack.count() - 1)
            self._plot_stack.removeWidget(w)
            if w is not None:
                w.deleteLater()
        self._canvas_widget = None
        self._plot_stack.setCurrentWidget(self._plot_placeholder)

    def show_analysis_figure(self, fig: "Figure") -> None:
        """Embed a matplotlib Figure in the plot area (replaces any existing analysis canvas)."""
        from matplotlib.backends.backend_qtagg import (  # type: ignore[import-untyped]
            FigureCanvasQTAgg,
        )

        if self._canvas_widget is not None:
            self._plot_stack.removeWidget(self._canvas_widget)
            self._canvas_widget.deleteLater()

        canvas = FigureCanvasQTAgg(fig)
        self._canvas_widget = canvas
        self._plot_stack.addWidget(canvas)
        self._plot_stack.setCurrentWidget(canvas)
        logger.debug("show_analysis_figure: tab_id=%r canvas set", self.tab_id)

    def set_running(
        self, is_running: bool, has_context: bool = True, has_soc: bool = True
    ) -> None:
        can_run = has_context and has_soc and not is_running
        can_act = (
            has_context and not is_running
        )  # analyze/save/writeback need context but not soc
        self.run_btn.setEnabled(can_run)
        self.cancel_btn.setEnabled(is_running)
        self.cfg_form.setEnabled(not is_running)
        self.analyze_btn.setEnabled(can_act)
        self.save_data_btn.setEnabled(can_act)
        self.save_image_btn.setEnabled(can_act)
        self.save_both_btn.setEnabled(can_act)
        self.apply_writeback_btn.setEnabled(can_act)


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

        project_btn = QPushButton("Project…")
        project_btn.clicked.connect(self._on_project_clicked)
        toolbar.addWidget(project_btn)

        connection_btn = QPushButton("Connection…")
        connection_btn.clicked.connect(self._on_connection_clicked)
        toolbar.addWidget(connection_btn)

        devices_btn = QPushButton("Devices…")
        devices_btn.clicked.connect(self._on_devices_clicked)
        toolbar.addWidget(devices_btn)

        predictor_btn = QPushButton("Predictor…")
        predictor_btn.clicked.connect(self._on_predictor_clicked)
        toolbar.addWidget(predictor_btn)

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
        tab_w.show_writeback_spec(spec, ml=self._ctrl.get_current_ml())

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
        has_context = self._ctrl.has_context()
        has_soc = self._ctrl.has_soc()
        self._new_tab_btn.setEnabled(not is_running)
        for tab_w in self._tab_widgets.values():
            tab_w.set_running(is_running, has_context=has_context, has_soc=has_soc)
        if is_running:
            # clear stale plot content before a new run starts
            for tab_w in self._tab_widgets.values():
                tab_w.reset_plot()
        else:
            # clear any leave=True bars that were not popped during the run
            for tab_w in self._tab_widgets.values():
                tab_w.progress_stack.reset_all()

    def refresh_context_panel(self) -> None:
        label = self._ctrl.get_active_context_label()
        has_context = self._ctrl.has_context()
        has_soc = self._ctrl.has_soc()
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
        for tab_w in self._tab_widgets.values():
            tab_w.set_running(
                self._ctrl.is_running(), has_context=has_context, has_soc=has_soc
            )

    def refresh_config_panels(self) -> None:
        for tab_id, tab_w in self._tab_widgets.items():
            schema = self._ctrl.get_tab_default_cfg(tab_id)
            if schema is not None:
                tab_w.populate_cfg(schema)

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
        return tab_w._plot_stack

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

        # populate cfg form from adapter default
        schema = self._ctrl.get_tab_default_cfg(tab_id)
        if schema is not None:
            tab_w.populate_cfg(schema)

        # apply current project / running state
        tab_w.set_running(
            self._ctrl.is_running(),
            has_context=self._ctrl.has_context(),
            has_soc=self._ctrl.has_soc(),
        )

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
        tab_w.save_both_btn.clicked.connect(lambda: self._on_save_both_clicked(tab_id))

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
        logger.info("_on_run_clicked: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        try:
            schema = tab_w.read_schema()
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
        overrides = dict(tab_w._writeback_overrides)
        try:
            self._ctrl.apply_writeback_with_overrides(tab_id, keys, overrides)
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

    def _on_save_both_clicked(self, tab_id: str) -> None:
        logger.info("_on_save_both_clicked: tab_id=%r", tab_id)
        tab_w = self._tab_widgets.get(tab_id)
        if tab_w is None:
            return
        data_path = tab_w.get_data_path()
        image_path = tab_w.get_image_path()
        errors = []
        try:
            self._ctrl.save_data(tab_id, data_path)
        except RuntimeError as exc:
            errors.append(f"Data: {exc}")
        try:
            self._ctrl.save_image(tab_id, image_path)
        except RuntimeError as exc:
            errors.append(f"Image: {exc}")

        if errors:
            msg = " / ".join(errors)
            logger.warning("_on_save_both_clicked: blocked/failed — %s", msg)
            self.show_status_message(msg)
        else:
            self.show_status_message("Data and image saved successfully.")

    def _on_project_clicked(self) -> None:
        from .project_dialog import ProjectDialog

        dlg = ProjectDialog(self._ctrl, parent=self)
        dlg.exec()

    def _on_connection_clicked(self) -> None:
        from .connection_dialog import ConnectionDialog

        dlg = ConnectionDialog(self._ctrl, parent=self)
        dlg.exec()

    def _on_devices_clicked(self) -> None:
        from .device_dialog import DeviceDialog

        dlg = DeviceDialog(self._ctrl, parent=self)
        dlg.exec()

    def _on_predictor_clicked(self) -> None:
        from .predictor_dialog import PredictorDialog

        dlg = PredictorDialog(self._ctrl, parent=self)
        dlg.exec()
