"""MainWindow — the top-level View for the v2_gui framework.

Implements ViewProtocol; all state lives in Controller/State, never here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
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
        # show only the MAX_LAYERS innermost (last-pushed) bars
        for i, bar in enumerate(reversed(self._bars)):
            bar.setVisible(i < self.MAX_LAYERS)


# ---------------------------------------------------------------------------
# Per-experiment tab widget
# ---------------------------------------------------------------------------


class ExpTabWidget(QWidget):
    """A single experiment tab: Config | Plot | Result areas."""

    def __init__(self, tab_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.tab_id = tab_id

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)

        # --- progress stack at top ---
        self.progress_stack = _ProgressStack()
        root_layout.addWidget(self.progress_stack)

        # --- main content splitter ---
        splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        root_layout.addWidget(splitter, stretch=1)

        # Config area (left pane)
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.addWidget(QLabel("Config"))
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

        # Plot area (centre pane) — placeholder for now
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.addWidget(QLabel("Plot"))
        self.plot_placeholder = QLabel("(no plot yet)")
        self.plot_placeholder.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        plot_layout.addWidget(self.plot_placeholder, stretch=1)
        splitter.addWidget(plot_panel)

        # Result area (right pane)
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.addWidget(QLabel("Result"))
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setPlaceholderText("(results shown here after run)")

        analyze_btn_row = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze")
        analyze_btn_row.addWidget(self.analyze_btn)

        result_layout.addWidget(self.result_display, stretch=1)
        result_layout.addLayout(analyze_btn_row)
        splitter.addWidget(result_panel)

        splitter.setSizes([250, 400, 300])

    def show_result(self, result: Any) -> None:
        self.result_display.setPlainText(str(result))

    def set_running(self, is_running: bool) -> None:
        self.run_btn.setEnabled(not is_running)
        self.cancel_btn.setEnabled(is_running)
        self.analyze_btn.setEnabled(not is_running)


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
        self.resize(1200, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- toolbar row: adapter selector + New Tab button ---
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
        result = self._ctrl.get_tab_result(tab_id)
        if result is not None:
            tab_w.show_result(result)

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

    def show_analysis_image(self, tab_id: str, fig: Any) -> None:  # Phase 9
        logger.debug(
            "show_analysis_image: tab_id=%r fig=%s", tab_id, type(fig).__name__
        )

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

        # wire per-tab buttons
        tab_w.run_btn.clicked.connect(lambda: self._on_run_clicked(tab_id))
        tab_w.cancel_btn.clicked.connect(self._on_cancel_clicked)
        tab_w.analyze_btn.clicked.connect(lambda: self._on_analyze_clicked(tab_id))

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
        # Phase 8: pass an empty schema as placeholder; real cfg editor in Phase 9
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
        logger.info("_on_analyze_clicked: tab_id=%r (not yet implemented)", tab_id)
        self.show_status_message("Analyze not yet implemented (Phase 9)")

    def _on_apply_writeback_clicked(self) -> None:
        pass  # Phase 9

    def _on_save_data_clicked(self) -> None:
        pass  # Phase 9

    def _on_save_image_clicked(self) -> None:
        pass  # Phase 9

    def _on_context_selected(self, label: str) -> None:
        _ = label  # Phase 10
