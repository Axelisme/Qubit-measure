from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from .controller import GuiController
from .panels import (
    AnalyzePanel,
    DevicePanel,
    ExpCfgSchemaPanel,
    LibraryPanel,
    MetaPanel,
    RunPanel,
)
from .state import BufferKind

try:
    from zcu_tools.liveplot.backend.pyside6 import clear_plot_host, set_plot_host
except Exception:  # pragma: no cover
    def set_plot_host(*args, **kwargs) -> None:
        return None

    def clear_plot_host() -> None:
        return None

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication,
        QFileSystemModel,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
        QStackedWidget,
        QTabBar,
        QTreeView,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PySide6 is required for GUI mode. Install it first, e.g. `uv add pyside6`."
    ) from exc


class RunWorker(QThread):
    progress = Signal(int, int)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(
        self, controller: GuiController, cancel_event: threading.Event
    ) -> None:
        super().__init__()
        self.controller = controller
        self.cancel_event = cancel_event

    def run(self) -> None:
        try:
            self.controller.run_mock_experiment(
                on_progress=lambda n, t: self.progress.emit(n, t),
                should_cancel=lambda: self.cancel_event.is_set(),
            )
            self.completed.emit({})
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path, backend: str = "mock") -> None:
        super().__init__()
        self.setWindowTitle(f"ZCU GUI (v2_gui) - backend={backend}")
        self.resize(1600, 920)
        self.controller = GuiController(project_root)
        self.controller.bootstrap_groups()
        self.worker: Optional[RunWorker] = None
        self.cancel_event = threading.Event()

        self._build_layout()
        self._wire_events()
        self._load_contexts()
        self._reload_tree()
        self._reload_groups()
        self._reload_right_panels()

    def _build_layout(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        split = QSplitter(Qt.Orientation.Horizontal, self)
        layout.addWidget(split)
        split.addWidget(self._build_left_panel())
        split.addWidget(self._build_center_panel())
        split.addWidget(self._build_right_panel())
        split.setSizes([320, 840, 420])
        run_action = QAction("Run Mock", self)
        run_action.triggered.connect(self._on_run_clicked)
        self.menuBar().addAction(run_action)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("flx_dir files", self))
        self.file_model = QFileSystemModel(self)
        self.tree = QTreeView(self)
        self.tree.setModel(self.file_model)
        layout.addWidget(self.tree)

        layout.addWidget(QLabel("Context labels", self))
        self.context_list = QListWidget(self)
        layout.addWidget(self.context_list)
        row = QHBoxLayout()
        activate_btn = QPushButton("Activate", self)
        new_btn = QPushButton("New", self)
        clone_btn = QPushButton("Clone", self)
        activate_btn.clicked.connect(self._on_activate_context)
        new_btn.clicked.connect(self._on_new_context)
        clone_btn.clicked.connect(self._on_clone_context)
        row.addWidget(activate_btn)
        row.addWidget(new_btn)
        row.addWidget(clone_btn)
        layout.addLayout(row)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        self.group_tabs = QTabBar(self)
        self.group_tabs.setTabsClosable(False)
        layout.addWidget(self.group_tabs)
        nav_row = QHBoxLayout()
        self.prev_buffer_btn = QPushButton("<", self)
        self.next_buffer_btn = QPushButton(">", self)
        self.buffer_title = QLabel("", self)
        nav_row.addWidget(self.prev_buffer_btn)
        nav_row.addWidget(self.buffer_title)
        nav_row.addWidget(self.next_buffer_btn)
        layout.addLayout(nav_row)
        self.center_stack = QStackedWidget(self)
        self.run_panel = RunPanel(self)
        self.analyze_panel = AnalyzePanel(self)
        self.empty_panel = QLabel("No buffer", self)
        self.center_stack.addWidget(self.empty_panel)
        self.center_stack.addWidget(self.run_panel)
        self.center_stack.addWidget(self.analyze_panel)
        layout.addWidget(self.center_stack)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        self.right_tabs = QTabBar(self)
        self.right_tabs.addTab("A: exp_cfg")
        self.right_tabs.addTab("B: device")
        self.right_tabs.addTab("C: metadict")
        self.right_tabs.addTab("D: library")
        layout.addWidget(self.right_tabs)
        self.right_stack = QStackedWidget(self)
        self.cfg_panel = ExpCfgSchemaPanel(self)
        self.device_panel = DevicePanel(self)
        self.meta_panel = MetaPanel(self)
        self.library_panel = LibraryPanel(self)
        self.right_stack.addWidget(self.cfg_panel)
        self.right_stack.addWidget(self.device_panel)
        self.right_stack.addWidget(self.meta_panel)
        self.right_stack.addWidget(self.library_panel)
        layout.addWidget(self.right_stack)
        return panel

    def _wire_events(self) -> None:
        self.tree.doubleClicked.connect(self._on_tree_clicked)
        self.group_tabs.currentChanged.connect(self._on_group_changed)
        self.right_tabs.currentChanged.connect(self.right_stack.setCurrentIndex)
        self.prev_buffer_btn.clicked.connect(lambda: self._step_buffer(-1))
        self.next_buffer_btn.clicked.connect(lambda: self._step_buffer(1))
        self.run_panel.runRequested.connect(self._on_run_clicked)
        self.run_panel.stopRequested.connect(self._on_stop_clicked)
        self.run_panel.analyzeRequested.connect(self._on_analyze_clicked)
        self.analyze_panel.saveFigureRequested.connect(self._on_save_analysis_clicked)
        self.analyze_panel.applyRequested.connect(self._on_apply_analysis_clicked)
        self.cfg_panel.applyRequested.connect(self._on_apply_cfg)
        self.device_panel.applyRequested.connect(self._on_apply_device)
        self.meta_panel.applyRequested.connect(self._on_apply_meta)

    def _load_contexts(self) -> None:
        self.context_list.clear()
        for label in self.controller.list_contexts():
            self.context_list.addItem(label)
        for i in range(self.context_list.count()):
            if self.context_list.item(i).text() == self.controller.exp_manager.label:
                self.context_list.setCurrentRow(i)
                break

    def _reload_tree(self) -> None:
        active_dir = self.controller.active_dir()
        self.file_model.setRootPath(str(active_dir))
        self.tree.setRootIndex(self.file_model.index(str(active_dir)))

    def _reload_groups(self) -> None:
        self.group_tabs.blockSignals(True)
        while self.group_tabs.count() > 0:
            self.group_tabs.removeTab(0)
        for group in self.controller.state.groups.values():
            self.group_tabs.addTab(group.title)
        self.group_tabs.blockSignals(False)
        if self.controller.state.current_group_id is not None:
            keys = list(self.controller.state.groups.keys())
            self.group_tabs.setCurrentIndex(keys.index(self.controller.state.current_group_id))
        self._render_current_buffer()

    def _reload_right_panels(self) -> None:
        self.cfg_panel.set_schema_and_cfg(
            self.controller.get_exp_cfg_schema(),
            dict(self.controller.exp_cfg),
        )
        self.device_panel.set_device_infos(self.controller.get_device_infos())
        self.meta_panel.set_rows(self.controller.get_meta_rows())
        self.library_panel.set_snapshot(self.controller.get_library_snapshot())

    def _render_current_buffer(self) -> None:
        current = self.controller.state.current_buffer()
        if current is None:
            self.center_stack.setCurrentIndex(0)
            self.buffer_title.setText("No buffer")
            return
        self.buffer_title.setText(f"{current.title} [{current.kind.value}]")
        if current.kind == BufferKind.RUN:
            self.center_stack.setCurrentWidget(self.run_panel)
            return
        if current.kind == BufferKind.ANALYZE:
            analysis = dict(current.payload.get("analysis", self.controller.last_analysis or {}))
            self.analyze_panel.set_analysis(analysis)
            self.center_stack.setCurrentWidget(self.analyze_panel)
            return
        self.center_stack.setCurrentIndex(0)

    def _step_buffer(self, step: int) -> None:
        group = self.controller.state.current_group()
        if group is None or not group.buffer_ids:
            return
        group.current_index = (group.current_index + step) % len(group.buffer_ids)
        self._render_current_buffer()

    def _on_group_changed(self, index: int) -> None:
        keys = list(self.controller.state.groups.keys())
        if 0 <= index < len(keys):
            self.controller.state.set_current_group(keys[index])
            self._render_current_buffer()
            self._reload_right_panels()

    def _on_tree_clicked(self, index: Any) -> None:
        path = Path(self.file_model.filePath(index))
        if not path.is_file():
            return
        self.controller.open_file_buffer(path)
        self._reload_groups()

    def _on_activate_context(self) -> None:
        item = self.context_list.currentItem()
        if item is None:
            return
        self.controller.set_context(item.text())
        self._reload_tree()
        self._reload_right_panels()

    def _on_new_context(self) -> None:
        label, ok = QInputDialog.getText(self, "New Flux Context", "Context label:")
        if not ok or not label.strip():
            return
        self.controller.create_context(label.strip())
        self._load_contexts()
        self._reload_tree()
        self._reload_right_panels()

    def _on_clone_context(self) -> None:
        src = self.context_list.currentItem()
        if src is None:
            return
        label, ok = QInputDialog.getText(self, "Clone Flux Context", "Context label:")
        if not ok or not label.strip():
            return
        self.controller.create_context(label.strip(), clone_from=src.text())
        self._load_contexts()
        self._reload_tree()
        self._reload_right_panels()

    def _on_apply_cfg(self, cfg: dict[str, Any]) -> None:
        self.controller.update_exp_cfg(cfg)
        self.statusBar().showMessage("exp_cfg updated", 2000)

    def _on_apply_device(self, device_name: str, updates: dict[str, Any]) -> None:
        for field, value in updates.items():
            result = self.controller.update_device_field(device_name, field, value)
            if not result["ok"]:
                QMessageBox.warning(self, "device", result["message"])
                return
        self.statusBar().showMessage(f"Applied {device_name} changes", 2000)
        self._reload_right_panels()

    def _on_apply_meta(self, updated: dict[str, Any]) -> None:
        self.controller.replace_meta_dict(updated)
        self.controller.save_meta_dict()
        self.statusBar().showMessage("MetaDict updated", 2000)
        self._reload_right_panels()

    def _on_run_clicked(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Run", "A run is already in progress")
            return
        self.cancel_event.clear()
        clear_plot_host()
        set_plot_host(self.run_panel.liveplot_host)
        self.run_panel.reset_progress()
        self.run_panel.set_status("Running")
        self.worker = RunWorker(self.controller, self.cancel_event)
        self.worker.progress.connect(self._on_run_progress)
        self.worker.completed.connect(self._on_run_completed)
        self.worker.failed.connect(self._on_run_failed)
        self.worker.start()

    def _on_stop_clicked(self) -> None:
        self.cancel_event.set()
        self.controller.request_stop_active_run()
        clear_plot_host()
        self.run_panel.reset_progress()
        self.run_panel.set_status("Stopping...")

    def _on_run_progress(self, current: int, total: int) -> None:
        self.run_panel.set_progress(current, total)
        self.run_panel.set_status(f"Running {current}/{max(total, 1)}")

    def _on_run_completed(self, payload: dict[str, Any]) -> None:
        del payload
        clear_plot_host()
        self.run_panel.set_progress(1, 1)
        self.run_panel.set_status("Run completed")

    def _on_run_failed(self, message: str) -> None:
        clear_plot_host()
        self.run_panel.reset_progress()
        self.run_panel.set_status("Run failed")
        QMessageBox.warning(self, "Run failed", message)

    def _on_analyze_clicked(self) -> None:
        try:
            result = self.controller.analyze_last_result()
        except Exception as exc:
            QMessageBox.warning(self, "Analyze", str(exc))
            return
        self.statusBar().showMessage(f"Analyze done: {result}", 2000)
        self._reload_groups()
        self._render_current_buffer()

    def _on_save_analysis_clicked(self) -> None:
        self.controller.save_analysis_figure()
        self._render_current_buffer()

    def _on_apply_analysis_clicked(self) -> None:
        if not self.controller.last_analysis:
            return
        self.controller.apply_analysis_to_context()
        self._reload_right_panels()


def run_app(project_root: Optional[Path] = None, backend: str = "mock") -> None:
    if backend != "mock":
        raise ValueError(f"Unsupported backend in fake-only mode: {backend}")
    root = project_root or Path.cwd()
    app = QApplication.instance() or QApplication([])
    win = MainWindow(root, backend=backend)
    win.show()
    app.exec()
