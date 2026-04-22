from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any, Optional

from .controller import GuiController
from .state import BufferDescriptor, BufferKind

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QAction, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileSystemModel,
        QFormLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QTabBar,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
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
            result = self.controller.run_mock_experiment(
                on_progress=lambda n, t: self.progress.emit(n, t),
                should_cancel=lambda: self.cancel_event.is_set(),
            )
            self.completed.emit({"points": len(result.x), "partial": result.partial})
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path, backend: str = "mock") -> None:
        super().__init__()
        self.setWindowTitle(f"ZCU GUI (v2_gui) - backend={backend}")
        self.resize(1600, 920)

        self.controller = GuiController(project_root, backend=backend)
        self.controller.bootstrap_groups()

        self.worker: Optional[RunWorker] = None
        self.cancel_event = threading.Event()

        self._init_ui()
        self._load_contexts()
        self._reload_tree()
        self._reload_groups()
        self._reload_right_panels()

    def _init_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        main_split = QSplitter(Qt.Horizontal, self)
        layout.addWidget(main_split)

        left = self._build_left_panel()
        center = self._build_center_panel()
        right = self._build_right_panel()

        main_split.addWidget(left)
        main_split.addWidget(center)
        main_split.addWidget(right)
        main_split.setSizes([320, 820, 460])

        run_action = QAction("Run Mock", self)
        run_action.triggered.connect(self._on_run_clicked)
        self.menuBar().addAction(run_action)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        split = QSplitter(Qt.Vertical, self)

        top = QWidget(self)
        top_layout = QVBoxLayout(top)
        top_layout.addWidget(QLabel("flx_dir files"))
        self.file_model = QFileSystemModel(self)
        self.tree = QTreeView(self)
        self.tree.clicked.connect(self._on_tree_clicked)
        self.tree.setModel(self.file_model)
        self.tree.setHeaderHidden(False)
        top_layout.addWidget(self.tree)

        bottom = QWidget(self)
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.addWidget(QLabel("Context labels"))
        self.context_list = QListWidget(self)
        self.context_list.itemSelectionChanged.connect(self._on_context_changed)
        bottom_layout.addWidget(self.context_list)

        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([520, 240])

        layout.addWidget(split)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        self.group_tabs = QTabBar(self)
        self.group_tabs.currentChanged.connect(self._on_group_changed)
        layout.addWidget(self.group_tabs)

        nav = QWidget(self)
        nav_layout = QHBoxLayout(nav)
        self.prev_btn = QPushButton("<", self)
        self.next_btn = QPushButton(">", self)
        self.prev_btn.clicked.connect(lambda: self._step_buffer(-1))
        self.next_btn.clicked.connect(lambda: self._step_buffer(1))
        self.buffer_title = QLabel("", self)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.buffer_title)
        nav_layout.addWidget(self.next_btn)
        layout.addWidget(nav)

        self.buffer_holder = QVBoxLayout()
        layout.addLayout(self.buffer_holder)

        bottom = QWidget(self)
        bottom_layout = QHBoxLayout(bottom)
        self.progress = QProgressBar(self)
        self.progress_label = QLabel("Idle", self)
        bottom_layout.addWidget(self.progress)
        bottom_layout.addWidget(self.progress_label)
        layout.addWidget(bottom)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        self.right_tabs = QTabBar(self)
        self.right_tabs.addTab("A: exp_cfg")
        self.right_tabs.addTab("B: device")
        self.right_tabs.addTab("C: metadict")
        self.right_tabs.addTab("D: library")
        self.right_tabs.currentChanged.connect(self._on_right_tab_changed)
        layout.addWidget(self.right_tabs)

        self.right_holder = QVBoxLayout()
        layout.addLayout(self.right_holder)
        return panel

    def _load_contexts(self) -> None:
        self.context_list.clear()
        for label in self.controller.list_contexts():
            self.context_list.addItem(label)
        self.context_list.setCurrentRow(0)

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
            idx = keys.index(self.controller.state.current_group_id)
            self.group_tabs.setCurrentIndex(idx)
        self._render_current_buffer()

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_current_buffer(self) -> None:
        self._clear_layout(self.buffer_holder)
        buffer = self.controller.state.current_buffer()
        if buffer is None:
            self.buffer_holder.addWidget(QLabel("No buffer"))
            return

        self.buffer_title.setText(f"{buffer.title} [{buffer.kind.value}]")

        if buffer.kind == BufferKind.RUN:
            self.buffer_holder.addWidget(self._make_run_buffer_widget())
        elif buffer.kind == BufferKind.ANALYZE:
            self.buffer_holder.addWidget(self._make_analyze_buffer_widget(buffer))
        elif buffer.kind == BufferKind.COMMENT:
            self.buffer_holder.addWidget(self._make_comment_buffer_widget(buffer))
        elif buffer.kind == BufferKind.FILE_IMAGE:
            self.buffer_holder.addWidget(self._make_image_view(buffer))
        elif buffer.kind == BufferKind.FILE_CSV:
            self.buffer_holder.addWidget(self._make_csv_view(buffer))
        else:
            self.buffer_holder.addWidget(self._make_text_view(buffer))

    def _make_run_buffer_widget(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)

        form = QWidget(self)
        form_lay = QFormLayout(form)
        self.points_edit = QLineEdit(
            str(self.controller.exp_cfg.get("sweep_points", 101)), self
        )
        self.delay_edit = QLineEdit(
            str(self.controller.exp_cfg.get("step_delay_s", 0.02)), self
        )
        self.center_edit = QLineEdit(
            str(self.controller.exp_cfg.get("center_mhz", 6812.3)), self
        )
        self.width_edit = QLineEdit(
            str(self.controller.exp_cfg.get("width_mhz", 10.0)), self
        )
        form_lay.addRow("points", self.points_edit)
        form_lay.addRow("step_delay_s", self.delay_edit)
        form_lay.addRow("center_mhz", self.center_edit)
        form_lay.addRow("width_mhz", self.width_edit)
        lay.addWidget(form)

        btns = QWidget(self)
        btn_lay = QHBoxLayout(btns)
        run_btn = QPushButton("Run", self)
        stop_btn = QPushButton("Stop", self)
        analyze_btn = QPushButton("Analyze", self)
        save_btn = QPushButton("Save", self)
        run_btn.clicked.connect(self._on_run_clicked)
        stop_btn.clicked.connect(self._on_stop_clicked)
        analyze_btn.clicked.connect(self._on_analyze_clicked)
        save_btn.clicked.connect(self._on_save_run_clicked)
        btn_lay.addWidget(run_btn)
        btn_lay.addWidget(stop_btn)
        btn_lay.addWidget(analyze_btn)
        btn_lay.addWidget(save_btn)
        lay.addWidget(btns)

        self.run_text = QTextEdit(self)
        self.run_text.setReadOnly(True)
        self.run_text.setPlaceholderText("Run logs...")
        lay.addWidget(self.run_text)

        return w

    def _make_analyze_buffer_widget(self, buffer: BufferDescriptor) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)

        analysis = dict(
            buffer.payload.get("analysis", self.controller.last_analysis or {})
        )
        summary = QTextEdit(self)
        summary.setReadOnly(True)
        summary.setText(json.dumps(analysis, indent=2))
        lay.addWidget(summary)

        figure_path_text = analysis.get("figure_path")
        image_label = QLabel(self)
        if isinstance(figure_path_text, str) and figure_path_text:
            figure_path = Path(figure_path_text)
            if figure_path.exists():
                pix = QPixmap(str(figure_path))
                if pix.isNull():
                    image_label.setText(f"Failed to load analysis image: {figure_path}")
                else:
                    image_label.setPixmap(
                        pix.scaledToWidth(760, Qt.SmoothTransformation)
                    )
            else:
                image_label.setText(f"Analysis image not found: {figure_path}")
        else:
            image_label.setText("No analysis image yet")
        lay.addWidget(image_label)

        btns = QWidget(self)
        btn_lay = QHBoxLayout(btns)
        save_btn = QPushButton("Save Figure", self)
        overwrite_meta_btn = QPushButton("Overwrite Meta", self)
        save_btn.clicked.connect(self._on_save_analysis_clicked)
        overwrite_meta_btn.clicked.connect(self._on_apply_analysis_clicked)
        btn_lay.addWidget(save_btn)
        btn_lay.addWidget(overwrite_meta_btn)
        lay.addWidget(btns)

        return w

    def _make_comment_buffer_widget(self, buffer: BufferDescriptor) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        editor = QPlainTextEdit(self)
        editor.setPlainText(str(buffer.payload.get("text", "")))

        def _save_comment() -> None:
            buffer.payload["text"] = editor.toPlainText()
            self.statusBar().showMessage("Comment saved to buffer payload", 2000)

        save_btn = QPushButton("Save Comment", self)
        save_btn.clicked.connect(_save_comment)

        lay.addWidget(editor)
        lay.addWidget(save_btn)
        return w

    def _make_image_view(self, buffer: BufferDescriptor) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        path_text = str(buffer.payload.get("path", "")).strip()
        if not path_text:
            lay.addWidget(QLabel("No artifact image path"))
            return w
        path = Path(path_text)
        label = QLabel(self)
        if not path.exists():
            label.setText(f"Image not found: {path}")
            lay.addWidget(label)
            return w
        pix = QPixmap(str(path))
        if pix.isNull():
            label.setText(f"Failed to load image: {path}")
        else:
            label.setPixmap(pix.scaledToWidth(760, Qt.SmoothTransformation))
        lay.addWidget(label)
        return w

    def _make_text_view(self, buffer: BufferDescriptor) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        path_text = str(buffer.payload.get("path", "")).strip()
        if not path_text:
            lay.addWidget(QLabel("No artifact text path"))
            return w
        path = Path(path_text)
        editor = QPlainTextEdit(self)
        if not path.exists():
            lay.addWidget(QLabel(f"Text file not found: {path}"))
            return w
        editor.setPlainText(path.read_text(encoding="utf-8"))

        def _save_file() -> None:
            path.write_text(editor.toPlainText(), encoding="utf-8")
            self.statusBar().showMessage(f"Saved {path.name}", 2000)

        save_btn = QPushButton("Save File", self)
        save_btn.clicked.connect(_save_file)
        lay.addWidget(editor)
        lay.addWidget(save_btn)
        return w

    def _make_csv_view(self, buffer: BufferDescriptor) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        path_text = str(buffer.payload.get("path", "")).strip()
        if not path_text:
            lay.addWidget(QLabel("No artifact csv path"))
            return w
        path = Path(path_text)
        if not path.exists():
            lay.addWidget(QLabel(f"CSV file not found: {path}"))
            return w
        table = QTableWidget(self)

        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)

        if rows:
            table.setColumnCount(len(rows[0]))
            table.setRowCount(len(rows) - 1)
            table.setHorizontalHeaderLabels(rows[0])
            for r, row in enumerate(rows[1:]):
                for c, value in enumerate(row):
                    table.setItem(r, c, QTableWidgetItem(value))

        def _save_csv() -> None:
            headers = [
                table.horizontalHeaderItem(c).text() for c in range(table.columnCount())
            ]
            out_rows = [headers]
            for r in range(table.rowCount()):
                out_row = []
                for c in range(table.columnCount()):
                    item = table.item(r, c)
                    out_row.append(item.text() if item else "")
                out_rows.append(out_row)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(out_rows)
            self.statusBar().showMessage(f"Saved {path.name}", 2000)

        save_btn = QPushButton("Save CSV", self)
        save_btn.clicked.connect(_save_csv)

        lay.addWidget(table)
        lay.addWidget(save_btn)
        return w

    def _reload_right_panels(self) -> None:
        self._on_right_tab_changed(self.right_tabs.currentIndex())

    def _on_right_tab_changed(self, index: int) -> None:
        self._clear_layout(self.right_holder)
        if index == 0:
            self.right_holder.addWidget(self._build_panel_a())
        elif index == 1:
            self.right_holder.addWidget(self._build_panel_b())
        elif index == 2:
            self.right_holder.addWidget(self._build_panel_c())
        else:
            self.right_holder.addWidget(self._build_panel_d())

    def _build_panel_a(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        editor = QPlainTextEdit(self)
        editor.setPlainText(self.controller.get_exp_cfg_text())

        def _save() -> None:
            try:
                self.controller.update_exp_cfg_from_text(editor.toPlainText())
                self.statusBar().showMessage("exp_cfg updated", 2000)
            except Exception as exc:
                QMessageBox.warning(self, "A panel", str(exc))

        save_btn = QPushButton("Apply exp_cfg", self)
        save_btn.clicked.connect(_save)
        lay.addWidget(editor)
        lay.addWidget(save_btn)
        return w

    def _build_panel_b(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)

        table = QTableWidget(self)
        rows = self.controller.get_device_rows()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["device", "field", "value"])
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(str(row["device"])))
            table.setItem(r, 1, QTableWidgetItem(str(row["field"])))
            table.setItem(r, 2, QTableWidgetItem(str(row["value"])))

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        def _apply_selected() -> None:
            r = table.currentRow()
            if r < 0:
                return
            device = table.item(r, 0).text()
            field = table.item(r, 1).text()
            text_value = table.item(r, 2).text()
            value: Any = text_value
            try:
                value = float(text_value)
            except ValueError:
                pass

            result = self.controller.update_device_field(device, field, value)
            if result["ok"]:
                table.item(r, 2).setText(str(result["value"]))
                self.statusBar().showMessage("Device field applied", 2000)
            else:
                table.item(r, 2).setText(str(result["value"]))
                QMessageBox.warning(
                    self,
                    "B panel rollback",
                    f"Failed: {result['message']}\nOnly this field has been rolled back.",
                )

        apply_btn = QPushButton("Apply Selected Field", self)
        apply_btn.clicked.connect(_apply_selected)
        lay.addWidget(table)
        lay.addWidget(apply_btn)
        return w

    def _build_panel_c(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)

        table = QTableWidget(self)
        rows = self.controller.get_meta_rows()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["key", "value"])
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(str(row["key"])))
            table.setItem(r, 1, QTableWidgetItem(str(row["value"])))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        def _apply() -> None:
            updated: dict[str, Any] = {}
            for r in range(table.rowCount()):
                key_item = table.item(r, 0)
                val_item = table.item(r, 1)
                if key_item is None or val_item is None:
                    continue
                key = key_item.text().strip()
                if not key:
                    continue
                raw = val_item.text().strip()
                try:
                    val: Any = json.loads(raw)
                except Exception:
                    val = raw
                updated[key] = val

            self.controller.replace_meta_dict(updated)
            path = self.controller.save_meta_dict()
            self.statusBar().showMessage(
                f"MetaDict updated and saved: {path.name}", 2500
            )

        apply_btn = QPushButton("Apply Meta Changes", self)
        apply_btn.clicked.connect(_apply)
        lay.addWidget(table)
        lay.addWidget(apply_btn)
        return w

    def _build_panel_d(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)

        def _make_library_table(rows: list[dict[str, Any]]) -> QTableWidget:
            table = QTableWidget(self)
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["name", "cfg(json)"])
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                table.setItem(r, 0, QTableWidgetItem(str(row["name"])))
                table.setItem(r, 1, QTableWidgetItem(json.dumps(row["cfg"])))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            return table

        all_rows = self.controller.get_library_rows()
        module_rows = [
            {
                "name": str(row["name"]).split(":", 1)[1],
                "cfg": row["cfg"],
            }
            for row in all_rows
            if str(row["name"]).startswith("module:")
        ]
        waveform_rows = [
            {
                "name": str(row["name"]).split(":", 1)[1],
                "cfg": row["cfg"],
            }
            for row in all_rows
            if str(row["name"]).startswith("waveform:")
        ]

        lay.addWidget(QLabel("modules", self))
        module_table = _make_library_table(module_rows)
        lay.addWidget(module_table)
        add_module_btn = QPushButton("Add Module Row", self)

        def _add_module_row() -> None:
            r = module_table.rowCount()
            module_table.insertRow(r)
            module_table.setItem(r, 0, QTableWidgetItem(""))
            module_table.setItem(r, 1, QTableWidgetItem("{}"))

        add_module_btn.clicked.connect(_add_module_row)
        lay.addWidget(add_module_btn)

        lay.addWidget(QLabel("waveforms", self))
        waveform_table = _make_library_table(waveform_rows)
        lay.addWidget(waveform_table)
        add_waveform_btn = QPushButton("Add Waveform Row", self)

        def _add_waveform_row() -> None:
            r = waveform_table.rowCount()
            waveform_table.insertRow(r)
            waveform_table.setItem(r, 0, QTableWidgetItem(""))
            waveform_table.setItem(r, 1, QTableWidgetItem("{}"))

        add_waveform_btn.clicked.connect(_add_waveform_row)
        lay.addWidget(add_waveform_btn)

        def _apply() -> None:
            updated: dict[str, dict[str, Any]] = {}

            for r in range(module_table.rowCount()):
                name_item = module_table.item(r, 0)
                cfg_item = module_table.item(r, 1)
                if name_item is None or cfg_item is None:
                    continue
                name = name_item.text().strip()
                cfg_text = cfg_item.text().strip()
                if not name:
                    continue
                try:
                    cfg = json.loads(cfg_text)
                except Exception as exc:
                    QMessageBox.warning(
                        self, "D panel", f"Module row {r + 1} JSON parse failed: {exc}"
                    )
                    return
                if not isinstance(cfg, dict):
                    QMessageBox.warning(
                        self, "D panel", f"Module row {r + 1} cfg must be a JSON object"
                    )
                    return
                updated[f"module:{name}"] = cfg

            for r in range(waveform_table.rowCount()):
                name_item = waveform_table.item(r, 0)
                cfg_item = waveform_table.item(r, 1)
                if name_item is None or cfg_item is None:
                    continue
                name = name_item.text().strip()
                cfg_text = cfg_item.text().strip()
                if not name:
                    continue
                try:
                    cfg = json.loads(cfg_text)
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "D panel",
                        f"Waveform row {r + 1} JSON parse failed: {exc}",
                    )
                    return
                if not isinstance(cfg, dict):
                    QMessageBox.warning(
                        self,
                        "D panel",
                        f"Waveform row {r + 1} cfg must be a JSON object",
                    )
                    return
                updated[f"waveform:{name}"] = cfg

            try:
                self.controller.replace_library_items(updated)
            except Exception as exc:
                QMessageBox.warning(self, "D panel", str(exc))
                return
            path = self.controller.save_module_library()
            self.statusBar().showMessage(
                f"Library updated and saved: {path.name}", 2500
            )

        apply_btn = QPushButton("Apply Library Changes", self)
        apply_btn.clicked.connect(_apply)
        lay.addWidget(apply_btn)
        return w

    def _on_context_changed(self) -> None:
        item = self.context_list.currentItem()
        if item is None:
            return
        self.controller.set_context(item.text())
        self._reload_tree()
        self._reload_right_panels()
        self.statusBar().showMessage(f"Switched context: {item.text()}", 2000)

    def _on_tree_clicked(self, index) -> None:
        path = Path(self.file_model.filePath(index))
        if not path.is_file():
            return
        self.controller.open_file_buffer(path)
        self._reload_groups()

    def _on_group_changed(self, index: int) -> None:
        keys = list(self.controller.state.groups.keys())
        if index < 0 or index >= len(keys):
            return
        self.controller.state.set_current_group(keys[index])
        self._render_current_buffer()

    def _step_buffer(self, step: int) -> None:
        group = self.controller.state.current_group()
        if group is None or not group.buffer_ids:
            return
        group.current_index = (group.current_index + step) % len(group.buffer_ids)
        self._render_current_buffer()

    def _collect_run_cfg(self) -> None:
        if hasattr(self, "points_edit"):
            self.controller.exp_cfg["sweep_points"] = int(
                float(self.points_edit.text())
            )
            self.controller.exp_cfg["step_delay_s"] = float(self.delay_edit.text())
            self.controller.exp_cfg["center_mhz"] = float(self.center_edit.text())
            self.controller.exp_cfg["width_mhz"] = float(self.width_edit.text())

    def _on_run_clicked(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Run", "A run is already in progress")
            return

        try:
            self._collect_run_cfg()
        except Exception as exc:
            QMessageBox.warning(self, "Run cfg", str(exc))
            return

        self.cancel_event.clear()
        self.progress.setValue(0)
        self.progress_label.setText("Running")

        self.worker = RunWorker(self.controller, self.cancel_event)
        self.worker.progress.connect(self._on_run_progress)
        self.worker.completed.connect(self._on_run_completed)
        self.worker.failed.connect(self._on_run_failed)
        self.worker.start()

    def _on_stop_clicked(self) -> None:
        self.cancel_event.set()
        self.progress_label.setText("Stopping...")

    def _on_run_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / max(total, 1))
        self.progress.setValue(pct)
        self.progress_label.setText(f"Running {current}/{total}")

    def _on_run_completed(self, payload: dict) -> None:
        self.progress.setValue(100)
        suffix = " (partial)" if payload.get("partial") else ""
        self.progress_label.setText(f"Run completed{suffix}")
        if hasattr(self, "run_text"):
            self.run_text.append(json.dumps(payload))

    def _on_run_failed(self, message: str) -> None:
        self.progress_label.setText("Run failed")
        QMessageBox.warning(self, "Run failed", message)

    def _on_analyze_clicked(self) -> None:
        try:
            result = self.controller.analyze_last_result()
        except Exception as exc:
            QMessageBox.warning(self, "Analyze", str(exc))
            return
        self._reload_groups()
        self._render_current_buffer()
        self.statusBar().showMessage(
            f"Analyze done: peak={result['peak_freq_mhz']:.3f} MHz", 2500
        )

    def _on_save_run_clicked(self) -> None:
        try:
            path = self.controller.save_run_payload()
        except Exception as exc:
            QMessageBox.warning(self, "Save run", str(exc))
            return
        self.statusBar().showMessage(f"Saved run: {path}", 3000)

    def _on_save_analysis_clicked(self) -> None:
        try:
            path = self.controller.save_analysis_figure()
        except Exception as exc:
            QMessageBox.warning(self, "Save figure", str(exc))
            return
        self.statusBar().showMessage(f"Saved analysis figure: {path}", 3000)

    def _on_apply_analysis_clicked(self) -> None:
        if not self.controller.last_analysis:
            QMessageBox.information(self, "Apply analysis", "No analysis yet")
            return
        try:
            meta_path, lib_path = self.controller.apply_analysis_to_context()
        except Exception as exc:
            QMessageBox.warning(self, "Apply analysis", str(exc))
            return
        self._reload_right_panels()
        self.statusBar().showMessage(
            f"Applied analysis and saved: {meta_path.name}, {lib_path.name}", 3000
        )


def run_app(
    project_root: Optional[Path] = None,
    backend: str = "mock",
) -> None:
    if backend != "mock":
        raise ValueError(f"Unsupported backend in fake-only mode: {backend}")
    root = project_root or Path.cwd()
    app = QApplication.instance() or QApplication([])
    win = MainWindow(root, backend=backend)
    win.show()
    app.exec()
