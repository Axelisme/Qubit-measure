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
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileSystemModel,
        QFormLayout,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
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
        QToolBox,
        QTreeWidget,
        QTreeWidgetItem,
        QTreeView,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PySide6 is required for GUI mode. Install it first, e.g. `uv add pyside6`."
    ) from exc

try:
    from zcu_tools.liveplot.backend.pyside6 import clear_plot_host, set_plot_host
except Exception:  # pragma: no cover
    def set_plot_host(*args, **kwargs) -> None:
        return None

    def clear_plot_host() -> None:
        return None


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

        self.controller = GuiController(project_root)
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
        self.tree.doubleClicked.connect(self._on_tree_clicked)
        self.tree.setModel(self.file_model)
        self.tree.setHeaderHidden(False)
        top_layout.addWidget(self.tree)

        bottom = QWidget(self)
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.addWidget(QLabel("Context labels"))
        self.context_list = QListWidget(self)
        bottom_layout.addWidget(self.context_list)
        context_btns = QWidget(self)
        context_btns_layout = QHBoxLayout(context_btns)
        activate_btn = QPushButton("Activate", self)
        new_btn = QPushButton("New", self)
        clone_btn = QPushButton("Clone", self)
        activate_btn.clicked.connect(self._on_activate_context)
        new_btn.clicked.connect(self._on_new_context)
        clone_btn.clicked.connect(self._on_clone_context)
        context_btns_layout.addWidget(activate_btn)
        context_btns_layout.addWidget(new_btn)
        context_btns_layout.addWidget(clone_btn)
        bottom_layout.addWidget(context_btns)

        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([520, 240])

        layout.addWidget(split)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        self.group_tabs = QTabBar(self)
        self.group_tabs.setTabsClosable(True)
        self.group_tabs.currentChanged.connect(self._on_group_changed)
        self.group_tabs.tabCloseRequested.connect(self._on_group_close_requested)
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

        self.liveplot_host = QWidget(self)
        host_layout = QVBoxLayout(self.liveplot_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.addWidget(QLabel("Live plot will be shown here during run.", self))
        lay.addWidget(self.liveplot_host)

        self.run_text = QTextEdit(self)
        self.run_text.setReadOnly(True)
        self.run_text.setPlaceholderText("Run logs...")
        lay.addWidget(self.run_text)

        run_buffer = self.controller.state.buffers.get(
            self.controller._active_group_model().run_buffer_id
        )
        self.comment_edit = QPlainTextEdit(self)
        initial_comment = ""
        if run_buffer is not None:
            initial_comment = str(run_buffer.payload.get("comment", ""))
        self.comment_edit.setPlainText(initial_comment)
        self.comment_edit.setPlaceholderText("Comment for this experiment run...")
        lay.addWidget(QLabel("Comment", self))
        lay.addWidget(self.comment_edit)

        def _save_comment() -> None:
            if run_buffer is not None:
                run_buffer.payload["comment"] = self.comment_edit.toPlainText()
                self.statusBar().showMessage("Comment saved", 2000)

        save_comment_btn = QPushButton("Save Comment", self)
        save_comment_btn.clicked.connect(_save_comment)
        lay.addWidget(save_comment_btn)

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
        toolbox = QToolBox(self)
        device_infos = self.controller.get_device_infos()
        if not device_infos:
            lay.addWidget(QLabel("No registered devices"))
            return w

        for device_name, info in device_infos.items():
            panel = QWidget(self)
            panel_layout = QVBoxLayout(panel)
            table = QTableWidget(self)
            editable_fields = [(k, v) for k, v in info.items() if k != "type" and k != "address"]
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["field", "value"])
            table.setRowCount(len(editable_fields))
            for r, (field, value) in enumerate(editable_fields):
                table.setItem(r, 0, QTableWidgetItem(str(field)))
                table.setItem(r, 1, QTableWidgetItem(str(value)))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            panel_layout.addWidget(table)

            def _make_apply(name: str, t: QTableWidget):
                def _apply() -> None:
                    for r in range(t.rowCount()):
                        field_item = t.item(r, 0)
                        value_item = t.item(r, 1)
                        if field_item is None or value_item is None:
                            continue
                        field = field_item.text()
                        text_value = value_item.text()
                        value: Any = text_value
                        try:
                            value = float(text_value)
                        except ValueError:
                            pass
                        result = self.controller.update_device_field(name, field, value)
                        if not result["ok"]:
                            QMessageBox.warning(
                                self,
                                "dev_cfg",
                                f"{name}.{field} update failed: {result['message']}",
                            )
                            return
                        value_item.setText(str(result["value"]))
                    self.statusBar().showMessage(f"Applied {name} changes", 2000)

                return _apply

            apply_btn = QPushButton("Apply Table", self)
            apply_btn.clicked.connect(_make_apply(device_name, table))
            panel_layout.addWidget(apply_btn)
            toolbox.addItem(panel, device_name)
        lay.addWidget(toolbox)
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

        def _parse_numeric(raw: str) -> Any:
            text = raw.strip()
            if not text:
                raise ValueError("value cannot be empty")
            try:
                return int(text)
            except ValueError:
                pass
            try:
                return float(text)
            except ValueError:
                pass
            try:
                return complex(text)
            except ValueError as exc:
                raise ValueError(
                    "value must be int, float, or complex"
                ) from exc

        def _add_row() -> None:
            key, ok = QInputDialog.getText(self, "Add metadict key", "Key:")
            if not ok:
                return
            key = key.strip()
            if not key:
                QMessageBox.warning(self, "metadict", "Key cannot be empty.")
                return
            raw_value, ok = QInputDialog.getText(
                self, "Add metadict value", "Value (int/float/complex):"
            )
            if not ok:
                return
            try:
                parsed = _parse_numeric(raw_value)
            except ValueError as exc:
                QMessageBox.warning(self, "metadict", str(exc))
                return
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(key))
            table.setItem(r, 1, QTableWidgetItem(str(parsed)))

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
                    val = _parse_numeric(raw)
                except ValueError as exc:
                    QMessageBox.warning(self, "metadict", f"Row {r + 1}: {exc}")
                    return
                updated[key] = val

            self.controller.replace_meta_dict(updated)
            path = self.controller.save_meta_dict()
            self.statusBar().showMessage(
                f"MetaDict updated and saved: {path.name}", 2500
            )

        add_btn = QPushButton("Add", self)
        add_btn.clicked.connect(_add_row)
        apply_btn = QPushButton("Apply Meta Changes", self)
        apply_btn.clicked.connect(_apply)
        lay.addWidget(table)
        lay.addWidget(add_btn)
        lay.addWidget(apply_btn)
        return w

    def _build_panel_d(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        tree = QTreeWidget(self)
        tree.setHeaderLabels(["name", "value"])
        snapshot = self.controller.get_library_snapshot()

        def _populate(parent: QTreeWidgetItem, key: str, value: Any) -> None:
            item = QTreeWidgetItem([str(key), ""])
            parent.addChild(item)
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    _populate(item, str(sub_key), sub_val)
                return
            if isinstance(value, list):
                for i, sub_val in enumerate(value):
                    _populate(item, f"[{i}]", sub_val)
                return
            item.setText(1, str(value))

        root_modules = QTreeWidgetItem(["modules", ""])
        root_waveforms = QTreeWidgetItem(["waveforms", ""])
        tree.addTopLevelItem(root_modules)
        tree.addTopLevelItem(root_waveforms)

        for name, cfg in snapshot.get("modules", {}).items():
            _populate(root_modules, name, cfg)
        for name, cfg in snapshot.get("waveforms", {}).items():
            _populate(root_waveforms, name, cfg)
        tree.expandToDepth(1)
        lay.addWidget(tree)
        return w

    def _on_activate_context(self) -> None:
        item = self.context_list.currentItem()
        if item is None:
            return
        self.controller.set_context(item.text())
        self._reload_tree()
        self._reload_right_panels()
        self.statusBar().showMessage(f"Activated context: {item.text()}", 2000)

    def _prompt_context_label(self, title: str) -> Optional[tuple[str, Optional[str]]]:
        devices = self.controller.get_supported_label_devices()
        selected_device_name: Optional[str] = None
        default_text = self.controller.suggest_auto_label()

        if not devices:
            text, ok = QInputDialog.getText(
                self,
                title,
                "Context label:",
                text=default_text,
            )
            if not ok:
                return None
            return text.strip(), selected_device_name

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        device_combo = QComboBox(dialog)
        for dev in devices:
            device_combo.addItem(f"{dev['name']} ({dev['value']})", dev["name"])
        label_edit = QLineEdit(default_text, dialog)
        form.addRow("Device", device_combo)
        form.addRow("Context label", label_edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog
        )
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        def _refresh_default_label() -> None:
            nonlocal selected_device_name
            selected_device_name = device_combo.currentData()
            label_edit.setText(self.controller.suggest_auto_label(selected_device_name))

        device_combo.currentIndexChanged.connect(_refresh_default_label)

        if dialog.exec() != QDialog.Accepted:
            return None
        return label_edit.text().strip(), selected_device_name

    def _on_new_context(self) -> None:
        result = self._prompt_context_label("New Flux Context")
        if result is None:
            return
        label, _ = result
        if not label:
            QMessageBox.warning(self, "New context", "Context label cannot be empty.")
            return
        try:
            self.controller.create_context(label=label)
        except Exception as exc:
            QMessageBox.warning(self, "New context", str(exc))
            return
        self._load_contexts()
        self._reload_tree()
        self._reload_right_panels()
        self.statusBar().showMessage(f"Created context: {label}", 2500)

    def _on_clone_context(self) -> None:
        src_item = self.context_list.currentItem()
        if src_item is None:
            QMessageBox.information(self, "Clone context", "Please select a source context.")
            return
        result = self._prompt_context_label("Clone Flux Context")
        if result is None:
            return
        label, _ = result
        if not label:
            QMessageBox.warning(self, "Clone context", "Context label cannot be empty.")
            return
        try:
            self.controller.create_context(label=label, clone_from=src_item.text())
        except Exception as exc:
            QMessageBox.warning(self, "Clone context", str(exc))
            return
        self._load_contexts()
        self._reload_tree()
        self._reload_right_panels()
        self.statusBar().showMessage(
            f"Cloned context from {src_item.text()} to {label}", 2500
        )

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

    def _on_group_close_requested(self, index: int) -> None:
        keys = list(self.controller.state.groups.keys())
        if index < 0 or index >= len(keys):
            return
        group_id = keys[index]

        if group_id in self.controller.group_models:
            QMessageBox.information(
                self,
                "Close tab",
                "Experiment tabs are fixed in this phase and cannot be closed.",
            )
            return

        group = self.controller.state.groups.get(group_id)
        if group is None:
            return

        for buffer_id in list(group.buffer_ids):
            self.controller.state.buffers.pop(buffer_id, None)
        self.controller.state.groups.pop(group_id, None)

        remaining = list(self.controller.state.groups.keys())
        self.controller.state.current_group_id = remaining[0] if remaining else None
        self._reload_groups()

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
        if hasattr(self, "liveplot_host"):
            clear_plot_host()
            set_plot_host(self.liveplot_host)

        self.worker = RunWorker(self.controller, self.cancel_event)
        self.worker.progress.connect(self._on_run_progress)
        self.worker.completed.connect(self._on_run_completed)
        self.worker.failed.connect(self._on_run_failed)
        self.worker.start()

    def _on_stop_clicked(self) -> None:
        QMessageBox.information(
            self,
            "Stop",
            "FakeExp uses its native run flow. GUI stop is not implemented in this phase.",
        )

    def _on_run_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / max(total, 1))
        self.progress.setValue(pct)
        self.progress_label.setText(f"Running {current}/{total}")

    def _on_run_completed(self, payload: dict) -> None:
        self.progress.setValue(100)
        clear_plot_host()
        suffix = " (partial)" if payload.get("partial") else ""
        self.progress_label.setText(f"Run completed{suffix}")
        if hasattr(self, "run_text"):
            self.run_text.append(json.dumps(payload))

    def _on_run_failed(self, message: str) -> None:
        clear_plot_host()
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
