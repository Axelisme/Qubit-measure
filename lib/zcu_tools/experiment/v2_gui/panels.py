from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .adapters import ConfigFieldSchema

try:
    from PySide6.QtCore import Signal
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("PySide6 is required for GUI mode.") from exc


class ExpCfgSchemaPanel(QWidget):
    applyRequested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._schema: list[ConfigFieldSchema] = []
        self._editors: dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)
        self._form = QFormLayout()
        layout.addLayout(self._form)
        apply_btn = QPushButton("Apply exp_cfg", self)
        apply_btn.clicked.connect(self._emit_apply)
        layout.addWidget(apply_btn)

    def set_schema_and_cfg(
        self, schema: list[ConfigFieldSchema], cfg: dict[str, Any]
    ) -> None:
        self._schema = list(schema)
        self._editors.clear()
        while self._form.rowCount() > 0:
            self._form.removeRow(0)

        for field in self._schema:
            editor = QLineEdit(self)
            value = cfg.get(field.key, field.default)
            editor.setText("" if value is None else str(value))
            self._form.addRow(field.label, editor)
            self._editors[field.key] = editor

    def _emit_apply(self) -> None:
        payload: dict[str, Any] = {}
        for field in self._schema:
            editor = self._editors.get(field.key)
            if editor is None:
                continue
            raw = editor.text().strip()
            if field.field_type == "int":
                payload[field.key] = int(float(raw))
            elif field.field_type == "float":
                payload[field.key] = float(raw)
            elif field.field_type == "bool":
                payload[field.key] = raw.lower() in {"1", "true", "yes", "on"}
            else:
                payload[field.key] = raw
        self.applyRequested.emit(payload)


class RunPanel(QWidget):
    runRequested = Signal()
    stopRequested = Signal()
    analyzeRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        btn_row = QHBoxLayout()
        run_btn = QPushButton("Run", self)
        stop_btn = QPushButton("Stop", self)
        analyze_btn = QPushButton("Analyze", self)
        run_btn.clicked.connect(self.runRequested.emit)
        stop_btn.clicked.connect(self.stopRequested.emit)
        analyze_btn.clicked.connect(self.analyzeRequested.emit)
        btn_row.addWidget(run_btn)
        btn_row.addWidget(stop_btn)
        btn_row.addWidget(analyze_btn)
        layout.addLayout(btn_row)
        self.status_label = QLabel("Idle", self)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.liveplot_host = QWidget(self)
        self.liveplot_host.setMinimumHeight(320)
        host_layout = QVBoxLayout(self.liveplot_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.addStretch(1)
        layout.addWidget(self.liveplot_host)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def reset_progress(self) -> None:
        self.progress_bar.setValue(0)

    def set_progress(self, current: int, total: int) -> None:
        safe_total = max(total, 1)
        pct = int(100 * current / safe_total)
        self.progress_bar.setValue(max(0, min(100, pct)))


class AnalyzePanel(QWidget):
    saveFigureRequested = Signal()
    applyRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QTextEdit(self)
        self.summary.setReadOnly(True)
        self.image_label = QLabel(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.image_label)
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Figure", self)
        apply_btn = QPushButton("Overwrite Meta", self)
        save_btn.clicked.connect(self.saveFigureRequested.emit)
        apply_btn.clicked.connect(self.applyRequested.emit)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    def set_analysis(self, analysis: Dict[str, Any]) -> None:
        self.summary.setText(json.dumps(analysis, indent=2))
        fig_path = analysis.get("figure_path")
        if not isinstance(fig_path, str) or not fig_path:
            self.image_label.setText("No analysis image yet")
            return
        path = Path(fig_path)
        if not path.exists():
            self.image_label.setText(f"Analysis image not found: {path}")
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self.image_label.setText(f"Failed to load analysis image: {path}")
            return
        self.image_label.setPixmap(pix.scaledToWidth(760))


class DevicePanel(QWidget):
    applyRequested = Signal(str, dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tables: dict[str, QTableWidget] = {}
        self._layout = QVBoxLayout(self)

    def set_device_infos(self, infos: dict[str, dict[str, Any]]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._tables.clear()
        for dev_name, info in infos.items():
            self._layout.addWidget(QLabel(dev_name, self))
            table = QTableWidget(self)
            editable = [(k, v) for k, v in info.items() if k not in {"type", "address"}]
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["field", "value"])
            table.setRowCount(len(editable))
            for row, (field, value) in enumerate(editable):
                table.setItem(row, 0, QTableWidgetItem(str(field)))
                table.setItem(row, 1, QTableWidgetItem(str(value)))
            self._tables[dev_name] = table
            self._layout.addWidget(table)
            apply_btn = QPushButton(f"Apply {dev_name}", self)
            apply_btn.clicked.connect(
                lambda _=False, name=dev_name: self._emit_apply(name)
            )
            self._layout.addWidget(apply_btn)

    def _emit_apply(self, name: str) -> None:
        table = self._tables.get(name)
        if table is None:
            return
        payload: dict[str, Any] = {}
        for row in range(table.rowCount()):
            field = table.item(row, 0)
            value = table.item(row, 1)
            if field is None or value is None:
                continue
            raw = value.text()
            try:
                payload[field.text()] = float(raw)
            except ValueError:
                payload[field.text()] = raw
        self.applyRequested.emit(name, payload)


class MetaPanel(QWidget):
    applyRequested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["key", "value"])
        layout.addWidget(self.table)
        apply_btn = QPushButton("Apply Meta Changes", self)
        apply_btn.clicked.connect(self._emit_apply)
        layout.addWidget(apply_btn)

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row["key"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(row["value"])))

    def _emit_apply(self) -> None:
        payload: dict[str, Any] = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_item = self.table.item(row, 1)
            if key_item is None or val_item is None:
                continue
            key = key_item.text().strip()
            if not key:
                continue
            text = val_item.text().strip()
            try:
                value: Any = int(text)
            except ValueError:
                try:
                    value = float(text)
                except ValueError:
                    value = text
            payload[key] = value
        self.applyRequested.emit(payload)


class LibraryPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["name", "value"])
        layout.addWidget(self.tree)

    def set_snapshot(self, snapshot: dict[str, dict[str, Any]]) -> None:
        self.tree.clear()
        modules = QTreeWidgetItem(["modules", ""])
        waveforms = QTreeWidgetItem(["waveforms", ""])
        self.tree.addTopLevelItem(modules)
        self.tree.addTopLevelItem(waveforms)

        def _populate(parent: QTreeWidgetItem, key: str, value: Any) -> None:
            node = QTreeWidgetItem([str(key), ""])
            parent.addChild(node)
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _populate(node, str(sub_key), sub_value)
                return
            if isinstance(value, list):
                for idx, sub_value in enumerate(value):
                    _populate(node, f"[{idx}]", sub_value)
                return
            node.setText(1, str(value))

        for name, cfg in snapshot.get("modules", {}).items():
            _populate(modules, name, cfg)
        for name, cfg in snapshot.get("waveforms", {}).items():
            _populate(waveforms, name, cfg)
