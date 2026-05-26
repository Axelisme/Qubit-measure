from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence

from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
    WritebackItem,
)

from .cfg_form import CfgFormWidget

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller

logger = logging.getLogger(__name__)


class WritebackWidget(QWidget):
    apply_requested: Signal = Signal(list)  # list[WritebackItem]

    def __init__(
        self,
        ctrl: "Controller",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._items: list[WritebackItem] = []
        self._checks: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._hint = QLabel(
            "Select the items to write back. Use Edit to adjust values first."
        )
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        layout.addWidget(self._rows_container)

        self._apply_btn = QPushButton("Apply Selected")
        self._apply_btn.setFixedHeight(30)
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self._apply_btn)

        self._refresh_apply_enabled()

    def populate(self, items: Sequence[WritebackItem]) -> None:
        # Clear old rows
        while self._rows_layout.count():
            child = self._rows_layout.takeAt(0)
            if child is not None:
                w = child.widget()
                if w is not None:
                    w.deleteLater()

        self._items = copy.deepcopy(list(items))
        self._checks.clear()

        for item in self._items:
            row = QWidget(self)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = self._make_label_text(item)
            cb = QCheckBox(label)
            cb.setChecked(item.selected)
            cb.stateChanged.connect(self._refresh_apply_enabled)
            row_layout.addWidget(cb, 1)
            self._checks[item.key] = cb

            if self._is_editable(item):
                edit_btn = QPushButton("Edit")
                edit_btn.clicked.connect(
                    lambda _=False, it=item, chk=cb: self._edit_item(it, chk)
                )
                row_layout.addWidget(edit_btn)

            self._rows_layout.addWidget(row)

        self._refresh_apply_enabled()

    def _refresh_apply_enabled(self, *_: int) -> None:
        self._apply_btn.setEnabled(any(cb.isChecked() for cb in self._checks.values()))

    def get_selected_items(self) -> list[WritebackItem]:
        selected: list[WritebackItem] = []
        for item in self._items:
            check = self._checks[item.key]
            item.selected = check.isChecked()
            if item.selected:
                selected.append(item)
        return selected

    def _on_apply_clicked(self) -> None:
        self.apply_requested.emit(self.get_selected_items())

    def _is_editable(self, item: WritebackItem) -> bool:
        if isinstance(item, MetaDictWriteback):
            return True
        if isinstance(item, ModuleWriteback):
            return item.edit_schema is not None
        if isinstance(item, WaveformWriteback):
            return item.edit_schema is not None
        return False

    def _make_label_text(self, item: WritebackItem) -> str:
        if isinstance(item, MetaDictWriteback):
            return (
                f"{item.key}  ({item.current_value!r} -> {item.proposed_value!r})\n"
                f"  {item.description}"
            )
        if isinstance(item, ModuleWriteback):
            status = (
                "Config edited" if item.edited_schema is not None else "Config modified"
            )
            name_part = (
                f" -> {item.module_name}" if item.module_name != item.key else ""
            )
            return f"{item.key}{name_part}  ({status})\n  {item.description}"
        if isinstance(item, WaveformWriteback):
            status = (
                "Config edited" if item.edited_schema is not None else "Config modified"
            )
            name_part = (
                f" -> {item.waveform_name}" if item.waveform_name != item.key else ""
            )
            return f"{item.key}{name_part}  ({status})\n  {item.description}"
        return f"{item.key}\n  {item.description}"

    def _edit_item(self, item: WritebackItem, cb: QCheckBox) -> None:
        if isinstance(item, MetaDictWriteback):
            self._edit_md_item(item, cb)
        elif isinstance(item, ModuleWriteback):
            self._edit_cfg_item(item, cb, item.module_name)
        elif isinstance(item, WaveformWriteback):
            self._edit_cfg_item(item, cb, item.waveform_name)

    def _edit_md_item(self, item: MetaDictWriteback, cb: QCheckBox) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Value: {item.key}")
        layout = QVBoxLayout(dialog)

        form = QFormLayout()
        name_edit = QLineEdit(item.md_key)
        value_edit = QLineEdit(str(item.proposed_value))
        form.addRow("Key:", name_edit)
        form.addRow("Value:", value_edit)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def save() -> None:
            try:
                item.md_key = name_edit.text().strip() or item.md_key
                item.proposed_value = _coerce_scalar_input(
                    value_edit.text(),
                    item.proposed_value,
                )
                cb.setText(self._make_label_text(item))
                dialog.accept()
            except Exception as exc:
                QMessageBox.critical(dialog, "Validation Error", str(exc))

        save_btn.clicked.connect(save)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()

    def _edit_cfg_item(
        self,
        item: ModuleWriteback | WaveformWriteback,
        cb: QCheckBox,
        initial_name: str,
    ) -> None:
        if item.edit_schema is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Config: {item.key}")
        dialog.setMinimumSize(560, 500)

        layout = QVBoxLayout(dialog)
        name_form = QFormLayout()
        name_edit = QLineEdit(initial_name)
        name_form.addRow("Name:", name_edit)
        layout.addLayout(name_form)

        hint = QLabel("Edit the configuration below. Click Save to confirm.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = CfgFormWidget()
        schema = copy.deepcopy(item.edited_schema or item.edit_schema)
        form_widget.populate(schema, self._ctrl)
        initial_valid = form_widget.is_valid()
        logger.debug(
            "_edit_cfg_item: key=%r initial_valid=%r schema_spec=%r",
            item.key,
            initial_valid,
            type(schema.spec).__name__,
        )
        scroll.setWidget(form_widget)
        layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.setEnabled(initial_valid)
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        form_widget.validity_changed.connect(save_btn.setEnabled)
        cancel_btn.clicked.connect(dialog.reject)

        def save() -> None:
            try:
                updated = form_widget.read_schema()
                if isinstance(item, ModuleWriteback):
                    item.module_name = name_edit.text().strip() or item.module_name
                else:
                    item.waveform_name = name_edit.text().strip() or item.waveform_name
                item.edited_schema = updated
                cb.setText(self._make_label_text(item))
                dialog.accept()
            except Exception as exc:
                QMessageBox.critical(dialog, "Validation Error", str(exc))

        save_btn.clicked.connect(save)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(lambda _: form_widget.clear())
        dialog.open()


def _coerce_scalar_input(text: str, original: Any) -> Any:
    if isinstance(original, bool):
        lowered = text.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise RuntimeError(f"Invalid bool value: {text}")
    if isinstance(original, int) and not isinstance(original, bool):
        return int(text)
    if isinstance(original, float):
        return float(text)
    return text
