from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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

from zcu_tools.gui.app.main.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
    WritebackItem,
)

from .cfg_form import CfgFormWidget

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller

logger = logging.getLogger(__name__)


class WritebackWidget(QWidget):
    apply_requested: Signal = Signal()  # apply the persistent draft as-is

    def __init__(
        self,
        ctrl: Controller,
        parent: QWidget | None = None,
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

        # These are the *persistent* State items (ADR-0008) — do NOT copy. UI
        # edits (checkbox, value, cfg via the editor model) land on the same
        # objects the agent and apply read.
        self._items = list(items)
        self._checks.clear()

        for item in self._items:
            row = QWidget(self)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = self._make_label_text(item)
            cb = QCheckBox(label)
            cb.setChecked(item.selected)
            cb.stateChanged.connect(lambda _state, it=item: self._on_check_toggled(it))
            row_layout.addWidget(cb, 1)
            self._checks[item.session_id] = cb

            if self._is_editable(item):
                edit_btn = QPushButton("Edit")
                edit_btn.clicked.connect(
                    lambda _=False, it=item, chk=cb: self._edit_item(it, chk)
                )
                row_layout.addWidget(edit_btn)

            self._rows_layout.addWidget(row)

        self._refresh_apply_enabled()

    def _on_check_toggled(self, item: WritebackItem) -> None:
        # The persistent item's selected flag follows the checkbox directly.
        item.selected = self._checks[item.session_id].isChecked()
        self._refresh_apply_enabled()

    def _refresh_apply_enabled(self, *_: int) -> None:
        self._apply_btn.setEnabled(any(cb.isChecked() for cb in self._checks.values()))

    def _on_apply_clicked(self) -> None:
        self.apply_requested.emit()

    def _is_editable(self, item: WritebackItem) -> bool:
        if isinstance(item, MetaDictWriteback):
            # Scalar md values are hand-editable; a non-scalar (list/matrix, e.g.
            # the singleshot confusion matrix) is a derived value applied verbatim
            # — shown read-only, no Edit dialog (the scalar coercion can't parse a
            # matrix and the user does not hand-tune it).
            return _is_scalar_md_value(item.proposed_value)
        if isinstance(item, ModuleWriteback):
            return item.edit_schema is not None
        if isinstance(item, WaveformWriteback):
            return item.edit_schema is not None
        return False

    def _make_label_text(self, item: WritebackItem) -> str:
        if isinstance(item, MetaDictWriteback):
            return (
                f"{item.target_name} -> {item.proposed_value!r}\n  {item.description}"
            )
        if isinstance(item, (ModuleWriteback, WaveformWriteback)):
            return f"{item.target_name}\n  {item.description}"
        return f"{item.target_name}\n  {item.description}"

    def _edit_item(self, item: WritebackItem, cb: QCheckBox) -> None:
        if isinstance(item, MetaDictWriteback):
            self._edit_md_item(item, cb)
        elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
            self._edit_cfg_item(item, cb)

    def _edit_md_item(self, item: MetaDictWriteback, cb: QCheckBox) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Value: {item.target_name}")
        layout = QVBoxLayout(dialog)

        form = QFormLayout()
        # target_name is the apply destination, decoupled from the stable
        # session_id (ADR-0008) — editable here so the user can retarget.
        name_edit = QLineEdit(item.target_name)
        form.addRow("Apply as:", name_edit)
        value_edit = QLineEdit(str(item.proposed_value))
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
                new_name = _require_target_name(name_edit.text())
                item.proposed_value = _coerce_scalar_input(
                    value_edit.text(),
                    item.proposed_value,
                )
                item.target_name = new_name
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
    ) -> None:
        # The item carries a persistent, service-owned cfg model (editor_id,
        # ADR-0008). Attach the dialog widget to *that* model — the user's edits
        # land on the same model the agent edits and apply reads. On close the
        # widget detaches but the model persists (torn down only on reanalyze).
        if item.editor_id is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Config: {item.target_name}")
        dialog.setMinimumSize(560, 500)

        layout = QVBoxLayout(dialog)

        hint = QLabel("Edit the configuration below. Edits apply immediately.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # target_name is the apply destination, decoupled from the stable
        # session_id (ADR-0008) — editable so the user can retarget. Like the cfg
        # edits below, a valid change applies immediately; a blank field is left
        # on the previous name (revert on focus-out).
        name_row = QFormLayout()
        name_edit = QLineEdit(item.target_name)
        name_row.addRow("Apply as:", name_edit)
        layout.addLayout(name_row)

        def _commit_name() -> None:
            text = name_edit.text().strip()
            if not text:
                name_edit.setText(item.target_name)  # revert, no blank target
                return
            item.target_name = text

        name_edit.editingFinished.connect(_commit_name)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = CfgFormWidget()
        form_widget.attach(self._ctrl.get_cfg_editor_root(item.editor_id))
        scroll.setWidget(form_widget)
        layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        close_btn = QPushButton("Close")
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        close_btn.clicked.connect(dialog.accept)

        def _on_finished(*_: Any) -> None:
            _commit_name()
            form_widget.detach()
            cb.setText(self._make_label_text(item))

        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(_on_finished)
        dialog.open()


def _is_scalar_md_value(value: Any) -> bool:
    """A md proposed_value the scalar Edit dialog can round-trip.

    Scalars (bool/int/float/complex/str/None) are hand-editable via
    ``_coerce_scalar_input``; a non-scalar (list/matrix, e.g. the confusion
    matrix) is a derived value applied verbatim, so the UI treats it read-only.
    """
    return value is None or isinstance(value, (bool, int, float, complex, str))


def _require_target_name(text: str) -> str:
    """Validate an apply-destination name (mirrors the tab.writeback_set guard)."""
    name = text.strip()
    if not name:
        raise RuntimeError("Apply-as name must not be empty")
    return name


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
    # complex before float: a complex md value (e.g. a single-shot IQ centre)
    # parses via Python's ``complex("1+2j")``. ``float`` would reject "1+2j", so
    # this branch must precede it. ``complex`` also accepts a bare real ("1.5"),
    # which keeps a real-only re-entry valid.
    if isinstance(original, complex):
        return complex(text.strip())
    if isinstance(original, float):
        return float(text)
    return text
