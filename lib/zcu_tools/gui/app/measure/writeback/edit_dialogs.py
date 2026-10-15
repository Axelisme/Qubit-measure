from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
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

from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.form import FormErrors, FormWritePath, WriteRejected
from zcu_tools.gui.widgets import error_colour, secondary_colour, set_foreground
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from .items import ModuleWriteback, ParameterWriteback

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.session.library_entry import ModuleEntryForm

    from .control import WritebackPane
    from .service import WritebackValues

logger = logging.getLogger(__name__)


class WritebackEditDialogs:
    """Own transient writeback editors for one pane without owning its draft.

    The parent owns Qt lifetime; ctrl admits all writes for tab_id and pane.
    Module dialogs are reused until finished and detach from the shared form on close.
    """

    def __init__(
        self, parent: QWidget, ctrl: Controller, tab_id: str, pane: WritebackPane
    ) -> None:
        self._parent = parent
        self._ctrl = ctrl
        self._tab_id = tab_id
        self._pane: WritebackPane = pane
        self._module_dialogs: dict[str, QDialog] = {}

    def open_module_ids(self) -> tuple[str, ...]:
        """Return stable item IDs whose module editors still hold local input."""
        return tuple(self._module_dialogs)

    def edit_value(
        self,
        item: ParameterWriteback,
        values: WritebackValues,
        refresh: Callable[[], None],
    ) -> None:
        """Open a transactional scalar editor using captured preview facts.

        A successful pane-qualified write updates the persistent item and calls
        refresh. Validation and controller refusals stay in the open dialog.
        """

        def save(name: str, value: Any) -> None:
            self._ctrl.set_writeback_item_for_pane(
                self._tab_id,
                self._pane,
                item.session_id,
                target_name=name,
                proposed_value=value,
            )
            item.target_name = name
            item.proposed_value = value
            refresh()

        dialog = _ValueDialog(self._parent, item, values, save)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()

    def _edit_module_form(
        self, session_id: str, edit: Callable[[ModuleEntryForm], None]
    ) -> None:
        """Re-admit a write or option command and translate only expected refusals."""
        try:
            form = self._ctrl.get_writeback_item_form_for_pane(
                self._tab_id, self._pane, session_id
            )
            edit(form)
        except ExpectedError as exc:
            raise WriteRejected(str(exc)) from exc

    def edit_module(
        self,
        item: ModuleWriteback,
        refresh: Callable[..., None],
    ) -> None:
        """Open the shared draft form; report validity to the owning ledger row."""
        try:
            form = self._ctrl.get_writeback_item_form_for_pane(
                self._tab_id, self._pane, item.session_id
            )
        except Exception as exc:
            logger.exception(
                "failed to resolve writeback draft item %s", item.session_id
            )
            QMessageBox.critical(self._parent, "Unable to edit writeback", str(exc))
            return

        existing = self._module_dialogs.get(item.session_id)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return
        dialog = QDialog(self._parent)
        dialog.setWindowTitle(f"Edit Config: {item.target_name}")
        dialog.setMinimumSize(560, 500)

        layout = QVBoxLayout(dialog)

        hint = QLabel("Edit the configuration below. Edits apply immediately.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
            self._edit_module_form(
                item.session_id, lambda form: form.state.write(batch, form.env)
            )

        def run_command(path: FormWritePath, name: str) -> None:
            self._edit_module_form(
                item.session_id,
                lambda form: form.state.run_command(path, name, form.env),
            )

        editor = FormEditor(form.state, form.env, write=write, run_command=run_command)

        def _refresh_preview(*_: object) -> None:
            refresh(
                valid=editor.errors() is None
                and not isinstance(form.result(), FormErrors),
            )

        commit_name = self._add_module_target_row(layout, item, _refresh_preview)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(editor)
        layout.addWidget(scroll, stretch=1)
        editor.validity_changed.connect(_refresh_preview)
        unsubscribe = form.state.subscribe(_refresh_preview)
        dialog.destroyed.connect(unsubscribe)
        self._module_dialogs[item.session_id] = dialog
        _refresh_preview()
        self._add_cfg_close_action(dialog, layout)

        def _on_finished(*_: object) -> None:
            if self._module_dialogs.get(item.session_id) is not dialog:
                return
            # Local unparsed text belongs to this view and is dropped on Close.
            # Model/expression errors remain in the shared State. Neither writes.
            try:
                commit_name()
            finally:
                unsubscribe()
                editor.validity_changed.disconnect(_refresh_preview)
                editor.detach()
                self._module_dialogs.pop(item.session_id)
            refresh(valid=not isinstance(form.result(), FormErrors))

        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(_on_finished)
        dialog.open()

    def _add_module_target_row(
        self,
        layout: QVBoxLayout,
        item: ModuleWriteback,
        refresh_preview: Callable[[], None],
    ) -> Callable[[], None]:
        """Add Apply-as; return its immediate commit action for Close.

        Blank text reverts to the existing name. A changed nonblank name updates
        the persistent item via its pane-qualified control and refreshes validity.
        """
        name_row = QFormLayout()
        name_edit = QLineEdit(item.target_name)
        name_row.addRow("Apply as:", name_edit)
        layout.addLayout(name_row)

        def commit_name() -> None:
            text = name_edit.text().strip()
            if not text:
                name_edit.setText(item.target_name)
                return
            if text == item.target_name:
                return
            self._ctrl.set_writeback_item_for_pane(
                self._tab_id, self._pane, item.session_id, target_name=text
            )
            refresh_preview()

        name_edit.editingFinished.connect(commit_name)
        return commit_name

    def _add_cfg_close_action(self, dialog: QDialog, layout: QVBoxLayout) -> None:
        """Append a Close action to the modal editor."""
        button_row = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)


class _EditDialog(QDialog):
    """Shared palette-aware text feedback for the two draft editors."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._foregrounds: dict[QLabel, Callable[[QWidget], QColor]] = {}

    def theme_label(self, label: QLabel, colour: Callable[[QWidget], QColor]) -> None:
        self._foregrounds[label] = colour
        set_foreground(label, colour(self))

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if a0 is not None and a0.type() == QEvent.Type.PaletteChange:
            for label, colour in self._foregrounds.items():
                set_foreground(label, colour(self))

    def context_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        make_label_selectable(label)
        return label


class _ValueDialog(_EditDialog):
    def __init__(
        self,
        parent: QWidget,
        item: ParameterWriteback,
        values: WritebackValues,
        save: Callable[[str, Any], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit value — {item.target_name}")
        self.setMinimumWidth(420)
        self.resize(520, self.sizeHint().height())
        self._item = item
        self._unit = values.unit
        self._save = save
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(self.context_label(item.description))
        path = self.context_label(f"Writes to  {values.actual_path}")
        self.theme_label(path, secondary_colour)
        layout.addWidget(path)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        current = "—" if values.current is None else str(values.current)
        if values.current is not None and values.unit:
            current += f" {values.unit}"
        current_label = self.context_label(current)
        current_label.setObjectName("writebackCurrentReadonly")
        self.theme_label(current_label, secondary_colour)
        form.addRow("Current", current_label)
        # Create the target first to retain existing frontend field discovery order.
        self._name = QLineEdit(item.target_name, self)
        self._name.setObjectName("writebackApplyAs")
        self._value = QLineEdit(str(item.proposed_value), self)
        self._value.setObjectName("writebackValue")
        proposal = QHBoxLayout()
        proposal.addWidget(self._value, stretch=1)
        if values.unit:
            proposal.addWidget(QLabel(values.unit))
        form.addRow("Proposed", proposal)
        if item.stderr is not None:
            stderr = format(item.stderr, ".6g")
            if values.unit:
                stderr += f" {values.unit}"
            stderr_label = self.context_label(f"± {stderr} (fit stderr)")
            self.theme_label(stderr_label, secondary_colour)
            form.addRow("", stderr_label)
        form.addRow("Apply as", self._name)
        layout.addLayout(form)
        self._error = self.context_label("")
        self._error.setObjectName("writebackEditError")
        self.theme_label(self._error, error_colour)
        layout.addWidget(self._error)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        self._save_button = QPushButton("Save")
        self._save_button.setDefault(True)
        self._save_button.clicked.connect(self._commit)
        buttons.addWidget(cancel)
        buttons.addWidget(self._save_button)
        layout.addLayout(buttons)
        self._name.textChanged.connect(self._validate)
        self._value.textChanged.connect(self._validate)
        self._validate()

    def _parsed(self) -> tuple[str, Any]:
        name = _require_target_name(self._name.text())
        text = self._value.text().strip()
        if self._unit and text.endswith(self._unit):
            text = text[: -len(self._unit)].rstrip()
        original = self._item.proposed_value
        try:
            value = _coerce_scalar_input(text, original)
        except (ValueError, RuntimeError) as exc:
            if isinstance(original, bool):
                message = "Enter true or false"
            elif isinstance(original, int):
                message = "Enter an integer"
            elif isinstance(original, complex):
                message = "Enter a complex number, e.g. 1+2j"
            else:
                message = "Enter a number, e.g. 6012.35"
            raise ValueError(message) from exc
        return name, value

    def _show_error(self, text: str) -> None:
        self._error.setText(text)
        self._error.setVisible(bool(text))

    def _validate(self) -> None:
        try:
            self._parsed()
        except (ValueError, RuntimeError) as exc:
            self._show_error(str(exc))
            self._save_button.setEnabled(False)
        else:
            self._show_error("")
            self._save_button.setEnabled(True)

    def _commit(self) -> None:
        try:
            name, value = self._parsed()
            self._save(name, value)
        except Exception as exc:
            logger.exception("failed to edit writeback value %s", self._item.session_id)
            self._show_error(str(exc))
            return
        self.accept()


def is_scalar_value(value: Any) -> bool:
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
