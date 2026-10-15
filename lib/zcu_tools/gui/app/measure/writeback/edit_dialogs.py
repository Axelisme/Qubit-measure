from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.form import FormErrors, FormModel, FormWritePath, WriteRejected
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
        """Open the shared draft form, or an inline refusal if it is unavailable.

        Re-admit every field/option write and retarget through the controller.
        Report local and shared validity to the ledger until the view closes.
        """
        try:
            form = self._ctrl.get_writeback_item_form_for_pane(
                self._tab_id, self._pane, item.session_id
            )
        except Exception as exc:
            logger.exception(
                "failed to resolve writeback draft item %s", item.session_id
            )
            _show_module_refusal(self._parent, item, str(exc))
            return
        existing = self._module_dialogs.get(item.session_id)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return

        def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
            self._edit_module_form(
                item.session_id, lambda form: form.state.write(batch, form.env)
            )

        def run_command(path: FormWritePath, name: str) -> None:
            self._edit_module_form(
                item.session_id,
                lambda form: form.state.run_command(path, name, form.env),
            )

        def retarget(name: str) -> None:
            self._ctrl.set_writeback_item_for_pane(
                self._tab_id, self._pane, item.session_id, target_name=name
            )
            item.target_name = name
            refresh_preview()

        def preview() -> WritebackValues:
            return self._ctrl.writeback_control.get_writeback_values_for_pane(
                self._tab_id, self._pane, session_id=item.session_id
            )[item.session_id]

        editor = FormEditor(form.state, form.env, write=write, run_command=run_command)
        dialog = _ModuleDialog(self._parent, item, editor, retarget)

        def refresh_preview(*_: object) -> None:
            valid = editor.errors() is None and not isinstance(
                form.result(), FormErrors
            )
            refresh(valid=valid)
            dialog.refresh_feedback(preview() if valid else None)

        editor.validity_changed.connect(refresh_preview)
        unsubscribe = form.state.subscribe(refresh_preview)
        dialog.destroyed.connect(unsubscribe)
        self._module_dialogs[item.session_id] = dialog
        refresh_preview()
        dialog.adjustSize()

        def on_finished(*_: object) -> None:
            if self._module_dialogs.get(item.session_id) is not dialog:
                return
            unsubscribe()
            editor.validity_changed.disconnect(refresh_preview)
            editor.detach()
            self._module_dialogs.pop(item.session_id)
            refresh(valid=not isinstance(form.result(), FormErrors))

        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(on_finished)
        dialog.open()


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


@dataclass(frozen=True)
class ModuleChange:
    """One changed JSON-like leaf, formatted for library comparison views.

    path is a dotted field/key/index path. current and proposed are display
    strings (full numeric precision); an absent leaf is shown as an em dash.
    """

    path: str
    current: str
    proposed: str

    def text(self) -> str:
        """Return the shared dialog/ledger change line."""
        return f"{self.path}  {self.current} → {self.proposed}"


def _module_leaves(value: object, path: str = "") -> dict[str, object]:
    if isinstance(value, dict) and value:
        leaves: dict[str, object] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            leaves.update(_module_leaves(child, child_path))
        return leaves
    if isinstance(value, (list, tuple)) and value:
        leaves = {}
        for index, child in enumerate(value):
            child_path = f"{path}.{index}" if path else str(index)
            leaves.update(_module_leaves(child, child_path))
        return leaves
    return {path: value}


def module_changes(current: object, proposed: object) -> list[ModuleChange]:
    """Compare JSON-like library entry previews by leaf, in sorted path order.

    Dict keys and sequence indexes become dotted paths. Empty containers are
    leaves; additions/removals display an em dash. Equal values are omitted.
    A missing library entry (current=None) is reported by the caller separately.
    No preview lookup, draft mutation, or unit conversion occurs here.
    """
    before = {} if current is None else _module_leaves(current)
    after = _module_leaves(proposed)
    changes: list[ModuleChange] = []
    for path in sorted(before.keys() | after.keys()):
        if path in before and path in after and before[path] == after[path]:
            continue
        changes.append(
            ModuleChange(
                path,
                repr(before[path]) if path in before else "—",
                repr(after[path]) if path in after else "—",
            )
        )
    return changes


def _bounded_lines(lines: list[str], limit: int) -> str:
    shown = lines[:limit]
    if len(lines) > limit:
        shown.append(f"… and {len(lines) - limit} more")
    return "\n".join(shown)


def _show_module_refusal(parent: QWidget, item: ModuleWriteback, message: str) -> None:
    dialog = _EditDialog(parent)
    dialog.setWindowTitle(f"Edit module — {item.target_name}")
    dialog.setMinimumWidth(520)
    layout = QVBoxLayout(dialog)
    layout.addWidget(dialog.context_label(item.description))
    error = dialog.context_label(message)
    error.setObjectName("writebackEditError")
    dialog.theme_label(error, error_colour)
    layout.addWidget(error)
    buttons = QHBoxLayout()
    buttons.addStretch(1)
    done = QPushButton("Done")
    done.clicked.connect(dialog.accept)
    buttons.addWidget(done)
    layout.addLayout(buttons)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.open()


class _ModuleDialog(_EditDialog):
    def __init__(
        self,
        parent: QWidget,
        item: ModuleWriteback,
        editor: FormEditor[FormModel],
        retarget: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit module — {item.target_name}")
        self.setMinimumSize(520, 420)
        self._item = item
        self._editor = editor
        self._retarget = retarget
        self._retarget_error = ""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(self.context_label(item.description))
        self._path = self.context_label("")
        self.theme_label(self._path, secondary_colour)
        layout.addWidget(self._path)
        hint = self.context_label(
            "Changes go to the writeback draft at once.\nThe library is only written by Apply Selected."
        )
        self.theme_label(hint, secondary_colour)
        layout.addWidget(hint)
        name_row = QFormLayout()
        name_row.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._name = QLineEdit(item.target_name)
        self._name.setObjectName("writebackApplyAs")
        name_row.addRow("Apply as", self._name)
        layout.addLayout(name_row)
        self._changes_group = QGroupBox()
        changes_layout = QVBoxLayout(self._changes_group)
        self._changes_title = QLabel("Changes from library")
        font = self._changes_title.font()
        font.setBold(True)
        self._changes_title.setFont(font)
        changes_layout.addWidget(self._changes_title)
        self._changes = self.context_label("")
        self._changes.setObjectName("writebackModuleChanges")
        changes_layout.addWidget(self._changes)
        layout.addWidget(self._changes_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(editor)
        layout.addWidget(scroll, stretch=1)
        self._error = self.context_label("")
        self._error.setObjectName("writebackEditError")
        self.theme_label(self._error, error_colour)
        layout.addWidget(self._error)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        done = QPushButton("Done")
        # Enter finishes a field, not the dialog, like the shared FormEditor.
        done.setAutoDefault(False)
        done.clicked.connect(self.accept)
        buttons.addWidget(done)
        layout.addLayout(buttons)
        self._name.editingFinished.connect(self._commit_name)

    def _commit_name(self) -> bool:
        text = self._name.text().strip()
        if not text:
            self._name.setText(self._item.target_name)
            self._retarget_error = ""
            self._refresh_errors()
            return True
        if text == self._item.target_name:
            return True
        try:
            self._retarget(text)
        except Exception as exc:
            logger.exception(
                "failed to retarget writeback module %s", self._item.session_id
            )
            self._retarget_error = str(exc)
            self._refresh_errors()
            return False
        self._retarget_error = ""
        self._refresh_errors()
        return True

    def done(self, a0: int) -> None:
        if self._commit_name():
            super().done(a0)

    def refresh_feedback(self, values: WritebackValues | None) -> None:
        if values is None:
            self._path.setText("Writes to library  Unavailable (invalid configuration)")
            self._changes_title.setText("Changes from library")
            self._changes.setText("Unavailable while the configuration is invalid")
            self._changes.setToolTip("")
        else:
            self._path.setText(f"Writes to library  {values.actual_path}")
            changes = module_changes(values.current, values.proposed)
            self._changes_title.setText(f"Changes from library ({len(changes)})")
            lines = [change.text() for change in changes]
            self._changes.setText(
                "New library entry"
                if values.current is None
                else _bounded_lines(lines, 8)
                if lines
                else "No changes"
            )
            self._changes.setToolTip("\n".join(lines))
        self._refresh_errors()

    def _refresh_errors(self) -> None:
        errors = self._editor.errors()
        lines = (
            []
            if errors is None
            else [
                f"{'.'.join(map(str, path)) or 'Configuration'}: {message}"
                for path, message in errors.messages.items()
            ]
        )
        text = _bounded_lines(lines, 5)
        if self._retarget_error:
            text = "\n".join(filter(None, [self._retarget_error, text]))
        if self._editor.local_parse_errors() is not None:
            text += "\nInvalid text is discarded when you press Done."
        self._error.setText(text)
        self._error.setToolTip("\n".join(lines))
        self._error.setVisible(bool(text))


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
        self._add_actions(layout)
        self._name.textChanged.connect(self._validate)
        self._value.textChanged.connect(self._validate)
        self._validate()

    def _add_actions(self, layout: QVBoxLayout) -> None:
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

    def _parsed(self) -> tuple[str, Any]:
        name = _require_target_name(self._name.text())
        text = self._value.text()
        if self._unit and text.rstrip().endswith(self._unit):
            text = text.rstrip()[: -len(self._unit)].rstrip()
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
