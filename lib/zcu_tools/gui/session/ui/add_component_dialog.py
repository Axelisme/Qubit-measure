"""Add component dialog: one scrollable form per registered component kind."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtGui import QShowEvent
from qtpy.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.gui.session.ui.component_editing import (
    field_choices,
    parse_field_input,
    set_red,
    show_error,
)
from zcu_tools.resources.entry import (
    FieldSpec,
    component_field_specs,
    component_registry,
    role_candidates,
)

if TYPE_CHECKING:
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.resources.entry.schema import PointDocument, SetupDocument


logger = logging.getLogger(__name__)

_SCREEN_HEIGHT_SHARE = 0.9
_INDENT_PX = 18
_MIN_WIDTH_PX = 480


@dataclass(frozen=True)
class _FieldRow:
    """One leaf field of the form: its spec, name label and input widget."""

    spec: FieldSpec
    label: QLabel
    editor: QLineEdit | QComboBox

    @property
    def path(self) -> str:
        return ".".join(self.spec.segments)

    def text(self) -> str:
        """Current input text; "" means unset (blank drop-down entry or empty)."""
        if isinstance(self.editor, QComboBox):
            return self.editor.currentText()
        return self.editor.text()


class AddComponentDialog(QDialog):
    """Collect a new component's name, kind and fields, then add it via the owner.

    The dialog states its destination ("Adds to: current point P1" or "Adds
    to: setup template"). Below Name and Kind, a scroll area lists every field
    of component_field_specs(kind): group fields are indented header rows
    with their fields below; each field shows its unit and its description as
    a tooltip. Choice and boolean fields use a drop-down whose blank first
    entry means unset. A role reference field (FieldSpec.reference_role)
    drops down the destination document's components that match the role;
    other text fields complete component names. Changing Kind rebuilds the
    fields. Required fields that are still empty have red labels.

    Add sends only filled fields, nested by their segments, to
    control.add_component. Empty required fields block Add locally; every
    other check belongs to the owner. On success the dialog calls
    on_added(name) and accepts; on failure the error stays in red inside
    the dialog and every input is kept. The dialog height never exceeds 90%
    of its screen's available height; the fields scroll instead, and Add and
    Cancel stay outside the scroll area.
    """

    def __init__(
        self,
        control: ContextControlPort,
        *,
        setup: bool,
        on_added: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        """Build the form for the first registered kind; do not show it yet.

        setup=True adds to the opened entry's setup template, False to the
        current point. on_added receives the new component's name after the
        owner accepted it, before the dialog closes. The destination text
        reads control.current_point_label() once, here; callers open a new
        dialog per Add.
        """
        super().__init__(parent)
        self._control = control
        self._setup = setup
        self._on_added = on_added
        self._rows: list[_FieldRow] = []
        self._groups: dict[tuple[str, ...], FieldSpec] = {}
        self.setWindowTitle("Add component")
        layout = QVBoxLayout(self)

        destination = QLabel(self._destination_text())
        destination.setObjectName("add_destination")
        destination.setWordWrap(True)
        layout.addWidget(destination)

        identity = QFormLayout()
        self._name_label = QLabel("Name *")
        self._name_label.setObjectName("label:add_name")
        self._name = QLineEdit()
        self._name.setObjectName("add_name")
        self._name.setPlaceholderText("Required")
        self._name.textChanged.connect(self._update_required)
        identity.addRow(self._name_label, self._name)
        self._kind = QComboBox()
        self._kind.setObjectName("add_kind")
        self._kind.addItems(list(component_registry.kinds()))
        identity.addRow("Kind", self._kind)
        layout.addLayout(identity)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("add_fields")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._scroll, stretch=1)

        self._error = QLabel()
        self._error.setObjectName("add_error")
        self._error.setWordWrap(True)
        self._error.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        set_red(self._error, red=True)
        self._error.setVisible(False)
        layout.addWidget(self._error)

        layout.addWidget(self._build_buttons())

        self._kind.currentTextChanged.connect(self._rebuild_fields)
        self._rebuild_fields()

    def _build_buttons(self) -> QDialogButtonBox:
        buttons = QDialogButtonBox()
        add = QPushButton("Add")
        add.setObjectName("add_submit")
        add.setDefault(True)
        add.clicked.connect(self._on_add)
        buttons.addButton(add, QDialogButtonBox.ButtonRole.AcceptRole)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("add_cancel")
        cancel.clicked.connect(self.reject)
        buttons.addButton(cancel, QDialogButtonBox.ButtonRole.RejectRole)
        return buttons

    def showEvent(self, a0: QShowEvent | None) -> None:
        super().showEvent(a0)
        self._fit_to_screen()

    def _destination_text(self) -> str:
        if self._setup:
            return "Adds to: setup template"
        label = self._control.current_point_label()
        if label is None:
            return "Adds to: current point (unsaved draft)"
        return f"Adds to: current point {label}"

    # ------------------------------------------------------------------
    # Field form
    # ------------------------------------------------------------------

    def _rebuild_fields(self) -> None:
        """Replace the scroll area's form with one built for the selected kind."""
        self._rows.clear()
        self._groups.clear()
        form = QWidget()
        grid = QGridLayout(form)
        grid.setColumnStretch(1, 1)
        kind = self._kind.currentText()
        specs = component_field_specs(kind) if kind else ()
        # Reference choices and name completion come from the destination document.
        document = self._control.inspect_document(setup=self._setup)
        for row, spec in enumerate(specs):
            label = QLabel(spec.segments[-1])
            label.setObjectName(f"label:{'.'.join(spec.segments)}")
            label.setIndent(_INDENT_PX * (len(spec.segments) - 1))
            label.setToolTip(_tooltip(spec))
            if spec.value_kind == "group":
                font = label.font()
                font.setBold(True)
                label.setFont(font)
                self._groups[spec.segments] = spec
                grid.addWidget(label, row, 0, 1, 3)
                continue
            editor = self._editor(spec, document)
            unit = QLabel(spec.unit or "")
            unit.setObjectName(f"unit:{'.'.join(spec.segments)}")
            grid.addWidget(label, row, 0)
            grid.addWidget(editor, row, 1)
            grid.addWidget(unit, row, 2)
            self._rows.append(_FieldRow(spec, label, editor))
        grid.setRowStretch(len(specs), 1)
        # setWidget deletes the previous form together with its inputs.
        self._scroll.setWidget(form)
        self._update_required()
        if self.isVisible():
            self._fit_to_screen()

    def _editor(
        self, spec: FieldSpec, document: SetupDocument | PointDocument
    ) -> QLineEdit | QComboBox:
        path = ".".join(spec.segments)
        if spec.reference_role is not None:
            choices = role_candidates(document, spec.reference_role)
        else:
            choices = field_choices(spec.value_kind, spec.choices)
        editor: QLineEdit | QComboBox
        if choices:
            editor = QComboBox()
            editor.addItem("")  # Blank first entry: the field stays unset.
            editor.addItems(list(choices))
            editor.currentTextChanged.connect(self._update_required)
        else:
            editor = QLineEdit()
            editor.textChanged.connect(self._update_required)
            if spec.value_kind == "text" and document.components:
                names = list(document.components)
                editor.setCompleter(QCompleter(names, editor))
        editor.setObjectName(f"field:{path}")
        editor.setToolTip(_tooltip(spec))
        return editor

    def _required_now(self, spec: FieldSpec) -> bool:
        """A required field counts once every enclosing group is supplied.

        A group is supplied when it is itself required or any field inside it
        is filled; an untouched optional group is not sent at all.
        """
        if not spec.required:
            return False
        for depth in range(1, len(spec.segments)):
            prefix = spec.segments[:depth]
            group = self._groups[prefix]
            if not group.required and not self._group_filled(prefix):
                return False
        return True

    def _group_filled(self, prefix: tuple[str, ...]) -> bool:
        return any(
            row.text() and row.spec.segments[: len(prefix)] == prefix
            for row in self._rows
        )

    def _update_required(self) -> None:
        set_red(self._name_label, red=not self._name.text())
        for row in self._rows:
            required = self._required_now(row.spec)
            row.label.setText(f"{row.spec.segments[-1]}{' *' if required else ''}")
            set_red(row.label, red=required and not row.text())
            if isinstance(row.editor, QLineEdit):
                row.editor.setPlaceholderText("Required" if required else "(unset)")

    def _fit_to_screen(self) -> None:
        """Size to the form, capped at 90% of the screen's available height."""
        screen = self.screen()
        if screen is None:
            raise RuntimeError("Add component dialog has no screen")
        limit = int(screen.availableGeometry().height() * _SCREEN_HEIGHT_SHARE)
        self.setMaximumHeight(limit)
        form = self._scroll.widget()
        if form is None:
            raise RuntimeError("Add component dialog has no field form")
        frame = 2 * self._scroll.frameWidth()
        scrollbar = self._scroll.verticalScrollBar()
        bar_width = scrollbar.sizeHint().width() if scrollbar is not None else 0
        self._scroll.setMinimumWidth(form.sizeHint().width() + frame + bar_width)
        chrome = self.sizeHint().height() - self._scroll.sizeHint().height()
        wanted = chrome + form.sizeHint().height() + frame
        width = max(self.width(), self.sizeHint().width(), _MIN_WIDTH_PX)
        self.resize(width, min(limit, wanted))

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        self._update_required()
        missing = [
            row.path
            for row in self._rows
            if self._required_now(row.spec) and not row.text()
        ]
        if not self._name.text():
            missing.insert(0, "name")
        if missing:
            self._error.setText("\n".join(f"{path}: Required" for path in missing))
            self._error.setToolTip("")
            self._error.setVisible(True)
            return
        fields: YamlMap = {}
        for row in self._rows:
            raw = row.text()
            if not raw:
                continue
            try:
                value = parse_field_input(row.spec.value_kind, raw)
            except ValueError as exc:
                show_error(self._error, exc, field=row.path)
                return
            _insert(fields, row.spec.segments, value)
        name = self._name.text()
        try:
            self._control.add_component(
                name, self._kind.currentText(), fields, setup=self._setup
            )
        except Exception as exc:
            # GUI isolation point: show the owner's rejection, keep every input.
            logger.exception("Component addition failed")
            show_error(self._error, exc)
            return
        self._on_added(name)
        self.accept()


def _insert(fields: YamlMap, segments: tuple[str, ...], value: YamlValue) -> None:
    """Place value at segments inside fields, creating group mappings on the way."""
    node = fields
    for segment in segments[:-1]:
        nested = node.setdefault(segment, {})
        if not isinstance(nested, dict):
            raise TypeError(f"{'.'.join(segments)}: ancestor {segment!r} is a value")
        node = nested
    node[segments[-1]] = value


def _tooltip(spec: FieldSpec) -> str:
    path = ".".join(spec.segments)
    return path if spec.description is None else f"{spec.description}\n{path}"
