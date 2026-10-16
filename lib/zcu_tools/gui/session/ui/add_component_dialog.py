"""Add component dialog: one scrollable form per registered component kind."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import QPoint, Qt
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
    edit_choices,
    edit_completions,
    field_tooltip,
    parse_field_input,
    set_error_foreground,
    show_error,
)
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.resources.entry import (
    FieldSpec,
    component_field_specs,
    component_registry,
)

if TYPE_CHECKING:
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.resources.entry.schema import PointDocument, SetupDocument


logger = logging.getLogger(__name__)

_SCREEN_HEIGHT_SHARE = 0.9
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
    to: setup template"). Below Name and Kind, a scroll area lists only required
    leaf fields whose enclosing groups are all required, using full dotted-path
    labels without group headers or indentation. Kinds without these fields show
    "No required fields. Edit the others in Inspect after adding." Each field
    shows its unit and its description as a tooltip. Choice and boolean fields
    use a drop-down whose blank first entry means unset. A role reference field
    (FieldSpec.reference_role) drops down the destination document's components
    that match the role. Other text fields complete component names.
    Changing Kind rebuilds the
    fields. Required fields that are still empty have red labels.

    Add sends only filled fields, nested by their segments, to
    control.add_component. Empty Name or required fields disable Add; its tooltip
    lists missing paths. Parse errors stay in red inside the dialog; every other
    check belongs to the owner. On success the dialog calls
    on_added(name) and accepts; on failure the error stays in red inside
    the dialog and every input is kept. The dialog height never exceeds 90%
    of its screen's available height; the fields scroll instead, and Add and
    Cancel stay outside the scroll area. First show and Kind changes fit and
    reposition the entire window frame inside that screen's available geometry.
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
        dialog per Add. Finished retires these inputs; late submissions cannot
        write after Cancel or Close.
        """
        super().__init__(parent)
        self._control = control
        self._retired = False
        self.finished.connect(self._retire_inputs)
        self._setup = setup
        self._on_added = on_added
        self._rows: list[_FieldRow] = []
        self.setWindowTitle("Add component")
        layout = QVBoxLayout(self)

        destination = QLabel(self._destination_text())
        make_label_selectable(destination)
        destination.setObjectName("add_destination")
        destination.setWordWrap(True)
        layout.addWidget(destination)

        content = QWidget(self)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content, stretch=1)
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
        content_layout.addLayout(identity)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("add_fields")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content_layout.addWidget(self._scroll, stretch=1)

        self._error = QLabel()
        self._error.setObjectName("add_error")
        self._error.setWordWrap(True)
        make_label_selectable(self._error)
        set_error_foreground(self._error, error=True)
        self._error.setVisible(False)
        layout.addWidget(self._error)

        layout.addWidget(self._build_buttons())

        self._kind.currentTextChanged.connect(self._rebuild_fields)
        self._rebuild_fields()

    def _retire_inputs(self) -> None:
        """Discard submissions delivered after this Add interaction closes."""
        self._retired = True

    def _build_buttons(self) -> QDialogButtonBox:
        buttons = QDialogButtonBox()
        self._add = QPushButton("Add")
        self._add.setObjectName("add_submit")
        self._add.setDefault(True)
        self._add.clicked.connect(self._on_add)
        buttons.addButton(self._add, QDialogButtonBox.ButtonRole.AcceptRole)
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
            # The tree disables Add without a saved point, so this is a caller bug.
            raise RuntimeError("Adding to the current point requires a selected point")
        return f"Adds to: current point {label}"

    # ------------------------------------------------------------------
    # Field form
    # ------------------------------------------------------------------

    def _rebuild_fields(self) -> None:
        """Replace the scroll area's form with one built for the selected kind."""
        self._rows.clear()
        form = QWidget()
        grid = QGridLayout(form)
        grid.setColumnStretch(1, 1)
        kind = self._kind.currentText()
        specs = component_field_specs(kind) if kind else ()
        groups = {spec.segments: spec for spec in specs if spec.value_kind == "group"}
        required = [
            spec
            for spec in specs
            if spec.value_kind != "group"
            and spec.required
            and all(
                groups[spec.segments[:depth]].required
                for depth in range(1, len(spec.segments))
            )
        ]
        # Reference choices and name completion come from the destination document.
        document = self._control.inspect_document(setup=self._setup)
        for row, spec in enumerate(required):
            label = QLabel(f"{spec.path} *")
            label.setObjectName(f"label:{spec.path}")
            label.setIndent(0)
            label.setToolTip(field_tooltip(spec, spec.path))
            editor = self._editor(spec, document)
            unit = QLabel(spec.unit or "")
            unit.setObjectName(f"unit:{spec.path}")
            grid.addWidget(label, row, 0)
            grid.addWidget(editor, row, 1)
            grid.addWidget(unit, row, 2)
            self._rows.append(_FieldRow(spec, label, editor))
        if not required:
            hint = QLabel(
                "No required fields. Edit the others in Inspect after adding."
            )
            hint.setObjectName("add_fields_hint")
            grid.addWidget(hint, 0, 0, 1, 3)
        grid.setRowStretch(max(1, len(required)), 1)
        # setWidget deletes the previous form together with its inputs.
        self._scroll.setWidget(form)
        self._scroll.updateGeometry()
        self._update_required()
        if self.isVisible():
            self._fit_to_screen()

    def _editor(
        self, spec: FieldSpec, document: SetupDocument | PointDocument
    ) -> QLineEdit | QComboBox:
        choices = edit_choices(document, spec)
        editor: QLineEdit | QComboBox
        if choices:
            editor = QComboBox()
            editor.addItem("")  # Blank first entry: the field stays unset.
            editor.addItems(list(choices))
            editor.currentTextChanged.connect(self._update_required)
        else:
            editor = QLineEdit()
            editor.textChanged.connect(self._update_required)
            completions = edit_completions(document, spec)
            if completions:
                editor.setCompleter(QCompleter(list(completions), editor))
        editor.setObjectName(f"field:{spec.path}")
        editor.setToolTip(field_tooltip(spec, spec.path))
        return editor

    def _update_required(self) -> None:
        missing = [] if self._name.text() else ["name"]
        set_error_foreground(self._name_label, error=not self._name.text())
        for row in self._rows:
            empty = not row.text()
            set_error_foreground(row.label, error=empty)
            if empty:
                missing.append(row.path)
            if isinstance(row.editor, QLineEdit):
                row.editor.setPlaceholderText("Required")
        self._add.setEnabled(not missing)
        self._add.setToolTip(f"Missing: {', '.join(missing)}" if missing else "")

    def _fit_to_screen(self) -> None:
        """Fit the form and keep the whole frame inside the available screen."""
        screen = self.screen()
        if screen is None:
            raise RuntimeError("Add component dialog has no screen")
        available = screen.availableGeometry()
        limit = int(available.height() * _SCREEN_HEIGHT_SHARE)
        self.setMaximumHeight(limit)
        form = self._scroll.widget()
        if form is None:
            raise RuntimeError("Add component dialog has no field form")
        frame = 2 * self._scroll.frameWidth()
        scrollbar = self._scroll.verticalScrollBar()
        bar_width = scrollbar.sizeHint().width() if scrollbar is not None else 0
        layout = self.layout()
        if layout is None:
            raise RuntimeError("Add component dialog has no layout")
        border_width = self.frameGeometry().width() - self.width()
        margins = layout.contentsMargins()
        field_width_limit = (
            available.width() - border_width - margins.left() - margins.right()
        )
        self._scroll.setMinimumWidth(
            min(form.sizeHint().width() + frame + bar_width, field_width_limit)
        )
        layout.activate()
        chrome = (
            self.minimumSizeHint().height() - self._scroll.minimumSizeHint().height()
        )
        wanted = chrome + form.sizeHint().height() + frame
        border_height = self.frameGeometry().height() - self.height()
        width = min(
            max(self.width(), self.sizeHint().width(), _MIN_WIDTH_PX),
            available.width() - border_width,
        )
        self.resize(width, min(limit, wanted, available.height() - border_height))
        frame_geometry = self.frameGeometry()
        left = max(
            available.left(),
            min(frame_geometry.left(), available.right() - frame_geometry.width() + 1),
        )
        top = max(
            available.top(),
            min(frame_geometry.top(), available.bottom() - frame_geometry.height() + 1),
        )
        self.move(self.pos() + QPoint(left, top) - frame_geometry.topLeft())

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        # Submits delivered after close belong to retired inputs.
        if self._retired:
            return
        self._update_required()
        if not self._add.isEnabled():
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
