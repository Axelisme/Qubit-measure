"""Component tree frontend for the session's parameter owner."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import yaml
from pydantic import TypeAdapter
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.resources.entry import component_registry, document_fields

if TYPE_CHECKING:
    from zcu_tools.gui.session.context_control import ContextControlPort


class ComponentTreeWidget(QWidget):
    """Inspect and edit registered components through ContextControlPort.

    The widget owns presentation only. Tree rows show component names, kind,
    fields and working-unit values. Missing required fields are red in the
    creation form. References use ordinary text, not graph-aware pickers.
    A setup/current selector controls editing destination. All writes use the
    service, and errors remain visible without mutating the rejected document.
    """

    def __init__(
        self, control: ContextControlPort, parent: QWidget | None = None
    ) -> None:
        """Bind the owner-thread control and render its current document.

        EMPTY displays no editable rows. Creation chooses a registered kind and
        collects required fields before Add. The current point view permits
        selecting a component as focus; setup mode does not change focus.
        """
        super().__init__(parent)
        self._control = control
        self._field_inputs: dict[str, tuple[QLineEdit, QLabel, bool, bool]] = {}
        self._row_values: dict[str, YamlValue] = {}
        layout = QVBoxLayout(self)
        self._destination = QComboBox()
        self._destination.addItems(["Current point", "Setup template"])
        self._destination.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self._destination)
        self._tree = QTreeWidget()
        self._tree.setObjectName("component_tree")
        self._tree.setHeaderLabels(["Component / field", "Kind / value"])
        self._tree.currentItemChanged.connect(self._on_selection)
        layout.addWidget(self._tree)
        self._error = QLabel()
        self._error.setObjectName("component_error")
        self._error.setWordWrap(True)
        self._mark_red(self._error)
        layout.addWidget(self._error)
        self._build_creation(layout)
        self._build_editing(layout)
        self.refresh()

    def refresh(self) -> None:
        """Render the cached selected document; never implicitly refresh files."""
        self._tree.clear()
        self._row_values.clear()
        self._focus_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        try:
            document = self._control.inspect_document(setup=self._is_setup())
        except FailedPreconditionError as exc:
            self._error.setText(str(exc))
            return
        roots = {
            name: QTreeWidgetItem(self._tree, [name, component.kind])
            for name, component in document.components.items()
        }
        for field in document_fields(document):
            if field.segments[0] not in roots:
                continue
            item = QTreeWidgetItem(
                roots[field.segments[0]],
                [".".join(field.segments[1:]), str(field.value)],
            )
            if field.editable_path is not None:
                self._row_values[field.editable_path] = field.value
                item.setToolTip(0, field.editable_path)
            item.setToolTip(1, repr(field.provenance))
        for root in roots.values():
            root.setExpanded(True)
        self._tree.resizeColumnToContents(0)

    def add_component(
        self, name: str, kind: str, fields: YamlMap, *, setup: bool = False
    ) -> bool:
        """Submit Add and refresh on success, returning True.

        Recoverable validation/resource failures display red text in the
        component_error QLabel and return False. Input fields stay available
        for correction; no default or empty value hides the failure.
        """
        try:
            self._control.add_component(name, kind, fields, setup=setup)
        except Exception as exc:  # noqa: BLE001 — UI isolates and displays command failures
            self._error.setText(str(exc))
            return False
        self._error.clear()
        self.refresh()
        return True

    def edit_parameter(
        self, path: str, value: YamlValue, *, setup: bool = False
    ) -> bool:
        """Edit a concrete field through its owner; report errors as for Add."""
        try:
            if setup:
                self._control.set_setup_parameter(path, value)
            else:
                capture = self._control.read_context()
                self._control.set_parameter(capture.target, path, value)
        except Exception as exc:  # noqa: BLE001 — UI isolates and displays command failures
            self._error.setText(str(exc))
            return False
        self._error.clear()
        self.refresh()
        return True

    def set_focus(self, name: str) -> None:
        """Select a current component through the owner; errors propagate."""
        if not self._is_setup():
            self._control.set_component_focus(name)

    def _is_setup(self) -> bool:
        return self._destination.currentIndex() == 1

    @staticmethod
    def _mark_red(label: QLabel) -> None:
        palette = label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor("red"))
        label.setPalette(palette)

    def _build_creation(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel("New component:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("Name")
        row.addWidget(self._name)
        self._kind = QComboBox()
        self._kind.addItems(list(component_registry.kinds()))
        row.addWidget(self._kind)
        add = QPushButton("Add")
        add.clicked.connect(self._on_add)
        row.addWidget(add)
        layout.addLayout(row)
        self._fields = QFormLayout()
        layout.addLayout(self._fields)
        self._kind.currentTextChanged.connect(self._rebuild_fields)
        self._rebuild_fields()

    def _rebuild_fields(self) -> None:
        while self._fields.rowCount():
            self._fields.removeRow(0)
        self._field_inputs.clear()
        kind = self._kind.currentText()
        if not kind:
            return
        model = component_registry.get(kind)
        for name, field in model.model_fields.items():
            if name == "kind":
                continue
            key = field.alias or name
            required = field.is_required()
            label = QLabel(f"{key}{' *' if required else ''}")
            edit = QLineEdit()
            if required:
                self._mark_red(label)
                edit.setPlaceholderText("Required")
            else:
                edit.setPlaceholderText("Optional (schema default)")
            is_text = field.annotation is str or str in get_args(field.annotation)
            self._field_inputs[key] = (edit, label, required, is_text)
            edit.textChanged.connect(self._update_required_labels)
            self._fields.addRow(label, edit)

    def _update_required_labels(self) -> None:
        for edit, label, required, _is_text in self._field_inputs.values():
            if required and not edit.text():
                self._mark_red(label)
            else:
                label.setPalette(self.palette())

    def _on_add(self) -> None:
        fields: YamlMap = {}
        missing = []
        try:
            for key, (edit, _label, required, is_text) in self._field_inputs.items():
                raw = edit.text()
                if not raw:
                    if required:
                        missing.append(key)
                    continue
                fields[key] = (
                    raw
                    if is_text
                    else TypeAdapter(YamlValue).validate_python(
                        yaml.safe_load(raw), strict=True
                    )
                )
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            self._error.setText(str(exc))
            return
        if missing:
            self._error.setText(f"Required fields missing: {', '.join(missing)}")
            self._update_required_labels()
            return
        self.add_component(
            self._name.text(), self._kind.currentText(), fields, setup=self._is_setup()
        )

    def _build_editing(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self._path = QLineEdit()
        self._path.setReadOnly(True)
        self._path.setPlaceholderText("Select a field to edit")
        self._value = QLineEdit()
        self._value.setPlaceholderText("Working-unit value")
        self._apply_btn = QPushButton("Set field")
        self._apply_btn.clicked.connect(self._on_edit)
        self._focus_btn = QPushButton("Set focus")
        self._focus_btn.clicked.connect(self._on_focus)
        for widget in (self._path, self._value, self._apply_btn, self._focus_btn):
            row.addWidget(widget)
        layout.addLayout(row)

    def _on_selection(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        path = current.toolTip(0) if current is not None else ""
        self._path.setText(path)
        editable = path in self._row_values and not path.endswith(".kind")
        self._apply_btn.setEnabled(editable)
        self._value.setEnabled(editable)
        if editable:
            value = self._row_values[path]
            self._value.setText(
                value
                if isinstance(value, str)
                else yaml.safe_dump(value, default_flow_style=True)
                .removesuffix("...\n")
                .strip()
            )
        else:
            self._value.clear()
        self._focus_btn.setEnabled(current is not None and not self._is_setup())

    def _on_edit(self) -> None:
        path = self._path.text()
        if path not in self._row_values:
            return
        raw = self._value.text()
        try:
            value = (
                raw
                if isinstance(self._row_values[path], str)
                else TypeAdapter(YamlValue).validate_python(
                    yaml.safe_load(raw), strict=True
                )
            )
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            self._error.setText(str(exc))
            return
        self.edit_parameter(path, value, setup=self._is_setup())

    def _on_focus(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        while item.parent() is not None:
            parent = item.parent()
            if parent is None:
                break
            item = parent
        try:
            self.set_focus(item.text(0))
        except Exception as exc:  # noqa: BLE001 — UI isolates and displays focus failures
            self._error.setText(str(exc))
