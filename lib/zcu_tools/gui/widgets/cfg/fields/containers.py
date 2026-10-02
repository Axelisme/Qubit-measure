"""Container widgets for shared cfg binding sections and references."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.cfg import ReferenceSpec, ReferenceValue, make_custom_reference_key
from zcu_tools.gui.cfg.binding import ReferenceField

from ..registry import FieldRenderContext
from .reference_shared import (
    NONE_KEY,
    CustomReferenceSelection,
    ReferenceSelection,
    display_missing_hint,
    display_reference_combo,
    display_reference_validity,
    selected_reference,
)


class CollapsibleSection(QWidget):
    """Shared header and body layout for collapsible app panels."""

    def __init__(
        self,
        label: str,
        *,
        collapsible: bool = True,
        collapsed: bool = False,
        no_header: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = None
        self._header_label: QLabel | None = None

        if not no_header:
            if collapsible:
                header = QWidget()
                header_row = QHBoxLayout(header)
                header_row.setContentsMargins(0, 0, 0, 0)
                header_row.setSpacing(2)

                self._toggle_btn = QPushButton("▼" if not collapsed else "▶")
                self._toggle_btn.setFixedWidth(16)
                self._toggle_btn.setFlat(True)
                self._toggle_btn.setCheckable(True)
                self._toggle_btn.setChecked(not collapsed)
                self._toggle_btn.clicked.connect(self._on_toggle)
                header_row.addWidget(self._toggle_btn)
                self._header_label = QLabel(f"<b>{label}</b>")
                header_row.addWidget(self._header_label, stretch=1)
                outer.addWidget(header)
            else:
                if label:
                    self._header_label = QLabel(f"<b>{label}</b>")
                    outer.addWidget(self._header_label)

        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(8, 2, 0, 2)
        self.body_layout.setSpacing(2)
        outer.addWidget(self._body)

        # For compatibility with old code that expects .form on this widget
        self.form = QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(4)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        self.body_layout.addLayout(self.form)

        if collapsed:
            self._body.setVisible(False)

    def _on_toggle(self, checked: bool) -> None:  # noqa: FBT001 - Qt signal
        if self._toggle_btn:
            self._toggle_btn.setText("▼" if checked else "▶")
        self._body.setVisible(checked)


# SectionWidget removed: sole tree (TreeCfgWidget) owns all section/subtree
# structure. CollapsibleSection is retained only for non-cfg app usage
# (e.g., feedback panel) and is decoupled from cfg form path.


class ReferenceInputWidget(QWidget):
    """Show a published reference choice and submit selection intent only.

    Tree/form owners render the children; this header never owns a cfg subtree.
    """

    _NONE_KEY = NONE_KEY

    def __init__(
        self,
        spec: ReferenceSpec,
        value: ReferenceValue | None,
        parent: QWidget | None = None,
        *,
        library_keys: tuple[str, ...],
        valid: bool,
        submit: Callable[[ReferenceSelection], None],
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self._value = value
        self._library_keys = library_keys
        self._valid = valid
        self._submit = submit

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self._combo = QComboBox()
        self._combo.setMinimumWidth(20)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        header.addWidget(self._combo, stretch=1)
        layout.addLayout(header)
        self._missing_ref_hint = QLabel()
        self._missing_ref_hint.setObjectName("missingRefHint")
        self._missing_ref_hint.setStyleSheet("color: #b00020; font-size: 11px;")
        self._missing_ref_hint.setVisible(False)
        layout.addWidget(self._missing_ref_hint)
        self.display(value, library_keys=library_keys, valid=valid)

    def display(
        self,
        value: ReferenceValue | None,
        *,
        library_keys: tuple[str, ...],
        valid: bool,
    ) -> None:
        """Apply owner publication without submitting a choice."""
        self._value = value
        self._library_keys = library_keys
        self._valid = valid
        display_reference_combo(self._combo, self._spec, value, library_keys)
        display_missing_hint(self._missing_ref_hint, value)
        display_reference_validity(self._combo, valid=valid)

    def _on_combo_changed(self, index: int) -> None:
        choice = selected_reference(self._combo.itemData(index))
        try:
            self._submit(choice)
        except Exception:
            # Rejecting a selection leaves the owner unchanged; restore the
            # last publication rather than keeping an optimistic combo choice.
            self.display(
                self._value, library_keys=self._library_keys, valid=self._valid
            )
            raise


class ReferenceWidget(ReferenceInputWidget):
    """Connect the shared reference header to an existing binding-owned form."""

    def __init__(
        self,
        field: ReferenceField,
        *,
        context: FieldRenderContext,
        parent: QWidget | None = None,
    ) -> None:
        self._field = field
        self._context = context
        self._path = context.path
        super().__init__(
            field.spec,
            field.get_value(),
            parent,
            library_keys=field.available_keys(),
            valid=field.is_valid(),
            submit=self._write_input,
        )
        field.on_change.connect(self._on_model_changed)
        field.on_validity_changed.connect(self._on_validity_changed)

    @property
    def field(self) -> ReferenceField:
        return self._field

    def _write_input(self, choice: ReferenceSelection) -> None:
        field = self._field
        if choice is None:
            field.set_enabled(False)
            return
        if field.spec.optional and not field.is_enabled:
            field.set_enabled(True)
        if isinstance(choice, CustomReferenceSelection):
            field.set_chosen_key(make_custom_reference_key(choice.label))
        else:
            field.set_chosen_key(choice)

    def _on_model_changed(self, *_: Any) -> None:
        self._on_validity_changed(self._field.is_valid())

    def _on_validity_changed(self, valid: bool) -> None:  # noqa: FBT001 - field callback
        self.display(
            self._field.get_value(),
            library_keys=self._field.available_keys(),
            valid=valid,
        )

    def refresh_section(self, path: str) -> bool:
        del path
        return False

    def teardown(self) -> None:
        self._field.on_change.disconnect(self._on_model_changed)
        self._field.on_validity_changed.disconnect(self._on_validity_changed)
