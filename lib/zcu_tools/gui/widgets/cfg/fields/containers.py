"""Container widgets for shared cfg binding sections and references."""

from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.cfg import (
    is_custom_reference_key,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    ReferenceField,
    SectionField,
    SweepField,
)

from ..decoration import FieldDecorationProtocol
from ..registry import FieldRenderContext, FieldWidgetProtocol
from .common import BaseLiveWidget
from .reference_shared import (
    apply_reference_validity,
    handle_reference_combo_change,
    refresh_missing_hint,
    refresh_reference_combo,
)


class _CollapsibleSection(QWidget):
    """Internal helper for collapsible headers."""

    def __init__(
        self,
        label: str,
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

    def _on_toggle(self, checked: bool) -> None:
        if self._toggle_btn:
            self._toggle_btn.setText("▼" if checked else "▶")
        self._body.setVisible(checked)

    def set_invalid(self, invalid: bool) -> None:
        style = "color: red;" if invalid else ""
        if self._header_label is not None:
            self._header_label.setStyleSheet(style)
        if self._toggle_btn is not None:
            self._toggle_btn.setStyleSheet(style)


# SectionWidget removed: sole tree (TreeCfgWidget) owns all section/subtree
# structure. _CollapsibleSection is retained only for non-cfg app usage
# (e.g., feedback panel) and is decoupled from cfg form path.


class ReferenceWidget(BaseLiveWidget):
    """Reference editor for sole tree: combo + missing hint only.

    All section/subtree structure is owned by TreeCfgWidget (shape elision);
    this widget never creates a SectionWidget or sub_container.
    """

    def __init__(
        self,
        field: ReferenceField,
        *,
        context: FieldRenderContext,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(field, parent)
        self._context = context
        self._path = context.path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        self._combo = QComboBox()
        self._refresh_combo_items()
        self._combo.setMinimumWidth(20)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        header.addWidget(self._combo, stretch=1)
        layout.addLayout(header)

        self._missing_ref_hint = QLabel()
        self._missing_ref_hint.setObjectName("missingRefHint")
        self._missing_ref_hint.setStyleSheet("color: #b00020; font-size: 11px;")
        self._missing_ref_hint.setVisible(False)
        layout.addWidget(self._missing_ref_hint)

        self._refresh_missing_ref_hint()

        # Reactive sync
        field.on_change.connect(self._on_model_changed)
        field.on_validity_changed.connect(self._on_validity_changed)
        self._on_validity_changed(field.is_valid())

    _NONE_KEY = "<None>"

    def _refresh_combo_items(self) -> None:
        refresh_reference_combo(self._combo, cast(ReferenceField, self._field))

    def _on_combo_changed(self, index: int) -> None:
        key = self._combo.itemData(index)
        field = cast(ReferenceField, self._field)
        handle_reference_combo_change(field, key)

    def _on_model_changed(self, *_: Any) -> None:
        self._refresh_combo_items()
        self._refresh_missing_ref_hint()

    def _refresh_missing_ref_hint(self) -> None:
        refresh_missing_hint(self._missing_ref_hint, cast(ReferenceField, self._field))

    def refresh_section(self, path: str) -> bool:
        # Section/subtree owned solely by TreeCfgWidget; reference header has no section to refresh.
        del path
        return False

    def teardown(self) -> None:
        field = cast(ReferenceField, self._field)
        field.on_change.disconnect(self._on_model_changed)
        field.on_validity_changed.disconnect(self._on_validity_changed)

    def _on_validity_changed(self, valid: bool) -> None:
        field = cast(ReferenceField, self._field)
        apply_reference_validity(self._combo, None, field, valid)
        self._refresh_missing_ref_hint()
