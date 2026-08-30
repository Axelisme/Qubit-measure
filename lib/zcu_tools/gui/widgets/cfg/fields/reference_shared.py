"""Shared reference editor authority for form and tree.

Single source for ReferenceField combo population, selection handling,
missing-reference hint, and invalid styling. Both ``ReferenceWidget`` (form)
and the tree's reference header use these helpers so the registry remains the
sole editor authority (A4/S1).
"""

from __future__ import annotations

import logging
from typing import cast

from qtpy.QtWidgets import QComboBox, QLabel, QToolButton  # type: ignore[attr-defined]

from zcu_tools.gui.cfg import is_custom_reference_key, make_custom_reference_key
from zcu_tools.gui.cfg.binding import ReferenceField

logger = logging.getLogger(__name__)

NONE_KEY = "<None>"


def refresh_reference_combo(combo: QComboBox, field: ReferenceField) -> None:
    """Populate ``combo`` exactly like the validated form renderer."""
    combo.blockSignals(True)
    combo.clear()
    current = field.get_chosen_key()
    if field.spec.optional:
        combo.addItem("None", NONE_KEY)
        combo.insertSeparator(combo.count())
    for spec in field.spec.allowed:
        label = spec.label or "Custom"
        key = make_custom_reference_key(label)
        combo.addItem(label, key)
    compatible = field.available_keys()
    if compatible:
        combo.insertSeparator(combo.count())
        for name in compatible:
            if name == current and field.is_modified():
                combo.addItem(f"Lib: {name} (modified)", name)
                combo.addItem(f"Revert to Lib: {name}", name)
            else:
                combo.addItem(f"Lib: {name}", name)
    if field.spec.optional and not field.is_enabled:
        combo.setCurrentIndex(0)
    else:
        idx = combo.findData(current)
        if idx < 0 and field.has_missing_library_ref():
            combo.addItem(f"Missing: {current}", current)
            idx = combo.findData(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    combo.blockSignals(False)


def handle_reference_combo_change(field: ReferenceField, key: object) -> None:
    """Apply a combo selection to ``field`` (mirrors ReferenceWidget)."""
    if key == NONE_KEY:
        field.set_enabled(False)
        return
    if field.spec.optional and not field.is_enabled:
        field.set_enabled(True)
    # ``key`` is expected to be str from combo data
    field.set_chosen_key(cast(str, key))


def refresh_missing_hint(label: QLabel, field: ReferenceField) -> None:
    if field.has_missing_library_ref():
        key = field.get_chosen_key()
        label.setText(
            f"Missing library reference: {key}. "
            "Switch key, or re-add an entry of that name to re-link."
        )
        label.setVisible(True)
        return
    label.setVisible(False)


def apply_reference_validity(
    combo: QComboBox,
    expand_btn: QToolButton | None,
    field: ReferenceField,
    valid: bool,
) -> None:
    style = "" if valid else "border: 1px solid red;"
    combo.setStyleSheet(style)
    if expand_btn is not None:
        expand_btn.setStyleSheet("" if valid else "color: red;")
    # Ensure missing hint visibility matches current field state
    # (caller should also call refresh_missing_hint if needed)
    logger.debug(
        "ReferenceShared.validity_changed: key=%r valid=%r",
        field.get_chosen_key(),
        valid,
    )
