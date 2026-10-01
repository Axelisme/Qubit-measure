"""Reference choice presentation shared by resource and legacy binding adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from qtpy.QtWidgets import QComboBox, QLabel  # type: ignore[attr-defined]

from zcu_tools.gui.cfg import (
    ReferenceSpec,
    ReferenceValue,
    make_custom_reference_key,
    parse_custom_reference_key,
)
from zcu_tools.gui.cfg.binding.observation import CfgNodeObservation

NONE_KEY = "<None>"


@dataclass(frozen=True)
class CustomReferenceSelection:
    label: str


ReferenceSelection: TypeAlias = CustomReferenceSelection | str | None


def reference_library_keys(node: CfgNodeObservation) -> tuple[str, ...]:
    """Project catalog keys from the cached reference observation, without I/O."""
    spec = node.spec
    if not isinstance(spec, ReferenceSpec):
        raise TypeError("reference options require a ReferenceSpec")
    options = node.options or ()
    labels = tuple(shape.label for shape in spec.allowed)
    if options[: len(labels)] != labels:
        raise ValueError("reference options do not match the definition")
    keys: list[str] = []
    for key in options[len(labels) :]:
        if not isinstance(key, str):
            raise TypeError("reference catalog keys must be strings")
        keys.append(key)
    return tuple(keys)


def display_reference_combo(
    combo: QComboBox,
    spec: ReferenceSpec,
    value: ReferenceValue | None,
    library_keys: tuple[str, ...],
) -> None:
    """Populate one header from a cached value and compatible catalog keys."""
    previous = combo.blockSignals(True)
    try:
        combo.clear()
        current = value.chosen_key if value is not None else None
        if spec.optional:
            combo.addItem("None", NONE_KEY)
            combo.insertSeparator(combo.count())
        for shape in spec.allowed:
            label = shape.label or "Custom"
            combo.addItem(label, make_custom_reference_key(label))
        if library_keys:
            combo.insertSeparator(combo.count())
            for name in library_keys:
                if value is not None and name == current and value.is_overridden:
                    combo.addItem(f"Lib: {name} (modified)", name)
                    combo.addItem(f"Revert to Lib: {name}", name)
                else:
                    combo.addItem(f"Lib: {name}", name)
        if value is None and spec.optional:
            combo.setCurrentIndex(0)
        elif current is not None:
            index = combo.findData(current)
            if index < 0 and value is not None and value.error is not None:
                combo.addItem(f"Missing: {current}", current)
                index = combo.findData(current)
            if index >= 0:
                combo.setCurrentIndex(index)
    finally:
        combo.blockSignals(previous)


def selected_reference(key: object) -> ReferenceSelection:
    if key == NONE_KEY:
        return None
    if not isinstance(key, str):
        raise TypeError("reference choice must be a string")
    label = parse_custom_reference_key(key)
    if label is not None:
        return CustomReferenceSelection(label)
    return key


def display_missing_hint(label: QLabel, value: ReferenceValue | None) -> None:
    if value is not None and value.error is not None:
        label.setText(
            f"Missing library reference: {value.chosen_key}. "
            "Switch key, or re-add an entry of that name to re-link."
        )
        label.setVisible(True)
    else:
        label.setVisible(False)


def display_reference_validity(combo: QComboBox, *, valid: bool) -> None:
    combo.setStyleSheet("" if valid else "border: 1px solid red;")
