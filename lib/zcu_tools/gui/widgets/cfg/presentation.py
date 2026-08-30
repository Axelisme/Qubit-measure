"""Shared presentation policy for cfg Qt widgets (form + tree).

Single source for choice visibility, literal hiding, label decoration, and
full decoration application. Both structural adapters import from here so the
decoration contract in ``lib/zcu_tools/gui/README.md#Shared Qt Cfg Widgets``
is implemented once.
"""

from __future__ import annotations

import logging
from typing import cast

from qtpy.QtGui import QBrush, QColor  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QLabel,
    QTreeWidgetItem,
    QWidget,
)

from zcu_tools.gui.cfg import ChoiceSectionSpec, DirectValue, LiteralSpec
from zcu_tools.gui.cfg.binding import CfgField, SectionField

from .decoration import FieldDecorationProtocol
from .fields._decoration import (
    apply_decoration,
    apply_widget_decoration,
    decorated_label_text,
)
from .registry import FieldRenderContext

logger = logging.getLogger(__name__)

_TONE_QCOLOR = {
    "muted": QColor("#6b7280"),
    "info": QColor("#2563eb"),
    "warning": QColor("#8a5a00"),
    "error": QColor("#b00020"),
}


def choice_visible_keys(field: SectionField) -> set[str] | None:
    spec = field.spec
    if not isinstance(spec, ChoiceSectionSpec):
        return None
    visible = set(spec.fields)
    for binding in spec.bindings:
        selector = field.fields.get(binding.selector_key)
        value = selector.get_value() if selector is not None else None
        choice = str(value.value) if isinstance(value, DirectValue) else ""
        try:
            active_spec = binding.choices[choice]
        except KeyError as exc:
            expected = ", ".join(sorted(binding.choices))
            raise ValueError(
                f"ChoiceSectionSpec selector {binding.selector_key!r} has unknown "
                f"value {choice!r}; expected one of: {expected}"
            ) from exc
        active = set(active_spec.fields)
        visible -= binding.controlled_field_keys() - active
    return visible


def resolve_decoration(
    path: str, field: CfgField, context: FieldRenderContext
) -> FieldDecorationProtocol | None:
    resolver = context.decoration_for_path
    if resolver is None:
        return None
    try:
        return cast(FieldDecorationProtocol, resolver(path, field))
    except Exception:
        logger.debug("decoration resolver failed for %r", path, exc_info=True)
        return None


def is_hidden(path: str, field: CfgField, context: FieldRenderContext) -> bool:
    decoration = resolve_decoration(path, field, context)
    if isinstance(field.spec, LiteralSpec):
        if decoration is None:
            return True
        return bool(getattr(decoration, "hidden", False))
    if decoration is not None:
        return bool(getattr(decoration, "hidden", False))
    return False


def decorated_label(
    field: CfgField, key: str, path: str, context: FieldRenderContext
) -> str:
    label = getattr(field.spec, "label", "") or key
    decoration = resolve_decoration(path, field, context)
    return decorated_label_text(label, decoration)


def apply_form_row_decoration(
    label_widget: QLabel,
    value_widget: QWidget,
    decoration: FieldDecorationProtocol | None,
) -> None:
    if decoration is None:
        return
    apply_decoration(label_widget, value_widget, decoration)


def apply_form_widget_decoration(
    value_widget: QWidget, decoration: FieldDecorationProtocol | None
) -> None:
    if decoration is None:
        return
    apply_widget_decoration(value_widget, decoration)


def apply_tree_item_decoration(
    item: QTreeWidgetItem,
    control: QWidget | None,
    decoration: FieldDecorationProtocol | None,
) -> None:
    """Apply full decoration contract to a tree row (section/reference/leaf)."""
    if decoration is None:
        return
    enabled = bool(getattr(decoration, "enabled", True))
    tooltip = str(getattr(decoration, "tooltip", "") or "")
    tone = str(getattr(decoration, "tone", "") or "normal")
    # enabled: disable control and mark item as disabled (affects child interactions)
    if not enabled:
        if control is not None:
            control.setEnabled(False)
        try:
            item.setDisabled(True)
        except Exception:
            pass
    if tooltip:
        item.setToolTip(0, tooltip)
        item.setToolTip(1, tooltip)
        if control is not None:
            control.setToolTip(tooltip)
    # tone: for tree, map to foreground color on the label column.
    # Use muted/info/warning/error mapping; normal = no override.
    color = _TONE_QCOLOR.get(tone)
    if color is not None:
        brush = QBrush(color)
        # Apply to both columns for consistency with form's label styling.
        item.setForeground(0, brush)
        item.setForeground(1, brush)
