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
from .fields._decoration import (  # single decoration projection (tone/enabled/tooltip)
    _TONE_STYLES,
    _decoration_widget_state,
    apply_decoration,
    apply_widget_decoration,
    decorated_label_text,
)
from .registry import FieldRenderContext

logger = logging.getLogger(__name__)

# Derive QColor map from the single _TONE_STYLES source to keep tone interpretation once.
_TONE_QCOLOR: dict[str, QColor] = {}
for _tone, _style in _TONE_STYLES.items():  # type: ignore[attr-defined]
    # style is "color: #...;" – extract hex
    try:
        _hex = _style.split(":")[1].strip().rstrip(";")
        _TONE_QCOLOR[_tone] = QColor(_hex)
    except Exception:
        continue


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
    # Use single decoration state (enabled/tooltip/tone) so form and tree share projection
    enabled, tooltip, style = _decoration_widget_state(decoration)  # type: ignore[arg-type]
    value_widget.setEnabled(enabled)
    if tooltip:
        value_widget.setToolTip(tooltip)
    if style:
        # For SectionWidget, this tints the header label/button via stylesheet inheritance
        try:
            value_widget.setStyleSheet(style)
        except Exception:
            pass


def apply_tree_item_decoration(
    item: QTreeWidgetItem,
    control: QWidget | None,
    decoration: FieldDecorationProtocol | None,
) -> None:
    """Apply full decoration contract to a tree row (section/reference/leaf)."""
    if decoration is None:
        return
    # Use single decoration state (enabled/tooltip/tone) from _decoration projection
    enabled, tooltip, style = _decoration_widget_state(decoration)  # type: ignore[arg-type]
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
    # tone: map style "color: #...;" to QColor via shared _TONE_QCOLOR (single source)
    # Derive tone from decoration.tone directly to reuse _TONE_QCOLOR
    tone = str(getattr(decoration, "tone", "") or "normal")
    color = _TONE_QCOLOR.get(tone)
    if color is not None:
        brush = QBrush(color)
        item.setForeground(0, brush)
        item.setForeground(1, brush)
    elif style:
        # Fallback: if style is non-empty but tone not in map, still apply via foreground
        # (should not happen as style derived from same tone)
        pass
