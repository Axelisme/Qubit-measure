"""Private decoration presentation helpers for cfg field widgets."""

from __future__ import annotations

from qtpy.QtWidgets import QLabel, QWidget  # type: ignore[attr-defined]

from ..decoration import FieldDecorationProtocol

_TONE_STYLES = {
    "muted": "color: #6b7280;",
    "info": "color: #2563eb;",
    "warning": "color: #8a5a00;",
    "error": "color: #b00020;",
}


def decorated_label_text(label: str, decoration: FieldDecorationProtocol | None) -> str:
    if decoration is None:
        return label
    text = f"{label}{decoration.label_suffix}"
    if decoration.badge:
        return f"{text} [{decoration.badge}]"
    return text


def decoration_enabled(decoration: FieldDecorationProtocol | None) -> bool:
    return decoration is None or decoration.enabled


def apply_decoration(
    label_widget: QLabel,
    value_widget: QWidget,
    decoration: FieldDecorationProtocol | None,
) -> None:
    if decoration is None:
        return
    enabled, tooltip, style = _decoration_widget_state(decoration)
    label_widget.setEnabled(enabled)
    value_widget.setEnabled(enabled)
    if tooltip:
        label_widget.setToolTip(tooltip)
        value_widget.setToolTip(tooltip)
    if style:
        label_widget.setStyleSheet(style)


def apply_widget_decoration(
    value_widget: QWidget, decoration: FieldDecorationProtocol | None
) -> None:
    if decoration is None:
        return
    enabled, tooltip, _style = _decoration_widget_state(decoration)
    value_widget.setEnabled(enabled)
    if tooltip:
        value_widget.setToolTip(tooltip)


def _decoration_widget_state(
    decoration: FieldDecorationProtocol,
) -> tuple[bool, str, str]:
    tone = decoration.tone or "normal"
    return (
        decoration.enabled,
        decoration.tooltip,
        _TONE_STYLES.get(tone, ""),
    )
