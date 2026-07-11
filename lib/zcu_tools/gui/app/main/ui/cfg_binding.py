"""Qt enhancers for the measure cfg binding policy."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]

from zcu_tools.gui.session.ui.value_source_input import (
    SessionValueSourceInputHost,
    SessionValueSourcePort,
    ValueSourceInputController,
)

TextInputEnhancer = Callable[[QLineEdit], object | None]


def make_value_source_input_enhancer(
    host: SessionValueSourcePort,
) -> TextInputEnhancer:
    source_host = SessionValueSourceInputHost(host)

    def enhance(line_edit: QLineEdit) -> object:
        controller = ValueSourceInputController(
            line_edit,
            source_host,
            parent=line_edit,
        )
        controller.resolve_failed.connect(line_edit.setToolTip)  # type: ignore[attr-defined]
        return controller

    return enhance


__all__ = ["TextInputEnhancer", "make_value_source_input_enhancer"]
