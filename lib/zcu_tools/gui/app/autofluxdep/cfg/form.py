"""Cfg-form seam for autofluxdep's typed node cfg editor.

The sibling ``cfg/__init__`` seam re-exports the pure spec/value data model; this
module re-exports only the measure Qt widgets pending the shared-widget move.
Autoflux binding and field types are imported from their owning packages.
"""

from __future__ import annotations

from zcu_tools.gui.app.main.ui.cfg_form import (
    CfgFormWidget,
    FieldDecoration,
    FieldDecorationPatch,
    FieldDecorationProvider,
    Tone,
    default_decoration_for_spec,
)
from zcu_tools.gui.app.main.ui.fields.common import ScalarWidget

__all__ = [
    "CfgFormWidget",
    "FieldDecoration",
    "FieldDecorationPatch",
    "FieldDecorationProvider",
    "ScalarWidget",
    "Tone",
    "default_decoration_for_spec",
]
