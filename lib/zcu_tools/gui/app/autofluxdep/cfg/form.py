"""Cfg-form seam for autofluxdep's typed node cfg editor.

The sibling ``cfg/__init__`` seam re-exports the pure spec/value data model; this
module re-exports the shared reactive Qt form and LiveModel types used by the
node detail pane. Autofluxdep supplies a decoration provider for generated fields,
but the rendering contract lives in the shared cfg form.
"""

from __future__ import annotations

from zcu_tools.gui.app.main.live_model import (
    LiveModelEnv,
    ScalarLiveField,
    SectionLiveField,
)
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
    "LiveModelEnv",
    "ScalarLiveField",
    "ScalarWidget",
    "SectionLiveField",
    "Tone",
    "default_decoration_for_spec",
]
