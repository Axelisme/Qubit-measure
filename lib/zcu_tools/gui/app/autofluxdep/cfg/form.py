"""Cfg-form seam — the SINGLE allowed import point of the measure-app *cfg form
widget* machinery (``gui.app.main.ui.cfg_form`` + ``gui.app.main.live_model``).

The sibling ``cfg/__init__`` seam re-exports the pure spec/value data model;
this one re-exports the reactive Qt form that renders it: ``CfgFormWidget`` (the
viewer that attaches to a LiveModel) + ``SectionLiveField`` / ``LiveModelEnv``
(the LiveModel layer it renders). autofluxdep's node detail pane builds a
``SectionLiveField`` over a placement's ``NodeCfgSchema`` value tree, wraps it in
a ``CfgFormWidget``, and writes edits back through the controller — exactly the
local-draft pattern measure's inspect / writeback dialogs use.

Keeping the widget import here (and only here) confines the app-to-app coupling
to one file: a future lift of the cfg form into a shared layer only retargets
this seam, not the node detail pane.
"""

from __future__ import annotations

from zcu_tools.gui.app.main.live_model import LiveModelEnv, SectionLiveField
from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

__all__ = [
    "CfgFormWidget",
    "LiveModelEnv",
    "SectionLiveField",
]
