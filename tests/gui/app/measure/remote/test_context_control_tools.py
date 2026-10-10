"""Context/value remote handlers dispatch through ContextControlPort."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from zcu_tools.gui.session.value_lookup import ValueInfo

from ._helpers import dispatch_handler as _dispatch


def _ctrl_with_context_control(context_control: MagicMock) -> SimpleNamespace:
    ctrl = SimpleNamespace(
        context_control=context_control, resources_versions=lambda: {}
    )
    for name in (
        "get_current_md",
        "get_current_ml",
        "set_md_attr",
        "del_md_attr",
        "rename_ml_module",
        "rename_ml_waveform",
        "del_ml_module",
        "del_ml_waveform",
        "list_value_sources",
        "read_value_source",
    ):
        setattr(ctrl, name, MagicMock())
    return ctrl


def test_context_md_ml_and_value_handlers_use_context_control_facet() -> None:
    ctx = MagicMock()
    ctx.list_value_sources.return_value = (
        ValueInfo("predictor.loaded", bool, "predictor", "Whether loaded."),
    )
    ctx.read_value_source.return_value = (
        ValueInfo("predictor.loaded", bool, "predictor", "Whether loaded."),
        True,
    )
    ctrl = _ctrl_with_context_control(ctx)

    assert _dispatch(ctrl, "value.list", {}) == {
        "values": [
            {
                "key": "predictor.loaded",
                "type": "bool",
                "owner": "predictor",
                "description": "Whether loaded.",
            }
        ]
    }
    assert _dispatch(ctrl, "value.read", {"key": "predictor.loaded"}) == {
        "key": "predictor.loaded",
        "type": "bool",
        "owner": "predictor",
        "description": "Whether loaded.",
        "value": True,
    }

