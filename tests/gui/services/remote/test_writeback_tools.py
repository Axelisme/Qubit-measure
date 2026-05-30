"""Writeback preview/apply dispatch handlers.

Drives the handlers against a mock Controller whose get_tab_writeback_items
returns crafted items, so we can assert preview serialization and the apply
mutation/dispatch path without a full run+analyze pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    MetaDictWriteback,
    ModuleWriteback,
    ScalarSpec,
)
from zcu_tools.gui.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError


def _edit_schema() -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields={"freq": ScalarSpec(label="Freq", type=float)}),
        value=CfgSectionValue(fields={"freq": DirectValue(6000.0)}),
    )


def _items() -> list:
    return [
        MetaDictWriteback(
            key="r_f",
            description="Resonator freq",
            proposed_value=6012.3,
        ),
        ModuleWriteback(
            key="readout_rf",
            description="readout module",
            edit_schema=_edit_schema(),
        ),
    ]


def _ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_writeback_items.side_effect = lambda tab_id: _items()
    ctrl.apply_writeback_items.return_value = ["r_f"]
    return ctrl


def _dispatch(ctrl, method, params):
    return METHOD_REGISTRY[method].handler(ctrl, params)


def test_preview_serializes_metadict_and_module():
    ctrl = _ctrl()
    res = _dispatch(ctrl, "writeback.preview", {"tab_id": "t"})
    item_list = list(res["items"])  # type: ignore[call-overload]
    items = {it["key"]: it for it in item_list}

    md = items["r_f"]
    assert md["kind"] == "metadict"
    assert md["key"] == "r_f"
    assert "current_value" not in md
    assert "md_key" not in md
    assert md["proposed_value"] == 6012.3

    mod = items["readout_rf"]
    assert mod["kind"] == "module"
    assert mod["key"] == "readout_rf"
    assert "name" not in mod
    assert mod["has_edit_schema"] is True
    # edit_schema serialized as tagged cfg (same format as tab.get_cfg)
    assert "freq" in mod["edit_schema_raw"]


def test_apply_passes_mutated_items_to_controller():
    ctrl = _ctrl()
    res = _dispatch(
        ctrl,
        "writeback.apply",
        {
            "tab_id": "t",
            "selections": [
                {"key": "r_f", "selected": True, "proposed_value": 6015.0},
                {"key": "readout_rf", "selected": False},
            ],
        },
    )
    assert res["applied_keys"] == ["r_f"]
    applied = ctrl.apply_writeback_items.call_args.args[1]
    by_key = {it.key: it for it in applied}
    assert by_key["r_f"].proposed_value == 6015.0
    assert by_key["readout_rf"].selected is False


def test_apply_unknown_key_rejected():
    ctrl = _ctrl()
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "writeback.apply",
            {"tab_id": "t", "selections": [{"key": "nope"}]},
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_apply_proposed_value_on_module_rejected():
    ctrl = _ctrl()
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "writeback.apply",
            {"tab_id": "t", "selections": [{"key": "readout_rf", "proposed_value": 1}]},
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_apply_edited_raw_roundtrips_through_codec():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "writeback.apply",
        {
            "tab_id": "t",
            "selections": [
                {
                    "key": "readout_rf",
                    "edited_raw": {
                        "freq": {"__kind": "direct", "value": 6015.0, "is_unset": False}
                    },
                }
            ],
        },
    )
    applied = ctrl.apply_writeback_items.call_args.args[1]
    mod = next(it for it in applied if it.key == "readout_rf")
    assert mod.edited_schema is not None
    assert mod.edited_schema.value.fields["freq"].value == 6015.0
