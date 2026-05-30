"""Writeback preview/set/apply dispatch handlers (ADR-0010 persistent draft).

Drives the handlers against a mock Controller whose get_tab_writeback_items
returns crafted persistent items, so we can assert preview serialization, the
writeback.set edit path, and the apply path without a full run+analyze pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import MetaDictWriteback, ModuleWriteback
from zcu_tools.gui.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError


def _items() -> list:
    md = MetaDictWriteback(
        target_name="r_f", description="Resonator freq", proposed_value=6012.3
    )
    md.session_id = "md-1"
    mod = ModuleWriteback(
        target_name="readout_rf", description="readout module", edit_schema=MagicMock()
    )
    mod.session_id = "ml-1"
    mod.editor_id = "editor-9"
    return [md, mod]


def _ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_writeback_items.side_effect = lambda tab_id: _items()
    ctrl.apply_writeback.return_value = ["md-1"]
    return ctrl


def _dispatch(ctrl, method, params):
    return METHOD_REGISTRY[method].handler(ctrl, params)


def test_preview_serializes_metadict_and_module():
    ctrl = _ctrl()
    res = _dispatch(ctrl, "writeback.preview", {"tab_id": "t"})
    item_list = list(res["items"])  # type: ignore[call-overload]
    items = {it["id"]: it for it in item_list}

    md = items["md-1"]
    assert md["kind"] == "metadict"
    assert md["id"] == "md-1"
    assert md["target_name"] == "r_f"
    assert "key" not in md
    assert md["proposed_value"] == 6012.3

    mod = items["ml-1"]
    assert mod["kind"] == "module"
    assert mod["target_name"] == "readout_rf"
    assert mod["editor_id"] == "editor-9"
    assert mod["has_edit_schema"] is True


def test_apply_reads_persistent_draft():
    ctrl = _ctrl()
    res = _dispatch(ctrl, "writeback.apply", {"tab_id": "t"})
    assert res["applied_ids"] == ["md-1"]
    ctrl.apply_writeback.assert_called_once_with("t")


def test_set_edits_metadict_item():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "writeback.set",
        {"tab_id": "t", "id": "md-1", "selected": False, "proposed_value": 6015.0},
    )
    ctrl.set_writeback_item.assert_called_once()
    args, kwargs = ctrl.set_writeback_item.call_args
    assert args[:2] == ("t", "md-1")
    assert kwargs["selected"] is False
    assert kwargs["proposed_value"] == 6015.0


def test_set_target_name_override():
    ctrl = _ctrl()
    _dispatch(
        ctrl,
        "writeback.set",
        {"tab_id": "t", "id": "ml-1", "target_name": "readout_v2"},
    )
    _, kwargs = ctrl.set_writeback_item.call_args
    assert kwargs["target_name"] == "readout_v2"


def test_set_empty_target_name_rejected():
    ctrl = _ctrl()
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl, "writeback.set", {"tab_id": "t", "id": "md-1", "target_name": ""}
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_set_unknown_id_rejected():
    ctrl = _ctrl()
    ctrl.set_writeback_item.side_effect = RuntimeError("unknown writeback session_id")
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl, "writeback.set", {"tab_id": "t", "id": "md-99", "selected": True}
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS
