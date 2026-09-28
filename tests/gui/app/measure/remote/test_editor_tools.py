"""CfgEditor wire handlers (dispatch layer).

Drives the editor.* handlers against a mock Controller to assert the wire shape
and error translation, without a live RemoteControlAdapter.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorError
from zcu_tools.gui.app.measure.services.ports import CfgEditResult
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.services.context import MlEntryValidationError

from ._helpers import dispatch_handler as _dispatch  # noqa: E402


@pytest.fixture
def draft():
    from zcu_tools.gui.cfg import (
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        DirectValue,
        ScalarSpec,
    )
    from zcu_tools.gui.cfg.binding import CfgDraft

    model = CfgDraft(
        CfgSchema(
            CfgSectionSpec(fields={"freq": ScalarSpec("Frequency", float)}),
            CfgSectionValue({"freq": DirectValue(5000.0)}),
        ),
        evaluate_expression=MagicMock(),
        provide_options=lambda source_id: (),
        references=MagicMock(),
    )
    try:
        yield model
    finally:
        model.close()


def test_open_returns_editor_id_and_tree(draft):
    ctrl = MagicMock()
    ctrl.open_cfg_editor.return_value = ("editor-abc", [{"path": "freq"}])
    ctrl.get_cfg_editor_draft.return_value = draft
    res = _dispatch(
        ctrl, "editor.new", {"item_kind": "module", "from_name": "readout_rf"}
    )
    assert res["editor_id"] == "editor-abc"
    tree = res["tree"]
    assert isinstance(tree, dict)
    assert tree["children"]["freq"]["input"]["resolved"] == 5000.0
    assert tree["valid"]
    ctrl.open_cfg_editor.assert_called_once_with("module", from_name="readout_rf")


def test_open_translates_cfg_editor_error():
    ctrl = MagicMock()
    ctrl.open_cfg_editor.side_effect = CfgEditorError("unknown module")
    with pytest.raises(RemoteError) as ei:
        _dispatch(ctrl, "editor.new", {"item_kind": "module", "from_name": "nope"})
    assert ei.value.code == ErrorCode.INVALID_PARAMS


def test_set_field_passes_through_result():
    ctrl = MagicMock()
    ctrl.cfg_editor_set_field.return_value = CfgEditResult(valid=True)
    res = _dispatch(
        ctrl,
        "editor.set_field",
        {"editor_id": "editor-abc", "path": "freq", "value": 5000.0},
    )
    assert res["valid"] is True
    ctrl.cfg_editor_set_field.assert_called_once_with("editor-abc", "freq", 5000.0)


def test_set_field_eval_value_passed_through():
    ctrl = MagicMock()
    ctrl.cfg_editor_set_field.return_value = CfgEditResult(valid=True)
    ev = {"__kind": "eval", "expr": "r_f"}
    _dispatch(
        ctrl,
        "editor.set_field",
        {"editor_id": "e", "path": "freq", "value": ev},
    )
    # Handler is a thin pass-through; the service decodes the tagged eval.
    ctrl.cfg_editor_set_field.assert_called_once_with("e", "freq", ev)


def test_get_projects_selected_node_from_the_model(draft):
    ctrl = MagicMock()
    ctrl.get_cfg_editor_draft.return_value = draft
    res = _dispatch(ctrl, "editor.get", {"editor_id": "e", "prefix": "freq"})
    tree = res["tree"]
    assert isinstance(tree, dict)
    assert tree["path"] == "freq"
    assert tree["input"]["resolved"] == 5000.0
    assert tree["kind"] == "scalar"


def test_get_unknown_editor_is_invalid_params():
    ctrl = MagicMock()
    ctrl.get_cfg_editor_draft.side_effect = CfgEditorError("unknown editor")
    with pytest.raises(RemoteError) as ei:
        _dispatch(ctrl, "editor.get", {"editor_id": "ghost"})
    assert ei.value.code == ErrorCode.INVALID_PARAMS


def test_commit_translates_validation_error():
    ctrl = MagicMock()
    ctrl.commit_cfg_editor.side_effect = MlEntryValidationError("invalid module")
    with pytest.raises(RemoteError) as ei:
        _dispatch(ctrl, "editor.commit", {"editor_id": "e", "name": "x"})
    assert ei.value.code == ErrorCode.INVALID_PARAMS


def test_commit_unknown_session_is_invalid_params():
    ctrl = MagicMock()
    ctrl.commit_cfg_editor.side_effect = CfgEditorError("unknown editor")
    with pytest.raises(RemoteError) as ei:
        _dispatch(ctrl, "editor.commit", {"editor_id": "nope", "name": "x"})
    assert ei.value.code == ErrorCode.INVALID_PARAMS


def test_commit_ok_returns_empty():
    ctrl = MagicMock()
    ctrl.commit_cfg_editor.return_value = None
    res = _dispatch(ctrl, "editor.commit", {"editor_id": "e", "name": "agent_mod"})
    assert res == {}
    ctrl.commit_cfg_editor.assert_called_once_with("e", "agent_mod")


def test_discard_returns_empty():
    ctrl = MagicMock()
    res = _dispatch(ctrl, "editor.discard", {"editor_id": "e"})
    assert res == {}
    ctrl.discard_cfg_editor.assert_called_once_with("e")
