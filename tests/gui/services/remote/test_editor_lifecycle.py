"""Per-connection lifecycle of CfgEditor sessions in RemoteControlService.

editor.open binds the returned id to the connection's _ClientState; commit /
discard forget it; a dropped connection reclaims any leftover sessions via
ctrl.discard_cfg_editors. These test the bookkeeping (_track_editor_lifecycle /
_reclaim_editors) directly, without a live socket.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.remote import ControlOptions, RemoteControlService
from zcu_tools.gui.services.remote.service import _ClientState


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001 — _MainThreadDispatcher is a QObject; needs an app
    yield


def _service():
    ctrl = MagicMock()
    ctrl.get_bus.return_value = None  # disables event-bus subscription wiring
    svc = RemoteControlService(controller=ctrl, opts=ControlOptions(port=0))
    return svc, ctrl


def _state() -> _ClientState:
    return _ClientState(peer="127.0.0.1:1", token_required=False)


def test_open_binds_id_to_client():
    svc, _ = _service()
    state = _state()
    svc._track_editor_lifecycle(
        state, "editor.open", {"item_kind": "module"}, {"editor_id": "editor-1"}
    )
    assert state.editor_ids == {"editor-1"}


def test_commit_forgets_id():
    svc, _ = _service()
    state = _state()
    state.editor_ids.add("editor-1")
    svc._track_editor_lifecycle(state, "editor.commit", {"editor_id": "editor-1"}, {})
    assert state.editor_ids == set()


def test_discard_forgets_id():
    svc, _ = _service()
    state = _state()
    state.editor_ids.add("editor-1")
    svc._track_editor_lifecycle(state, "editor.discard", {"editor_id": "editor-1"}, {})
    assert state.editor_ids == set()


def test_non_editor_method_ignored():
    svc, _ = _service()
    state = _state()
    svc._track_editor_lifecycle(
        state, "tab.new", {"adapter_name": "x"}, {"tab_id": "t"}
    )
    assert state.editor_ids == set()


def test_reclaim_discards_open_sessions_directly():
    svc, ctrl = _service()
    state = _state()
    state.editor_ids.update({"editor-1", "editor-2"})
    svc._reclaim_editors(state, marshal=False)
    ctrl.discard_cfg_editors.assert_called_once()
    (ids_arg,) = ctrl.discard_cfg_editors.call_args.args
    assert set(ids_arg) == {"editor-1", "editor-2"}
    assert state.editor_ids == set()


def test_reclaim_noop_when_no_sessions():
    svc, ctrl = _service()
    state = _state()
    svc._reclaim_editors(state, marshal=False)
    ctrl.discard_cfg_editors.assert_not_called()
