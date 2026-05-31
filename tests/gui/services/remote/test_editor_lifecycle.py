"""Per-connection lifecycle of CfgEditor sessions in RemoteControlAdapter.

editor.open binds the returned id to the connection's _ClientState; commit /
discard forget it; a dropped connection reclaims any leftover sessions via
ctrl.discard_cfg_editors. These test the bookkeeping (_track_editor_lifecycle /
_reclaim_editors) directly, without a live socket.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.services.remote.service import _ClientState


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001 — _MainThreadDispatcher is a QObject; needs an app
    yield


def _service():
    ctrl = MagicMock()
    ctrl.get_bus.return_value = None  # disables event-bus subscription wiring
    svc = RemoteControlAdapter(controller=ctrl, opts=ControlOptions(port=0))
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


# ---------------------------------------------------------------------------
# Per-editor change stream routing (_on_editor_event)
# ---------------------------------------------------------------------------


def _enqueued_events(state):
    """Decode every line currently queued on a client's outbound queue."""
    import json

    out = []
    while not state.outbound.empty():
        line = state.outbound.get_nowait()
        out.append(json.loads(line.decode("utf-8")))
    return out


def test_editor_event_only_to_subscribers():
    svc, _ = _service()
    sub = _state()
    sub.subscribed_editors.add("editor-1")
    other = _state()
    with svc._clients_lock:
        svc._clients[object()] = sub  # type: ignore[index]
        svc._clients[object()] = other  # type: ignore[index]

    svc._on_editor_event("editor-1", "editor_changed", {"paths": []})

    sub_events = _enqueued_events(sub)
    assert len(sub_events) == 1
    assert sub_events[0]["event"] == "editor_changed"
    assert sub_events[0]["payload"]["editor_id"] == "editor-1"
    # non-subscriber got nothing.
    assert _enqueued_events(other) == []


def test_editor_closed_clears_subscription():
    svc, _ = _service()
    state = _state()
    state.subscribed_editors.add("editor-1")
    with svc._clients_lock:
        svc._clients[object()] = state  # type: ignore[index]

    svc._on_editor_event("editor-1", "editor_closed", {"reason": "tab_closed"})

    events = _enqueued_events(state)
    assert events[0]["event"] == "editor_closed"
    assert events[0]["payload"]["reason"] == "tab_closed"
    # subscription auto-dropped after close push.
    assert "editor-1" not in state.subscribed_editors


def test_editor_subscribe_handler_updates_state():
    svc, _ = _service()
    state = _state()
    svc._handle_editor_subscribe(state, "1", {"editor_id": "editor-9"}, subscribe=True)
    assert state.subscribed_editors == {"editor-9"}
    svc._handle_editor_subscribe(state, "2", {"editor_id": "editor-9"}, subscribe=False)
    assert state.subscribed_editors == set()


def test_editor_subscribe_rejects_bad_id():
    from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError

    svc, _ = _service()
    state = _state()
    with pytest.raises(RemoteError) as ei:
        svc._handle_editor_subscribe(state, "1", {"editor_id": ""}, subscribe=True)
    assert ei.value.code == ErrorCode.INVALID_PARAMS
