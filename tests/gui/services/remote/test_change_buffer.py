"""Per-connection GUI change buffer (notification half of Phase 92).

Drives the buffer bump / drain logic directly against a RemoteControlService
with mock clients — no socket. The stale guard (which reads this buffer) is
tested separately.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import (
    GuiEvent,
    MlChangedPayload,
    TabClosedPayload,
)
from zcu_tools.gui.services.remote import ControlOptions, RemoteControlService
from zcu_tools.gui.services.remote.change_categories import (
    CAT_CONTEXT_CHANGED,
    CAT_TAB_CHANGED,
    assert_exhaustive,
    category_for,
)
from zcu_tools.gui.services.remote.service import _ClientState
from zcu_tools.gui.services.remote.wire import Response


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001 — _MainThreadDispatcher is a QObject; needs an app
    yield


def _service():
    ctrl = MagicMock()
    ctrl.get_bus.return_value = None
    return RemoteControlService(controller=ctrl, opts=ControlOptions(port=0))


def _client(svc, peer="127.0.0.1:1"):
    state = _ClientState(peer=peer, token_required=False)
    with svc._clients_lock:
        svc._clients[object()] = state  # type: ignore[index]
    return state


# ---------------------------------------------------------------------------
# category mapping
# ---------------------------------------------------------------------------


def test_all_gui_events_have_category_or_ignored():
    # Import-time assert already runs; call explicitly for a named test too.
    assert_exhaustive()


def test_category_for_extracts_object():
    cat = category_for(GuiEvent.TAB_CLOSED, TabClosedPayload(tab_id="tab-9"))
    assert cat == (CAT_TAB_CHANGED, "tab-9")
    cat2 = category_for(GuiEvent.ML_CHANGED, MlChangedPayload(ml=MagicMock()))
    assert cat2 == (CAT_CONTEXT_CHANGED, None)


# ---------------------------------------------------------------------------
# bump / exclude originator / drain
# ---------------------------------------------------------------------------


def test_bump_tallies_all_clients():
    svc = _service()
    a = _client(svc, "a")
    b = _client(svc, "b")
    svc._bump_change_buffer(CAT_TAB_CHANGED, "tab-1")
    assert a.change_buffer[(CAT_TAB_CHANGED, "tab-1")] == 1
    assert b.change_buffer[(CAT_TAB_CHANGED, "tab-1")] == 1


def test_bump_skips_originating_client():
    svc = _service()
    a = _client(svc, "a")
    b = _client(svc, "b")
    svc._originating_state = a  # a is mid-RPC; its own change must not tally
    svc._bump_change_buffer(CAT_CONTEXT_CHANGED, None)
    svc._originating_state = None
    assert (CAT_CONTEXT_CHANGED, None) not in a.change_buffer
    assert b.change_buffer[(CAT_CONTEXT_CHANGED, None)] == 1


def test_bump_accumulates_count():
    svc = _service()
    a = _client(svc, "a")
    for _ in range(3):
        svc._bump_change_buffer(CAT_TAB_CHANGED, "tab-1")
    assert a.change_buffer[(CAT_TAB_CHANGED, "tab-1")] == 3


def test_drain_snapshots_and_clears():
    svc = _service()
    a = _client(svc, "a")
    svc._bump_change_buffer(CAT_TAB_CHANGED, "tab-1")
    svc._bump_change_buffer(CAT_CONTEXT_CHANGED, None)
    summary = svc._drain_change_buffer(a)
    assert summary is not None
    cats = {(e["category"], e["object_id"]): e["count"] for e in summary}
    assert cats[(CAT_TAB_CHANGED, "tab-1")] == 1
    assert cats[(CAT_CONTEXT_CHANGED, None)] == 1
    # read-clears
    assert a.change_buffer == {}
    assert svc._drain_change_buffer(a) is None


# ---------------------------------------------------------------------------
# Response envelope carries gui_changes only when present
# ---------------------------------------------------------------------------


def test_response_omits_gui_changes_when_empty():
    wire = Response(id="1", ok=True, result={}).to_wire()
    assert "gui_changes" not in wire


def test_response_includes_gui_changes_on_ok_and_error():
    changes = [{"category": "tab_changed", "object_id": "t", "count": 1}]
    ok = Response(id="1", ok=True, result={}, gui_changes=changes).to_wire()
    assert ok["gui_changes"] == changes
    from zcu_tools.gui.services.remote.errors import ErrorEnvelope

    err = Response(
        id="2",
        ok=False,
        error=ErrorEnvelope(code="precondition_failed", message="x"),
        gui_changes=changes,
    ).to_wire()
    assert err["gui_changes"] == changes
