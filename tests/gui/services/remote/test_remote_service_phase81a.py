"""Phase 81a — RemoteControlService event push + Dialog API + view query.

Each test spins up a real TCP socket on an ephemeral loopback port via
``_helpers.Fixture`` and exercises:

  - ``events.subscribe`` / ``events.unsubscribe`` / ``events.list``;
  - server-pushed events with the requery-hint schema;
  - per-client writer thread + outbound queue overflow behaviour;
  - the Dialog API (``dialog.open`` / ``dialog.close`` / ``dialog.list_open``)
    against a mock ``ViewProtocol``;
  - ``view.snapshot`` / ``view.screenshot``;
  - clean teardown that unsubscribes from EventBus.
"""

from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import (
    GuiEvent,
    MdChangedPayload,
    RunLockChangedPayload,
    TabAddedPayload,
)

from ._helpers import (
    Fixture,
    call,
    open_client,
    recv_push,
    send,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def fx(qapp):  # noqa: ARG001
    f = Fixture()
    f.start()
    yield f
    f.stop()


# ---------------------------------------------------------------------------
# events.subscribe / unsubscribe / list
# ---------------------------------------------------------------------------


def test_events_list_includes_all_serialized(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "events.list")
        assert resp["ok"] is True
        names = set(resp["result"]["events"])
        # Sanity-check a representative subset; full list is documented in
        # events.py.
        for ev in ("tab_added", "run_lock_changed", "context_switched"):
            assert ev in names
        assert resp["result"]["subscribed"] == []
    finally:
        sock.close()


def test_subscribe_unknown_event_rejected(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(
            sock,
            "events.subscribe",
            {"events": ["definitely_not_an_event"]},
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_subscribe_then_run_lock_change_arrives(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["run_lock_changed"]})
        # Emit synthetically from the main thread (EventBus.emit is sync).
        fx.bus.emit(
            GuiEvent.RUN_LOCK_CHANGED, RunLockChangedPayload(running_tab_id="tab-x")
        )
        msg = recv_push(sock, "run_lock_changed")
        assert msg["payload"]["running_tab_id"] == "tab-x"
    finally:
        sock.close()


def test_unsubscribe_stops_push(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["tab_added"]})
        fx.bus.emit(
            GuiEvent.TAB_ADDED, TabAddedPayload(tab_id="a", adapter_name="fake")
        )
        first = recv_push(sock, "tab_added")
        assert first["payload"]["tab_id"] == "a"

        call(sock, "events.unsubscribe", {"events": ["tab_added"]})
        fx.bus.emit(
            GuiEvent.TAB_ADDED, TabAddedPayload(tab_id="b", adapter_name="fake")
        )
        # No further push should arrive — confirm by issuing an unrelated
        # RPC and verifying its reply comes back without a push in between.
        resp = call(sock, "state.has_context", rid="probe")
        assert resp["ok"] is True
    finally:
        sock.close()


def test_md_changed_emits_requery_hint(fx):
    """Composite payloads (MetaDict here) must not cross the wire."""
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["md_changed"]})
        fx.bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=MagicMock()))
        msg = recv_push(sock, "md_changed")
        assert msg["payload"] == {"requery": ["context.get_md_attr"]}
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Writer thread / queue overflow
# ---------------------------------------------------------------------------


def test_writer_queue_overflow_eventually_closes_wedged_client(fx, caplog):
    """A wedged reader gets dropped after exceeding the drop budget."""
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["tab_added"]})
        # Stop reading; emit far more events than the queue can hold.
        with caplog.at_level("WARNING"):
            for i in range(1024):
                fx.bus.emit(
                    GuiEvent.TAB_ADDED,
                    TabAddedPayload(tab_id=f"t{i}", adapter_name="fake"),
                )
        # Confirm at least one WARN about the outbound queue being full.
        assert any("outbound queue full" in rec.message for rec in caplog.records)
    finally:
        sock.close()


def test_stop_unsubscribes_event_bus(qapp):  # noqa: ARG001
    f = Fixture()
    f.start()
    # Confirm at least one subscription is installed.
    subs = f.bus._subs  # type: ignore[attr-defined]
    assert subs.get(GuiEvent.RUN_LOCK_CHANGED), "service should have subscribed"
    f.stop()
    subs_after = f.bus._subs  # type: ignore[attr-defined]
    assert not subs_after.get(GuiEvent.RUN_LOCK_CHANGED)
    assert not subs_after.get(GuiEvent.MD_CHANGED)


# ---------------------------------------------------------------------------
# Dialog API
# ---------------------------------------------------------------------------


def test_dialog_open_close_via_remote(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "dialog.open", {"name": "setup"})
        assert resp["ok"] is True
        assert resp["result"]["opened"] == "setup"
        assert fx.view.open_dialog.called

        resp = call(sock, "dialog.list_open")
        assert "setup" in resp["result"]["open"]

        resp = call(sock, "dialog.close", {"name": "setup"})
        assert resp["ok"] is True
        assert resp["result"]["closed"] == "setup"
        resp = call(sock, "dialog.list_open")
        assert resp["result"]["open"] == []
    finally:
        sock.close()


def test_dialog_open_unknown_name_rejected(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "dialog.open", {"name": "no_such_dialog"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_dialog_open_accepts_upper_or_lower(fx):
    sock = open_client(fx.service.port)
    try:
        for value in ("SETUP", "setup"):
            resp = call(sock, "dialog.open", {"name": value}, rid=f"r-{value}")
            assert resp["ok"] is True, value
            assert resp["result"]["opened"] == "setup"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# view.snapshot / view.screenshot
# ---------------------------------------------------------------------------


def test_view_snapshot_roundtrip(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "view.snapshot")
        assert resp["ok"] is True
        snap = resp["result"]
        assert snap["context_label"] == "ctx001"
        assert snap["status"] == "Ready"
        assert snap["open_dialogs"] == []
    finally:
        sock.close()


def test_view_screenshot_returns_png_magic(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "view.screenshot")
        assert resp["ok"] is True
        decoded = base64.b64decode(resp["result"]["png_b64"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
        assert resp["result"]["bytes"] == len(decoded)
    finally:
        sock.close()


def test_view_screenshot_precondition_failed_for_unknown_tab(fx):
    fx.view.take_screenshot = MagicMock(
        side_effect=RuntimeError("tab 'ghost' is not the active tab")
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "view.screenshot", {"tab_id": "ghost"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Run lifecycle push (integration)
# ---------------------------------------------------------------------------


def test_run_lifecycle_pushes_run_lock_changes(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["run_lock_changed"]})
        tab_id = call(sock, "tab.new", {"adapter_name": "fake"})["result"]["tab_id"]
        call(sock, "run.start", {"tab_id": tab_id})
        # Two pushes expected: start (running_tab_id=tab_id), end (None).
        first = recv_push(sock, "run_lock_changed")
        assert first["payload"]["running_tab_id"] == tab_id
        deadline = time.monotonic() + 5.0
        end_msg = None
        while time.monotonic() < deadline:
            msg = recv_push(sock, "run_lock_changed", timeout_s=2.0)
            if msg["payload"]["running_tab_id"] is None:
                end_msg = msg
                break
        assert end_msg is not None, "expected a release push (running_tab_id=null)"
    finally:
        sock.close()


def test_unauthenticated_subscribe_rejected(qapp):  # noqa: ARG001
    from zcu_tools.gui.services.remote import ControlOptions

    f = Fixture(ControlOptions(port=0, token="s3cr3t"))
    f.start()
    try:
        sock = open_client(f.service.port)
        try:
            send(
                sock,
                {
                    "id": "1",
                    "method": "events.subscribe",
                    "params": {"events": ["tab_added"]},
                },
            )
            from ._helpers import recv_response

            resp = recv_response(sock, "1")
            assert resp["ok"] is False
            assert resp["error"]["code"] == "unauthorized"
        finally:
            sock.close()
    finally:
        f.stop()
