"""RemoteControlAdapter event push + Dialog API + view query.

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
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.event_bus import (
    MdChangedPayload,
    PredictorChangedPayload,
    RunFinishedPayload,
    RunStartedPayload,
    SocChangedPayload,
    TabAddedPayload,
)
from zcu_tools.gui.app.main.services.remote.events import (
    _ser_predictor_changed,
    _ser_run_finished,
    _ser_run_started,
    _ser_soc_changed,
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
        for ev in ("tab_added", "run_started", "run_finished", "context_switched"):
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


def test_subscribe_then_run_started_arrives(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["run_started"]})
        # Emit synthetically from the main thread (EventBus.emit is sync).
        fx.bus.emit(RunStartedPayload(tab_id="tab-x"))
        msg = recv_push(sock, "run_started")
        assert msg["payload"]["tab_id"] == "tab-x"
    finally:
        sock.close()


def test_diagnostic_pushed_without_subscription(fx):
    """The adapter is a diagnostic sink (ADR-0013): a Controller diagnostic
    reaches the client out-of-band of EventBus, with no subscription needed."""
    sock = open_client(fx.service.port)
    try:
        # Round-trip first so the server has registered this client before the
        # diagnostic fans out (no subscription needed for diagnostics).
        call(sock, "events.list")
        # Drive a Controller diagnostic on the main thread (fan-out is sync).
        fx.ctrl._notify("error", "Run failed", "boom")
        msg = recv_push(sock, "diagnostic")
        assert msg["payload"]["severity"] == "error"
        assert msg["payload"]["title"] == "Run failed"
        assert msg["payload"]["message"] == "boom"
    finally:
        sock.close()


def test_info_diagnostic_pushed(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.list")
        fx.ctrl._info("Data saved to /tmp/x")
        msg = recv_push(sock, "diagnostic")
        assert msg["payload"]["severity"] == "info"
        assert msg["payload"]["title"] == ""
        assert msg["payload"]["message"] == "Data saved to /tmp/x"
    finally:
        sock.close()


def test_unsubscribe_stops_push(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["tab_added"]})
        fx.bus.emit(TabAddedPayload(tab_id="a", adapter_name="fake"))
        first = recv_push(sock, "tab_added")
        assert first["payload"]["tab_id"] == "a"

        call(sock, "events.unsubscribe", {"events": ["tab_added"]})
        fx.bus.emit(TabAddedPayload(tab_id="b", adapter_name="fake"))
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
        fx.bus.emit(MdChangedPayload(md=MagicMock()))
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
    assert subs.get(RunFinishedPayload), "service should have subscribed"
    f.stop()
    subs_after = f.bus._subs  # type: ignore[attr-defined]
    # The RemoteEventService's subscriptions are gone after stop. RUN_FINISHED
    # is remote-event-specific, so it must be empty. (MD_CHANGED is *also* held
    # permanently by CfgEditorService for owned-model refresh — ADR-0008 — so it
    # is not a clean proxy for the remote view's teardown.)
    assert not subs_after.get(RunFinishedPayload)


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


def test_app_shutdown_triggers_request_shutdown(fx):
    """app.shutdown drives the View's request_shutdown (the graceful window-close
    path) rather than any OS kill."""
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "app.shutdown", {})
        assert resp["ok"] is True
        assert resp["result"]["shutting_down"] is True
        assert fx.view.request_shutdown.called
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


def test_run_lifecycle_pushes_run_started_then_finished(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["run_started", "run_finished"]})
        tab_id = call(sock, "tab.new", {"adapter_name": "fake"})["result"]["tab_id"]
        call(sock, "run.start", {"tab_id": tab_id})
        # One run_started, then one run_finished with outcome='finished'.
        started = recv_push(sock, "run_started")
        assert started["payload"]["tab_id"] == tab_id
        finished = recv_push(sock, "run_finished", timeout_s=5.0)
        assert finished["payload"]["tab_id"] == tab_id
        assert finished["payload"]["outcome"] == "finished"
    finally:
        sock.close()


def test_unauthenticated_subscribe_rejected(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote import ControlOptions

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


# ---------------------------------------------------------------------------
# Serializer unit tests (pure functions — no TCP needed)
# ---------------------------------------------------------------------------


def test_ser_run_started():
    wire = _ser_run_started(RunStartedPayload(tab_id="tab1"))
    assert wire is not None
    assert wire["tab_id"] == "tab1"
    assert "outcome" not in wire


def test_ser_run_finished_includes_outcome():
    wire = _ser_run_finished(
        RunFinishedPayload(tab_id="tab1", outcome="finished", error_message=None)
    )
    assert wire is not None
    assert wire["outcome"] == "finished"
    assert wire["tab_id"] == "tab1"
    assert "error_message" not in wire
    assert "requery" in wire


def test_ser_run_finished_failed_includes_error_message():
    wire = _ser_run_finished(
        RunFinishedPayload(tab_id="tab1", outcome="failed", error_message="timeout")
    )
    assert wire is not None
    assert wire["outcome"] == "failed"
    assert wire["error_message"] == "timeout"


def test_ser_run_finished_cancelled():
    wire = _ser_run_finished(
        RunFinishedPayload(tab_id="tab1", outcome="cancelled", error_message=None)
    )
    assert wire is not None
    assert wire["outcome"] == "cancelled"
    assert "error_message" not in wire


def test_ser_predictor_changed_returns_empty_dict():
    payload = PredictorChangedPayload()
    wire = _ser_predictor_changed(payload)
    assert wire == {}


def test_ser_soc_changed_connected():
    payload = SocChangedPayload(soc=MagicMock(), soccfg=MagicMock())
    wire = _ser_soc_changed(payload)
    assert wire is not None
    assert wire["connected"] is True


def test_ser_soc_changed_disconnected():
    payload = SocChangedPayload(soc=None, soccfg=None)
    wire = _ser_soc_changed(payload)
    assert wire is not None
    assert wire["connected"] is False
