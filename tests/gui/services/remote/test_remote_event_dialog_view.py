"""RemoteControlAdapter event push + Dialog API + view query.

Each test spins up a real TCP socket on an ephemeral loopback port via
``_helpers.Fixture`` and exercises:

  - ``events.subscribe`` / ``events.unsubscribe`` / ``events.list``;
  - server-pushed events with the requery-hint schema;
  - per-client writer thread + outbound queue overflow behaviour;
  - ``app.shutdown`` and ``view.snapshot`` (internal-only wire methods kept for
    gui_stop / overview, with no agent tool) against a mock ``ViewProtocol``;
  - clean teardown that unsubscribes from EventBus.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabContentFact,
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.services.remote.events import (
    _ser_predictor_changed,
    _ser_run_finished,
    _ser_run_started,
    _ser_soc_changed,
    _ser_tab_content_changed,
    _ser_tab_interaction_changed,
)
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
    MdChangedPayload,
    MlChangedPayload,
    PredictorChangedPayload,
    SocChangedPayload,
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
# Wire-name lock: guard against silent renames when payloads move between files
# ---------------------------------------------------------------------------

_EXPECTED_WIRE_NAMES = frozenset(
    {
        # tab domain
        "tab_added",
        "tab_closed",
        "tab_content_changed",
        "tab_interaction_changed",
        # run domain
        "run_started",
        "run_finished",
        # session-core (device / context / predictor / soc)
        "predictor_changed",
        "device_changed",
        "device_setup_started",
        "device_setup_finished",
        "context_switched",
        "md_changed",
        "ml_changed",
        "soc_changed",
    }
)


def test_event_serializers_wire_names_locked():
    """All 14 measure-gui EVENT_SERIALIZERS wire names are byte-identical to the
    documented set.  Any payload rename or accidental removal fails here first,
    before a connected agent notices silent breakage."""
    from zcu_tools.gui.app.main.services.remote.events import (
        EVENT_SERIALIZERS,
        wire_event_name,
    )

    actual = frozenset(wire_event_name(pt) for pt in EVENT_SERIALIZERS)
    assert actual == _EXPECTED_WIRE_NAMES, (
        f"Wire name set changed!\n"
        f"  Added:   {sorted(actual - _EXPECTED_WIRE_NAMES)}\n"
        f"  Removed: {sorted(_EXPECTED_WIRE_NAMES - actual)}"
    )


def test_all_event_serializer_payload_shapes_are_locked() -> None:
    from zcu_tools.gui.app.main.services.remote.events import EVENT_SERIALIZERS

    opaque = MagicMock()
    cases = [
        (
            TabAddedPayload(tab_id="tab-x", adapter_name="fake"),
            {"tab_id": "tab-x", "adapter_name": "fake"},
        ),
        (TabClosedPayload(tab_id="tab-x"), {"tab_id": "tab-x"}),
        (
            TabContentChangedPayload(
                tab_id="tab-x", fact=TabContentFact.RUN_RESULT_COMMITTED
            ),
            {"tab_id": "tab-x", "requery": ["tab.snapshot"]},
        ),
        (
            TabInteractionChangedPayload(
                tab_id="tab-x", fact=TabInteractionFact.SAVE_SUCCEEDED
            ),
            {"tab_id": "tab-x", "requery": ["tab.snapshot"]},
        ),
        (RunStartedPayload(tab_id="tab-x"), {"tab_id": "tab-x"}),
        (
            RunFinishedPayload(tab_id="tab-x", outcome="failed", error_message="boom"),
            {
                "tab_id": "tab-x",
                "outcome": "failed",
                "error_message": "boom",
                "requery": ["tab.snapshot"],
            },
        ),
        (PredictorChangedPayload(), {}),
        (
            DeviceChangedPayload(name="flux"),
            {"name": "flux", "requery": ["device.list"]},
        ),
        (DeviceSetupStartedPayload(name="flux"), {"name": "flux"}),
        (
            DeviceSetupFinishedPayload(
                name="flux", outcome="failed", error_message="boom"
            ),
            {"name": "flux", "outcome": "failed", "error_message": "boom"},
        ),
        (
            ContextSwitchedPayload(md=opaque, ml=opaque),
            {"requery": ["context.active"]},
        ),
        (MdChangedPayload(md=opaque), {"requery": ["context.md_get_attr"]}),
        (MlChangedPayload(ml=opaque), {"requery": ["context.ml_get"]}),
        (SocChangedPayload(soc=None, soccfg=None), {"connected": False}),
    ]

    assert len(cases) == len(EVENT_SERIALIZERS) == 14
    for payload, expected in cases:
        assert EVENT_SERIALIZERS[type(payload)](payload) == expected


@pytest.mark.parametrize("fact", list(TabInteractionFact))
def test_interaction_fact_does_not_change_wire_payload(
    fact: TabInteractionFact,
) -> None:
    assert _ser_tab_interaction_changed(
        TabInteractionChangedPayload("tab-x", fact)
    ) == {"tab_id": "tab-x", "requery": ["tab.snapshot"]}


@pytest.mark.parametrize("fact", list(TabContentFact))
def test_content_fact_does_not_change_wire_payload(fact: TabContentFact) -> None:
    assert _ser_tab_content_changed(TabContentChangedPayload("tab-x", fact)) == {
        "tab_id": "tab-x",
        "requery": ["tab.snapshot"],
    }


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


def test_no_matching_subscription_skips_serializer_and_encode(qapp, monkeypatch):  # noqa: ARG001
    from zcu_tools.gui.remote import control_service as service_module

    f = Fixture()
    serializer = MagicMock(return_value={"tab_id": "tab-x"})
    f.service._event_serializers = {
        **f.service._event_serializers,
        RunStartedPayload: serializer,
    }
    encode = MagicMock(wraps=service_module.encode_line)
    monkeypatch.setattr(service_module, "encode_line", encode)
    f.start()
    sock = open_client(f.service.port)
    try:
        call(sock, "events.list")
        encode.reset_mock()  # ignore the events.list reply (endpoint-owned encode)

        f.bus.emit(RunStartedPayload(tab_id="tab-x"))

        serializer.assert_not_called()
        encode.assert_not_called()
    finally:
        sock.close()
        f.stop()


def test_multiple_subscribers_serialize_and_encode_once(qapp, monkeypatch):  # noqa: ARG001
    from zcu_tools.gui.remote import control_service as service_module

    f = Fixture()
    serializer = MagicMock(return_value={"tab_id": "tab-x"})
    f.service._event_serializers = {
        **f.service._event_serializers,
        RunStartedPayload: serializer,
    }
    encode = MagicMock(wraps=service_module.encode_line)
    monkeypatch.setattr(service_module, "encode_line", encode)
    f.start()
    sockets = [open_client(f.service.port), open_client(f.service.port)]
    try:
        for index, sock in enumerate(sockets):
            call(
                sock,
                "events.subscribe",
                {"events": ["run_started"]},
                rid=f"sub-{index}",
            )
        encode.reset_mock()

        f.bus.emit(RunStartedPayload(tab_id="tab-x"))

        serializer.assert_called_once_with(RunStartedPayload(tab_id="tab-x"))
        encode.assert_called_once()
        assert [recv_push(sock, "run_started")["payload"] for sock in sockets] == [
            {"tab_id": "tab-x"},
            {"tab_id": "tab-x"},
        ]
    finally:
        for sock in sockets:
            sock.close()
        f.stop()


def test_serializer_failure_is_lazy_logged_and_isolated(qapp, caplog):  # noqa: ARG001
    f = Fixture()
    serializer = MagicMock(side_effect=RuntimeError("broken serializer"))
    f.service._event_serializers = {
        **f.service._event_serializers,
        RunStartedPayload: serializer,
    }
    later_subscriber = MagicMock()
    f.bus.subscribe(RunStartedPayload, later_subscriber)
    f.start()
    sock = open_client(f.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["run_started"]})

        with caplog.at_level("ERROR"):
            f.bus.emit(RunStartedPayload(tab_id="tab-x"))

        serializer.assert_called_once()
        later_subscriber.assert_called_once()
        assert "Event serializer for run_started raised" in caplog.text
        assert call(sock, "events.list", rid="probe")["ok"] is True
    finally:
        sock.close()
        f.stop()


def test_serializer_returning_none_only_runs_for_subscriber(qapp):  # noqa: ARG001
    f = Fixture()
    serializer = MagicMock(return_value=None)
    f.service._event_serializers = {
        **f.service._event_serializers,
        RunStartedPayload: serializer,
    }
    f.start()
    sock = open_client(f.service.port)
    try:
        call(sock, "events.list")
        f.bus.emit(RunStartedPayload(tab_id="no-sub"))
        serializer.assert_not_called()

        call(sock, "events.subscribe", {"events": ["run_started"]}, rid="sub")
        f.bus.emit(RunStartedPayload(tab_id="sub"))
        serializer.assert_called_once()
        assert call(sock, "events.list", rid="probe")["ok"] is True
    finally:
        sock.close()
        f.stop()


def test_push_encode_failure_is_logged_and_contained(qapp, monkeypatch, caplog):  # noqa: ARG001
    from zcu_tools.gui.remote import control_service as service_module

    f = Fixture()
    original_encode = service_module.encode_line

    def fail_push_only(payload):
        if "event" in payload:
            raise TypeError("not encodable")
        return original_encode(payload)

    monkeypatch.setattr(service_module, "encode_line", fail_push_only)
    f.start()
    sock = open_client(f.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["run_started"]})

        with caplog.at_level("ERROR"):
            f.bus.emit(RunStartedPayload(tab_id="tab-x"))

        assert "Failed to encode push line for run_started" in caplog.text
        assert call(sock, "events.list", rid="probe")["ok"] is True
    finally:
        sock.close()
        f.stop()


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


def test_editor_subscription_lazily_builds_then_unsubscribe_stops_it(fx):
    sock = open_client(fx.service.port)
    try:
        subscribed = call(
            sock,
            "editor.subscribe",
            {"editor_id": "editor-race"},
        )
        assert subscribed["result"]["subscribed_editors"] == ["editor-race"]
        payload_factory = MagicMock(return_value=())

        fx.service._on_editor_event("editor-race", "editor_changed", payload_factory)

        assert recv_push(sock, "editor_changed")["payload"] == {
            "editor_id": "editor-race",
            "paths": [],
        }
        payload_factory.assert_called_once_with()

        unsubscribed = call(
            sock,
            "editor.unsubscribe",
            {"editor_id": "editor-race"},
            rid="unsub-editor",
        )
        assert unsubscribed["result"]["subscribed_editors"] == []
        payload_factory.reset_mock()
        fx.service._on_editor_event("editor-race", "editor_changed", payload_factory)
        payload_factory.assert_not_called()
    finally:
        sock.close()


def test_md_changed_emits_requery_hint(fx):
    """Composite payloads (MetaDict here) must not cross the wire."""
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["md_changed"]})
        fx.bus.emit(MdChangedPayload(md=MagicMock()))
        msg = recv_push(sock, "md_changed")
        assert msg["payload"] == {"requery": ["context.md_get_attr"]}
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
    from zcu_tools.gui.app.main.services.remote.events import EVENT_SERIALIZERS

    f = Fixture()
    f.start()
    # Confirm at least one subscription is installed.
    subs = f.bus._subs  # type: ignore[attr-defined]
    assert subs.get(RunFinishedPayload), "service should have subscribed"
    assert len(f.service._bus_subs) == len(EVENT_SERIALIZERS)
    f.stop()
    subs_after = f.bus._subs  # type: ignore[attr-defined]
    # The RemoteEventService's subscriptions are gone after stop. RUN_FINISHED
    # is remote-event-specific, so it must be empty. (MD_CHANGED is *also* held
    # permanently by CfgEditorService for owned-model refresh — ADR-0008 — so it
    # is not a clean proxy for the remote view's teardown.)
    assert len(f.service._bus_subs) == 0
    assert not subs_after.get(RunFinishedPayload)


# ---------------------------------------------------------------------------
# app.shutdown (internal wire — no agent tool; gui_stop drives it)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# view.snapshot (internal wire — no agent tool; _assemble_overview reads it)
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


def test_view_screenshot_roundtrip(fx):
    """view.screenshot grabs the whole window and returns base64 PNG + bytes (the
    raw consumer shape; the mcp gui_screenshot tool decodes + writes the file)."""
    import base64

    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "view.screenshot")
        assert resp["ok"] is True
        result = resp["result"]
        png = base64.b64decode(result["png_b64"])
        assert png.startswith(b"\x89PNG")
        assert result["bytes"] == len(png)
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
        call(sock, "tab.run_start", {"tab_id": tab_id})
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
