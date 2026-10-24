"""Remote diagnostics, graceful shutdown and view snapshots over real sockets."""

from __future__ import annotations

import pytest

from tests.gui._qt_waits import qt_until

from ._helpers import (
    Fixture,
    call,
    open_client,
    recv_push,
)


@pytest.fixture()
def fx(tmp_path, qapp):
    f = Fixture(project_root=str(tmp_path / "project"))
    f.start()
    yield f
    f.stop()


def test_diagnostic_pushed_without_subscription(fx):
    """The adapter is a diagnostic sink (ADR-0068): a Controller diagnostic
    reaches the client out-of-band of EventBus, with no subscription needed."""
    sock = open_client(fx.service.port)
    try:
        # Round-trip first so the server has registered this client before the
        # diagnostic fans out (no subscription needed for diagnostics).
        call(sock, "state.has_active_context")
        # Drive a Controller diagnostic on the main thread (fan-out is sync).
        fx.ctrl._notify("error", "Run failed", "boom")
        msg = recv_push(sock, "diagnostic")
        assert set(msg) == {"event", "payload"}
        assert msg["payload"]["severity"] == "error"
        assert msg["payload"]["title"] == "Run failed"
        assert msg["payload"]["message"] == "boom"
    finally:
        sock.close()


def test_info_diagnostic_pushed(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "state.has_active_context")
        fx.ctrl._info("Data saved to /tmp/x")
        msg = recv_push(sock, "diagnostic")
        assert msg["payload"]["severity"] == "info"
        assert msg["payload"]["title"] == ""
        assert msg["payload"]["message"] == "Data saved to /tmp/x"
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
        qt_until(
            lambda: fx.view.request_shutdown.called,
            label="accepted shutdown owner continuation",
        )
        assert fx.view.request_shutdown.called
    finally:
        sock.close()


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
