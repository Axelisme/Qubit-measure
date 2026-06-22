"""RemoteControlAdapter transport / dispatch / query tests.

Each test spins up a real TCP socket on an ephemeral loopback port and drives
a fixture Controller (already wired up with a fake adapter). The Qt event loop
is not freely running under pytest, so a helper interleaves
``QApplication.processEvents()`` between socket reads to give marshalled
handlers a chance to execute.
"""

from __future__ import annotations

import json
import socket
import time
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.app.main.adapter import ContextReadiness, ExpContext
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.main.services.remote.wire_version import (
    GUI_VERSION,
    WIRE_VERSION,
)
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.services.io_manager import IOManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        res_name="fake_res",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/fake_chip/fake_qubit",
        active_label="ctx001",
        readiness=ContextReadiness.ACTIVE,
    )


def _make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.make_live_container = MagicMock(return_value=None)
    # tab.list_all / overview read active_tab_id off the render view; return a real
    # (JSON-serializable) snapshot so the wire reply encodes cleanly.
    view.get_view_snapshot = MagicMock(
        return_value={"active_tab_id": None, "tab_ids": []}
    )
    return view


class _Fixture:
    """Hold strong refs to Controller + service to survive GC mid-test."""

    def __init__(self, opts: ControlOptions | None = None) -> None:
        self.state = State(_make_ctx())
        self.registry = Registry()
        register_all(self.registry)
        if not self.registry.has("fake"):
            self.registry.register("fake", FakeAdapter)
        self.view = _make_view()
        io_manager = IOManager()
        io_manager._em = MagicMock()
        self.bus = EventBus()
        self.ctrl = Controller(
            state=self.state,
            registry=self.registry,
            io_manager=io_manager,
            view=self.view,
            bus=self.bus,
        )
        if opts is None:
            opts = ControlOptions(port=0)
        # tab.list_all now reads active_tab_id off the render view (a view
        # projection), so the fixture must supply one — mirror _helpers.Fixture.
        self.service = RemoteControlAdapter(
            controller=self.ctrl, opts=opts, render_view=self.view
        )

    def start(self) -> int:
        return self.service.start()

    def stop(self) -> None:
        self.service.stop()


@pytest.fixture()
def fx(qapp):  # noqa: ARG001
    f = _Fixture()
    f.start()
    yield f
    f.stop()


def _send(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def _recv_response(sock: socket.socket, timeout_s: float = 3.0) -> dict:
    """Wait for one NDJSON response, pumping the Qt event loop in between."""
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    sock.setblocking(False)
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
            if chunk:
                buf.extend(chunk)
                if b"\n" in buf:
                    line, _, rest = bytes(buf).partition(b"\n")
                    return json.loads(line.decode("utf-8"))
                    # rest discarded — single-response helper
            else:
                # peer closed cleanly mid-recv → return ""
                if buf and b"\n" in buf:
                    line, _, _ = bytes(buf).partition(b"\n")
                    return json.loads(line.decode("utf-8"))
                raise AssertionError("peer closed without a response line")
        except BlockingIOError:
            pass
        app.processEvents()
        time.sleep(0.005)
    raise AssertionError(f"no response within {timeout_s}s")


def _open_client(port: int) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=1.0)
    return sock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_service_binds_loopback_only(fx):
    # The socket lives in the shared transport endpoint post-E3.
    addr = fx.service._endpoint._server_sock.getsockname()
    assert addr[0] == "127.0.0.1"
    assert fx.service.port > 0


def test_external_requires_token(qapp):  # noqa: ARG001
    with pytest.raises(RuntimeError, match="token"):
        RemoteControlAdapter(
            controller=MagicMock(),
            opts=ControlOptions(port=0, allow_external=True),
        )


def test_unknown_method_returns_error_code(fx):
    sock = _open_client(fx.service.port)
    try:
        _send(sock, {"id": "1", "method": "nope", "params": {}})
        resp = _recv_response(sock)
        assert resp["ok"] is False
        assert resp["error"]["code"] == "unknown_method"
    finally:
        sock.close()


def test_tab_new_list_close_roundtrip(fx):
    sock = _open_client(fx.service.port)
    try:
        _send(
            sock, {"id": "1", "method": "tab.new", "params": {"adapter_name": "fake"}}
        )
        resp = _recv_response(sock)
        assert resp["ok"] is True
        tab_id = resp["result"]["tab_id"]
        assert tab_id

        # tab.list_all returns the named shape {tabs, active_tab_id, running_tab_id}
        # where tabs is a list of {tab_id, adapter_name, is_running} objects.
        _send(sock, {"id": "2", "method": "tab.list_all", "params": {}})
        resp = _recv_response(sock)
        assert resp["ok"] is True
        tabs_list = resp["result"]["tabs"]
        ids = [t["tab_id"] for t in tabs_list]
        assert tab_id in ids

        _send(sock, {"id": "3", "method": "tab.close", "params": {"tab_id": tab_id}})
        resp = _recv_response(sock)
        assert resp["ok"] is True

        _send(sock, {"id": "4", "method": "tab.list_all", "params": {}})
        resp = _recv_response(sock)
        assert resp["result"]["tabs"] == []  # no tabs open
    finally:
        sock.close()


def test_invalid_typed_request_rejected(fx):
    sock = _open_client(fx.service.port)
    try:
        # missing 'kind'
        _send(sock, {"id": "1", "method": "soc.connect", "params": {}})
        resp = _recv_response(sock)
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"

        # wrong type for chip_name
        _send(
            sock,
            {
                "id": "2",
                "method": "startup.apply",
                "params": {
                    "chip_name": 42,
                    "qub_name": "Q1",
                    "res_name": "R1",
                    "result_dir": "/tmp/r",
                    "database_path": "/tmp/db",
                },
            },
        )
        resp = _recv_response(sock)
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_wire_version_reported(fx):
    sock = _open_client(fx.service.port)
    try:
        _send(sock, {"id": "1", "method": "wire.version", "params": {}})
        resp = _recv_response(sock)
        assert resp["ok"] is True
        assert resp["result"]["wire_version"] == WIRE_VERSION
        # The handshake also reports the (independent) GUI code revision.
        assert resp["result"]["gui_version"] == GUI_VERSION
    finally:
        sock.close()


def test_wire_version_is_no_auth(qapp):  # noqa: ARG001
    # wire.version is a handshake probe: it must answer before auth even on a
    # token-gated service, so a caller can detect a stale process on connect.
    f = _Fixture(ControlOptions(port=0, token="s3cr3t"))
    f.start()
    try:
        sock = _open_client(f.service.port)
        try:
            _send(sock, {"id": "1", "method": "wire.version", "params": {}})
            resp = _recv_response(sock)
            assert resp["ok"] is True
            assert resp["result"]["wire_version"] == WIRE_VERSION

            # A normal method is still gated.
            _send(sock, {"id": "2", "method": "state.has_context", "params": {}})
            assert _recv_response(sock)["error"]["code"] == "unauthorized"
        finally:
            sock.close()
    finally:
        f.stop()


def test_token_gated_when_set(qapp):  # noqa: ARG001
    f = _Fixture(ControlOptions(port=0, token="s3cr3t"))
    f.start()
    try:
        sock = _open_client(f.service.port)
        try:
            # Before auth: non-auth method rejected.
            _send(sock, {"id": "1", "method": "state.has_context", "params": {}})
            resp = _recv_response(sock)
            assert resp["ok"] is False
            assert resp["error"]["code"] == "unauthorized"

            # Bad token rejected.
            _send(sock, {"id": "2", "method": "auth", "params": {"token": "wrong"}})
            resp = _recv_response(sock)
            assert resp["ok"] is False
            assert resp["error"]["code"] == "unauthorized"

            # Good token authenticates.
            _send(sock, {"id": "3", "method": "auth", "params": {"token": "s3cr3t"}})
            assert _recv_response(sock)["ok"] is True

            _send(sock, {"id": "4", "method": "state.has_context", "params": {}})
            assert _recv_response(sock)["ok"] is True
        finally:
            sock.close()
    finally:
        f.stop()


def test_shutdown_closes_clients(fx):
    sock = _open_client(fx.service.port)
    try:
        fx.service.stop()
        # After stop, the service closes the client socket. Depending on
        # timing the peer observes either a clean EOF (recv -> b"") or a
        # reset (ConnectionResetError); both mean "socket is gone".
        sock.setblocking(True)
        sock.settimeout(2.0)
        try:
            data = sock.recv(4096)
        except (ConnectionResetError, ConnectionAbortedError, OSError):
            data = b""
        assert data == b""
    finally:
        sock.close()


def test_malformed_json_returns_invalid_params(fx):
    sock = _open_client(fx.service.port)
    try:
        sock.sendall(b"not json\n")
        resp = _recv_response(sock)
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_run_start_then_running_tab_then_finishes(fx):
    """Integration: drive a fake adapter from open to run-finished via polling."""
    sock = _open_client(fx.service.port)
    try:
        _send(
            sock, {"id": "1", "method": "tab.new", "params": {"adapter_name": "fake"}}
        )
        tab_id = _recv_response(sock)["result"]["tab_id"]

        _send(
            sock, {"id": "2", "method": "tab.run_start", "params": {"tab_id": tab_id}}
        )
        assert _recv_response(sock)["ok"] is True

        # Poll until run finishes (FakeAdapter completes very quickly).
        for _ in range(200):
            _send(sock, {"id": "p", "method": "run.running_tab", "params": {}})
            resp = _recv_response(sock)
            if resp["result"]["tab_id"] is None:
                break
            time.sleep(0.02)
        else:
            pytest.fail("run did not finish in time")

        _send(sock, {"id": "3", "method": "tab.snapshot", "params": {"tab_id": tab_id}})
        snap = _recv_response(sock)
        assert snap["ok"] is True
        # tab.snapshot always returns {tabs: [...]} (a single tab_id → a one-element list).
        assert snap["result"]["tabs"][0]["interaction"]["has_run_result"] is True
    finally:
        sock.close()
