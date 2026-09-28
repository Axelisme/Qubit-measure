"""RemoteControlAdapter transport / dispatch / query tests.

Each test spins up a real TCP socket on an ephemeral loopback port and drives
a fixture Controller (already wired up with a fake adapter). The Qt event loop
is not freely running under pytest, so a helper interleaves
``QApplication.processEvents()`` between socket reads to give marshalled
handlers a chance to execute.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.remote import (
    ControlOptions,
    RemoteControlAdapter,
)
from zcu_tools.gui.app.measure.remote.wire_version import (
    GUI_VERSION,
    WIRE_VERSION,
)
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import MeasureMcpSession
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

from ._helpers import call as _raw_call
from ._helpers import call_mcp_with_qt as _call_mcp_with_qt
from ._helpers import observe_run_inputs
from ._remote_core_support import (
    RemoteCoreFixture as _Fixture,
)
from ._remote_core_support import (
    open_client as _open_client,
)
from ._remote_core_support import (
    recv_response as _recv_response,
)
from ._remote_core_support import (
    send as _send,
)


@pytest.fixture()
def fx(qapp):
    f = _Fixture()
    f.start()
    yield f
    f.stop()


pytestmark = pytest.mark.uses_wall_clock


def test_service_binds_loopback_only(fx):
    # The socket lives in the shared transport endpoint post-E3.
    addr = fx.service._endpoint._server_sock.getsockname()
    assert addr[0] == "127.0.0.1"
    assert fx.service.port > 0


def test_external_requires_token(qapp):
    with pytest.raises(RuntimeError, match="token"):
        RemoteControlAdapter(
            controller=MagicMock(),
            opts=ControlOptions(port=0, allow_external=True),
            owner_scheduler=QtOwnerScheduler(),
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


def test_catalog_exposes_live_params_and_policy_on_the_control_socket(fx):
    sock = _open_client(fx.service.port)
    try:
        _send(sock, {"id": "catalog", "method": "rpc.catalog", "params": {}})
        reply = _recv_response(sock)
        assert reply["ok"] is True
        methods = {entry["method"]: entry for entry in reply["result"]["methods"]}
        assert "rpc.catalog" not in methods
        assert methods["adapter.guide"]["exposure"] == "rpc"
        assert methods["adapter.guide"]["tool_names"] == []
        assert methods["adapter.guide"]["params"]["required"] == ["adapter_name"]
        assert methods["soc.info"]["exposure"] == "rpc"
        assert methods["soc.info"]["timeout_seconds"] == 5.0
        assert methods["tab.run_start"]["exposure"] == "rpc"
        assert methods["tab.run_start"]["tool_names"] == []
        assert methods["tab.run_start"]["operation_key"] == "tab:{tab_id}"

        _send(sock, {"id": "bad", "method": "adapter.guide", "params": {}})
        assert _recv_response(sock)["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_notify_await_rejects_wait_beyond_transport_budget(fx):
    sock = _open_client(fx.service.port)
    try:
        _send(
            sock,
            {
                "id": "too-long",
                "method": "notify.await",
                "params": {"token": 1, "timeout": 601},
            },
        )
        rejected = _recv_response(sock)
        assert rejected["ok"] is False
        assert rejected["error"]["code"] == "invalid_params"
        assert rejected["error"]["reason"] == "invalid_timeout"

        # An unknown prompt returns immediately; this checks the upper bound
        # without waiting for a real user or for the backstop to expire.
        _send(
            sock,
            {
                "id": "bounded",
                "method": "notify.await",
                "params": {"token": 1, "timeout": 600},
            },
        )
        accepted = _recv_response(sock)
        assert accepted["ok"] is True
        assert accepted["result"] == {"reason": "dismiss"}
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


@pytest.mark.parametrize("operation_id", [True, "1", 1.5, None])
def test_cancel_rejects_invalid_operation_id_over_socket(fx, operation_id):
    sock = _open_client(fx.service.port)
    try:
        _send(
            sock,
            {
                "id": "invalid-cancel",
                "method": "operation.cancel",
                "params": {"operation_id": operation_id},
            },
        )
        reply = _recv_response(sock)
        assert reply["ok"] is False
        assert reply["error"]["code"] == "invalid_params"
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


def test_wire_version_is_no_auth(qapp):
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


def test_token_gated_when_set(qapp):
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


def test_measure_connect_authenticates_and_reconnects_to_token_gated_gui(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
    # This test owns authentication, not the unrelated status orientation reads.
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    first = _Fixture(ControlOptions(port=0, token="test-secret"))
    port = first.start()
    config = MCPBridgeConfig(
        tool_prefix="",
        server_display_name="measure-test",
        server_instructions="",
        app_name="gui",
        default_port=port,
        mcp_version=80,
        wire_version=WIRE_VERSION,
        pid_file=tmp_path / "unused.pid",
        log_file=tmp_path / "unused.log",
        run_script_name="run_measure_gui.py",
    )

    def resolver(config: MCPBridgeConfig, requested: int | None) -> int:
        return config.default_port if requested is None else requested

    session = MeasureMcpSession(
        config, resolve_connect_port=resolver, port_is_open=lambda _: True
    )
    bridge = McpBridge(config)
    session.attach_bridge(bridge)
    tools = build_measure_tools(
        MeasureToolContext(config, session, resolve_connect_port=resolver)
    )
    call = partial(_call_mcp_with_qt, tools)

    second: _Fixture | None = None
    try:
        for credential in (None, "wrong"):
            args: dict[str, Any] = {"port": port}
            if credential is not None:
                args["token"] = credential
            with pytest.raises(RuntimeError) as exc_info:
                call("connect", args)
            assert getattr(exc_info.value, "reason", None) == "unauthorized"
        assert call("connect", {"port": port, "token": "test-secret"})["port"] == port
        assert call("rpc_list", {"domain": "adapter"})["methods"]

        first.stop()
        second = _Fixture(ControlOptions(port=port, token="test-secret"))
        assert second.start() == port
        deadline = time.monotonic() + 2
        while bridge.is_connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not bridge.is_connected
        assert call("rpc_list", {"domain": "adapter"})["methods"]
    finally:
        bridge.disconnect()
        first.stop()
        if second is not None:
            second.stop()


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
        observe_run_inputs(
            fx, tab_id, lambda method, params: _raw_call(sock, method, params)["result"]
        )

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
        tab_snapshot = snap["result"]["tabs"][0]
        assert tab_snapshot["interaction"]["has_run_result"] is True
        assert set(tab_snapshot["save_paths"]) == {
            "data_path",
            "analysis_image_path",
            "post_analysis_image_path",
        }
        assert "image_path" not in tab_snapshot["save_paths"]
    finally:
        sock.close()
