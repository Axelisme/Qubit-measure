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
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from qick.asm_v2 import QickParam
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
from zcu_tools.gui.remote.framing import MAX_LINE_BYTES
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import MeasureMcpSession
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import WaveformCfgFactory
from zcu_tools.program.v2.mocksoc import make_mock_soccfg

from ._helpers import call as _raw_call
from ._helpers import call_mcp_with_qt as _call_mcp_with_qt
from ._helpers import mcp_client as _mcp_client
from ._helpers import observe_run_inputs

# Poll real socket workers while delivering owner-thread Qt events.
pytestmark = pytest.mark.uses_wall_clock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),
        soccfg=make_mock_soccfg(),
        res_name="fake_res",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/fake_chip/fake_qubit",
        active_label="ctx001",
        readiness=ContextReadiness.ACTIVE,
    )


def _make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.make_run_container = MagicMock(return_value=None)
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
        io_manager = MagicMock(spec=IOManager, has_project=True, has_context=True)
        io_manager.get_active_label.return_value = "ctx001"
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
            controller=self.ctrl,
            opts=opts,
            owner_scheduler=QtOwnerScheduler(),
            render_view=self.view,
        )

    def start(self) -> int:
        return self.service.start()

    def stop(self) -> None:
        self.service.stop()


@pytest.fixture()
def fx(qapp):
    f = _Fixture()
    f.start()
    yield f
    f.stop()


def _send(sock: socket.socket, obj: dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def _recv_response(sock: socket.socket, timeout_s: float = 3.0) -> dict[str, Any]:
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
                    line, _, _ = bytes(buf).partition(b"\n")
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
    return socket.create_connection(("127.0.0.1", port), timeout=1.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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


def _prepare_guarded_context(fx: _Fixture, ml: ModuleLibrary | None = None) -> None:
    fx.state.set_context(
        replace(
            fx.state.exp_context,
            md=MetaDict(),
            ml=ml if ml is not None else ModuleLibrary(),
            soccfg=make_mock_soccfg(),
        )
    )
    fx.state.version.bump("context")
    fx.state.version.bump("soc")


def _await_completed_run(
    call: Callable[[str, dict[str, Any]], dict[str, Any]], handle: int
) -> None:
    assert call("wait", {"op": handle, "timeout": 3})["status"] == "finished"


def _edit_context_as_gui(fx: _Fixture, key: str, value: object) -> None:
    sock = _open_client(fx.service.port)
    try:
        _send(
            sock,
            {
                "id": "edit",
                "method": "context.md_set_attr",
                "params": {"key": key, "value": value},
            },
        )
        assert _recv_response(sock)["ok"] is True
    finally:
        sock.close()


def test_socket_write_requires_a_full_read_on_that_connection(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    sock = _open_client(fx.service.port)
    try:
        args = {"tab_id": tab_id, "data_path": "missing.h5"}
        _send(sock, {"id": "unread", "method": "tab.load_data", "params": args})
        unread = _recv_response(sock)
        assert unread["error"]["reason"] == "stale_version"
        assert "context" in unread["error"]["data"]["stale"]
        assert f"tab:{tab_id}:result" in unread["error"]["data"]["stale"]
        assert f"tab:{tab_id}:analyze" in unread["error"]["data"]["stale"]

        for method, params in (
            ("tab.snapshot", {"tab_id": tab_id}),
            ("context.snapshot", {}),
        ):
            _send(sock, {"id": method, "method": method, "params": params})
            assert _recv_response(sock)["ok"] is True
        _send(sock, {"id": "read", "method": "tab.load_data", "params": args})
        after_read = _recv_response(sock)
        assert after_read["ok"] is False
        assert after_read["error"].get("reason") != "stale_version"
    finally:
        sock.close()


def test_socket_snapshot_exposes_result_replacement_and_restores_guard(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    sock = _open_client(fx.service.port)
    try:

        def rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
            _send(sock, {"id": method, "method": method, "params": params})
            return _recv_response(sock)

        assert rpc("context.snapshot", {})["ok"] is True
        initial = rpc("tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert initial["result_state"] == {
            "revision": 0,
            "available": False,
            "source_path": None,
        }
        assert initial["analysis_state"]["available"] is False
        assert initial["post_analysis_state"]["available"] is False
        # A non-serializable result stays in State; the operational projection
        # identifies replacements without requiring its payload on the wire.
        fx.state.update_tab_loaded_result(tab_id, object(), "loaded.h5")
        args = {"tab_id": tab_id, "data_path": "missing.h5"}
        assert rpc("tab.load_data", args)["error"]["reason"] == "stale_version"
        observed = rpc("tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert observed["result_state"] == {
            "revision": 1,
            "available": True,
            "source_path": "loaded.h5",
        }
        assert rpc("tab.load_data", args)["error"].get("reason") != "stale_version"
        fx.state.update_tab_loaded_result(tab_id, object(), "loaded.h5")
        assert rpc("tab.load_data", args)["error"]["reason"] == "stale_version"
        replacement = rpc("tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert replacement["result_state"]["revision"] == 2
        assert replacement["result_state"]["source_path"] == "loaded.h5"
        assert rpc("tab.load_data", args)["error"].get("reason") != "stale_version"
    finally:
        sock.close()


def test_socket_self_write_advances_only_its_prior_seen_context(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    first = _open_client(fx.service.port)
    second = _open_client(fx.service.port)
    try:
        for sock in (first, second):
            for method, params in (
                ("tab.snapshot", {"tab_id": tab_id}),
                ("context.snapshot", {}),
            ):
                _send(sock, {"id": method, "method": method, "params": params})
                assert _recv_response(sock)["ok"] is True

        _send(
            first,
            {
                "id": "write",
                "method": "context.md_set_attr",
                "params": {"key": "r_f", "value": 6000.0},
            },
        )
        assert _recv_response(first)["ok"] is True
        args = {"tab_id": tab_id, "data_path": "missing.h5"}
        _send(first, {"id": "self", "method": "tab.load_data", "params": args})
        assert _recv_response(first)["error"].get("reason") != "stale_version"
        _send(second, {"id": "other", "method": "tab.load_data", "params": args})
        assert _recv_response(second)["error"]["reason"] == "stale_version"

        _send(second, {"id": "refresh", "method": "context.snapshot", "params": {}})
        assert _recv_response(second)["result"]["md"]["r_f"] == 6000.0
        _send(second, {"id": "retry", "method": "tab.load_data", "params": args})
        assert _recv_response(second)["error"].get("reason") != "stale_version"
    finally:
        first.close()
        second.close()


def test_timed_out_socket_read_does_not_establish_guard_baseline(fx) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    original = fx.service._method_registry["context.snapshot"]

    def slow_snapshot(adapter, params):
        time.sleep(0.15)
        return original.handler(adapter, params)

    fx.service._method_registry = {
        **fx.service._method_registry,
        "context.snapshot": replace(
            original,
            handler=slow_snapshot,
            spec=replace(original.spec, timeout_seconds=0.01),
        ),
    }
    sock = _open_client(fx.service.port)
    try:
        _send(
            sock,
            {"id": "tab", "method": "tab.snapshot", "params": {"tab_id": tab_id}},
        )
        assert _recv_response(sock)["ok"] is True
        _send(sock, {"id": "slow", "method": "context.snapshot", "params": {}})
        timed_out = _recv_response(sock)
        assert timed_out["ok"] is False
        assert timed_out["error"]["code"] == "timeout"
        _send(
            sock,
            {
                "id": "write",
                "method": "tab.load_data",
                "params": {"tab_id": tab_id, "data_path": "missing.h5"},
            },
        )
        assert _recv_response(sock)["error"]["reason"] == "stale_version"
    finally:
        sock.close()


def test_mcp_created_tab_can_start_a_guarded_run_on_real_gui_state(
    fx, tmp_path: Path
) -> None:
    _prepare_guarded_context(fx)
    port = fx.service.port
    bridge, call = _mcp_client(port, tmp_path)
    try:
        assert call("connect", {"port": port})["port"] == port
        tab_id = call(
            "rpc_call", {"method": "tab.new", "params": {"adapter_name": "fake"}}
        )["tab_id"]
        assert call("rpc_call", {"method": "context.snapshot"})["md"] == {}
        assert "cfg" in call(
            "rpc_call", {"method": "soc.info", "params": {"include_cfg": True}}
        )
        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        started = call(
            "rpc_call", {"method": "tab.run_start", "params": {"tab_id": tab_id}}
        )
        assert started["handle"] > 0
        _await_completed_run(call, started["handle"])
    finally:
        bridge.disconnect()


def test_attached_gui_tab_runs_after_explicit_full_reads(fx, tmp_path: Path) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")  # Created by GUI before MCP attaches.
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        assert (
            call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})[
                "tabs"
            ][0]["tab_id"]
            == tab_id
        )
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        handle = call(
            "rpc_call", {"method": "tab.run_start", "params": {"tab_id": tab_id}}
        )["handle"]
        assert handle > 0
        _await_completed_run(call, handle)
    finally:
        bridge.disconnect()


def test_restarted_gui_requires_new_full_reads_before_running(
    qapp, tmp_path: Path
) -> None:
    first = _Fixture()
    port = first.start()
    _prepare_guarded_context(first)
    first_tab = first.ctrl.new_tab("fake")
    bridge, call = _mcp_client(port, tmp_path)
    second: _Fixture | None = None
    try:
        call("connect", {"port": port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": first_tab}})
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            first,
            first_tab,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        old_handle = call(
            "rpc_call", {"method": "tab.run_start", "params": {"tab_id": first_tab}}
        )["handle"]
        _await_completed_run(call, old_handle)

        first.stop()
        second = _Fixture(ControlOptions(port=port))
        assert second.start() == port
        _prepare_guarded_context(second)
        second_tab = second.ctrl.new_tab("fake")
        deadline = time.monotonic() + 2
        while bridge.is_connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not bridge.is_connected
        call("rpc_list", {"domain": "context"})  # Lazily reconnect to the new GUI.
        with pytest.raises(RuntimeError) as expired:
            call("wait", {"op": old_handle, "timeout": 0.01})
        assert getattr(expired.value, "reason", None) == "unknown_op"
        # The old snapshots must not supply the new GUI's nonzero tab, SoC or
        # context guard baseline. Explicit reads restore access to this tab.
        with pytest.raises(RuntimeError) as stale:
            call(
                "rpc_call",
                {"method": "tab.run_start", "params": {"tab_id": second_tab}},
            )
        assert getattr(stale.value, "reason", None) == "stale_version"
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": second_tab}})
        call("rpc_call", {"method": "context.snapshot"})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        observe_run_inputs(
            second,
            second_tab,
            lambda method, params: call(
                "rpc_call", {"method": method, "params": params}
            ),
        )
        new_handle = call(
            "rpc_call", {"method": "tab.run_start", "params": {"tab_id": second_tab}}
        )["handle"]
        assert new_handle > old_handle
        _await_completed_run(call, new_handle)
    finally:
        bridge.disconnect()
        first.stop()
        if second is not None:
            second.stop()


def test_load_after_gui_context_edit_requires_a_new_full_read(
    fx, tmp_path: Path
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        _edit_context_as_gui(fx, "r_f", 6000.0)
        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"

        assert call("rpc_call", {"method": "context.snapshot"})["md"]["r_f"] == 6000.0
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) != "stale_version"
    finally:
        bridge.disconnect()


@pytest.mark.parametrize(
    "lossy",
    [{"1": "first", 1: "second"}, QickParam(start=1.0, spans={"pulse": 0.1})],
    ids=["nested-key-collision", "qick-param"],
)
def test_failed_complete_context_read_does_not_advance_mcp_load_guard(
    fx, tmp_path: Path, lossy: object
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        md = fx.state.exp_context.md
        md.update({"nested": lossy})
        fx.state.version.bump("context")
        with pytest.raises(RuntimeError) as unreadable:
            call("rpc_call", {"method": "context.snapshot"})
        assert getattr(unreadable.value, "reason", None) == "unserializable_context"

        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"

        md.update({"nested": {"first": "a", "second": "b"}})
        fx.state.version.bump("context")
        snapshot = call("rpc_call", {"method": "context.snapshot"})
        assert snapshot["md"]["nested"] == {"first": "a", "second": "b"}
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) != "stale_version"
    finally:
        bridge.disconnect()


@pytest.mark.parametrize("mutate_cfg", [False, True])
def test_frozen_run_needs_cfg_observation_not_large_context_export(
    fx, tmp_path: Path, mutate_cfg: bool
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    editor_id, _ = fx.ctrl.open_seeded_cfg_editor(
        fx.state.get_tab(tab_id).cfg_schema, owner_key=tab_id
    )
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        _edit_context_as_gui(fx, "large", "x" * (MAX_LINE_BYTES // 2))
        _edit_context_as_gui(fx, "other_large", "x" * (MAX_LINE_BYTES // 2))
        with pytest.raises(RuntimeError) as oversized:
            call("rpc_call", {"method": "context.snapshot"})
        assert getattr(oversized.value, "reason", None) == "response_encoding_failed"
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
        _edit_context_as_gui(fx, "unrelated", 17)
        if mutate_cfg:
            fx.ctrl.cfg_editor_set_field(editor_id, "reps", 42)
        args = {"method": "tab.run_start", "params": {"tab_id": tab_id}}
        if mutate_cfg:
            with pytest.raises(RuntimeError) as stale:
                call("rpc_call", args)
            assert getattr(stale.value, "reason", None) == "stale_version"
        else:
            started = call("rpc_call", args)
            _await_completed_run(call, started["handle"])
    finally:
        bridge.disconnect()
        fx.ctrl.teardown_cfg_editor(editor_id)


@pytest.mark.parametrize("value_bytes", [2 << 20, MAX_LINE_BYTES - 2048])
def test_large_context_roundtrip_restores_load_guard(
    fx, tmp_path: Path, value_bytes: int
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        value = "x" * value_bytes
        call(
            "rpc_call",
            {
                "method": "context.md_set_attr",
                "params": {"key": "large", "value": value},
            },
        )
        _edit_context_as_gui(fx, "changed", 17)
        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"

        observed = call("rpc_call", {"method": "context.snapshot"})
        assert observed["md"]["large"] == value
        assert observed["md"]["changed"] == 17
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) != "stale_version"
    finally:
        bridge.disconnect()


def test_oversized_context_read_returns_error_without_advancing_guard(
    fx, tmp_path: Path
) -> None:
    _prepare_guarded_context(fx)
    tab_id = fx.ctrl.new_tab("fake")
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab_id}})
        call("rpc_call", {"method": "context.snapshot"})
        _edit_context_as_gui(fx, "large", "x" * (MAX_LINE_BYTES // 2))
        _edit_context_as_gui(fx, "other_large", "x" * (MAX_LINE_BYTES // 2))

        with pytest.raises(RuntimeError) as oversized:
            call("rpc_call", {"method": "context.snapshot"})
        assert getattr(oversized.value, "reason", None) == "response_encoding_failed"
        assert "may have executed" in str(oversized.value)
        args = {
            "method": "tab.load_data",
            "params": {"tab_id": tab_id, "data_path": "missing.h5"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"

        _edit_context_as_gui(fx, "large", "small")
        assert (
            call("rpc_call", {"method": "context.snapshot"})["md"]["large"] == "small"
        )
        with pytest.raises(RuntimeError) as missing_file:
            call("rpc_call", args)
        assert getattr(missing_file.value, "reason", None) != "stale_version"
    finally:
        bridge.disconnect()


def test_editor_commit_rejects_unseen_gui_edit_then_accepts_full_read(
    fx, tmp_path: Path
) -> None:
    ml = ModuleLibrary()
    ml.waveforms["seed"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    _prepare_guarded_context(fx, ml)
    bridge, call = _mcp_client(fx.service.port, tmp_path)
    try:
        call("connect", {"port": fx.service.port})
        editor_id = call(
            "rpc_call",
            {
                "method": "editor.new",
                "params": {"item_kind": "waveform", "from_name": "seed"},
            },
        )["editor_id"]
        call("rpc_call", {"method": "editor.get", "params": {"editor_id": editor_id}})
        call("rpc_call", {"method": "context.snapshot"})
        _edit_context_as_gui(fx, "r_f", 6000.0)
        args = {
            "method": "editor.commit",
            "params": {"editor_id": editor_id, "name": "copy"},
        }
        with pytest.raises(RuntimeError) as stale:
            call("rpc_call", args)
        assert getattr(stale.value, "reason", None) == "stale_version"
        call("rpc_call", {"method": "context.snapshot"})
        assert call("rpc_call", args) == {}
        assert ml.waveforms["copy"] == ml.waveforms["seed"]
    finally:
        bridge.disconnect()


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
