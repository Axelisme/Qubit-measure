"""Measure MCP connection and live GUI catalog contracts."""

import json
import os
import socket
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any, NoReturn, TypedDict

import psutil
import pytest
from zcu_tools.gui.app.measure.remote.method_entries import METHOD_ENTRIES
from zcu_tools.gui.app.measure.remote.method_entries._registry import (
    build_agent_catalog,
)
from zcu_tools.mcp.core import bridge as bridge_module
from zcu_tools.mcp.core.bridge import (
    DeliverFn,
    GuiTransportTimeoutError,
    OnClosedFn,
    SocketTransport,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client

# tolerated(capability): Real sockets verify framing, EOF, authentication, and
# same-port restart at the transport seam; recording transports cannot cover these.
# Real loopback disconnects are observed by the bridge's reader thread.
pytestmark = [pytest.mark.requires_loopback, pytest.mark.uses_wall_clock]


@pytest.fixture
def disconnected(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Notify after the real socket reader delivers EOF to the bridge."""
    observed = threading.Event()

    class NotifyingSocketTransport(SocketTransport):
        def attach(
            self,
            deliver_reply: DeliverFn,
            deliver_event: DeliverFn,
            on_closed: OnClosedFn,
        ) -> None:
            def closed(failure: Exception | None) -> None:
                on_closed(failure)
                observed.set()

            super().attach(deliver_reply, deliver_event, closed)

    # Keep real sockets, framing and EOF; only observe the transport callback seam.
    monkeypatch.setattr(bridge_module, "SocketTransport", NotifyingSocketTransport)
    return observed


def test_session_close_wakes_an_inflight_rpc_without_waiting_for_rpc_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    session = client.context.session
    binding = session.bind()
    sent = threading.Event()
    client.transport.sent.clear()

    def send(payload: dict[str, Any]) -> None:
        client.transport.sent.append((payload["method"], payload["params"]))
        sent.set()

    monkeypatch.setattr(client.transport, "send_line", send)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(
            binding.send_gui_rpc, "context.labels", {}, timeout_seconds=30.0
        )
        try:
            # The submitted RPC records transport dispatch before EOF or close. The 30s limit only guards deadlock.
            assert sent.wait(30)
            closing = pool.submit(session.close)
            # The submitted call completes after release/close; 30s only guards deadlock.
            closing.result(timeout=30)
            with pytest.raises(GuiRpcError) as error:
                # The submitted call completes after release/close; 30s only guards deadlock.
                pending.result(timeout=30)
            assert error.value.reason == "session_closed"
            assert not client.transport.is_open
        finally:
            client.context.bridge.disconnect()
    assert client.transport.sent == [("context.labels", {})]


class LoopbackExchange(TypedDict):
    """One wire request on a numbered socket attempt and its delivery outcome.

    attempt starts at one for each incarnation. reply is None for deliberate
    EOF; delivered is True only after the complete reply was sent.
    """

    attempt: int
    request: Mapping[str, object]
    reply: Mapping[str, object] | None
    delivered: bool


class LoopbackGui:
    """One disposable GUI wire incarnation, without a process or instrument."""

    def __init__(
        self, catalog: dict[str, Any], port: int = 0, *, token: str | None = None
    ) -> None:
        self._token = token
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", port))
        self._server.listen(1)
        self._server.settimeout(0.1)
        self.port = int(self._server.getsockname()[1])
        self._catalog = catalog
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self.sent: list[str] = []
        self.exchanges: list[LoopbackExchange] = []
        self.close_on: str | None = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._serving = self._pool.submit(self._serve)

    def _serve(self) -> None:
        attempt = 0
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = self._server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                self._socket = conn
                attempt += 1
                authenticated = self._token is None
                with conn, conn.makefile("rb") as stream:
                    for line in stream:
                        if self._stop.is_set():
                            return
                        request = json.loads(line)
                        exchange: LoopbackExchange = {
                            "attempt": attempt,
                            "request": request,
                            "reply": None,
                            "delivered": False,
                        }
                        self.exchanges.append(exchange)
                        method = request["method"]
                        self.sent.append(method)
                        if method == self.close_on:
                            return
                        if (
                            method == "auth"
                            and request["params"].get("token") == self._token
                        ):
                            authenticated = True
                            reply = {"id": request["id"], "ok": True, "result": {}}
                        elif method == "auth" or (
                            method != "wire.version" and not authenticated
                        ):
                            reply = {
                                "id": request["id"],
                                "ok": False,
                                "error": {
                                    "code": (
                                        "precondition_failed"
                                        if method == "auth" and self._token is None
                                        else "unauthorized"
                                    ),
                                    "message": "auth required",
                                },
                            }
                        else:
                            reply = {
                                "id": request["id"],
                                "ok": True,
                                "result": self._result(method, request["params"]),
                            }
                        exchange["reply"] = reply
                        try:
                            conn.sendall((json.dumps(reply) + "\n").encode())
                            exchange["delivered"] = True
                        except OSError:
                            return
        finally:
            self._server.close()

    def _result(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any]
        if method == "wire.version":
            result = {"wire_version": self._catalog["wire"], "gui_version": 83}
        elif method in {"control.connect", "control.status"}:
            expected = (
                {"label": "agent", "mode": "owner"}
                if method == "control.connect"
                else {}
            )
            if params != expected:
                raise AssertionError(f"Unexpected {method} params: {params}")
            result = {
                "holder": "rpc:agent",
                "locked": False,
                "online": True,
                "me": True,
            }
        elif method == "rpc.catalog":
            result = {"methods": self._catalog["methods"]}
        elif method == "resources.versions":
            result = {"versions": {}}
        elif method == "operation.active":
            result = {"operations": [{"op": 1, "tab": None, "kind": "device"}]}
        elif method == "operation.await":
            result = {"reason": "completed", "status": "finished"}
        elif method == "operation.cancel":
            result = {"status": "finished"}
        elif method in {"tab.run_cancel", "analyze.cancel", "device.cancel_operation"}:
            result = {"ok": True, "cancelled": True}
        else:
            result = overview_rpc(method, params)
        return result

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            with suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
        self._server.close()
        try:
            # Shutdown wakes recv/accept; 30s only guards deadlock in the producer.
            self._serving.result(timeout=30)
        finally:
            self._pool.shutdown(wait=True)


def assert_loopback_exchanges(
    exchanges: Sequence[LoopbackExchange],
    requests: Sequence[tuple[str, Mapping[str, object]]],
    replies: Sequence[Mapping[str, object]],
) -> None:
    """Compare a complete attempt's method/params, reply envelopes and delivery.

    Expected replies omit the request id; each reply must match its own request.
    """
    assert [
        (exchange["request"]["method"], exchange["request"]["params"])
        for exchange in exchanges
    ] == requests
    assert len(exchanges) == len(replies)
    for exchange, reply in zip(exchanges, replies, strict=True):
        assert exchange["reply"] == {"id": exchange["request"]["id"], **reply}
        assert exchange["delivered"] is True


def assert_auth_transcript(
    initial: Sequence[LoopbackExchange],
    restarted: Sequence[LoopbackExchange],
    *,
    wire_version: int,
    catalog: object,
) -> None:
    """Check this real socket composition's denied and successful auth attempts.

    The bridge and session each read wire.version; neither read is projected out.
    The initial valid attempt includes the connect tool's seven status reads.
    """
    assert {exchange["attempt"] for exchange in initial} == {1, 2, 3}
    assert {exchange["attempt"] for exchange in restarted} == {1}
    no_token, wrong_token, valid_token = (
        [exchange for exchange in initial if exchange["attempt"] == attempt]
        for attempt in (1, 2, 3)
    )
    wire = {
        "ok": True,
        "result": {"wire_version": wire_version, "gui_version": 83},
    }
    unauthorized = {
        "ok": False,
        "error": {"code": "unauthorized", "message": "auth required"},
    }
    identity = ("control.connect", {"label": "agent", "mode": "owner"})
    assert_loopback_exchanges(
        no_token,
        [("wire.version", {}), ("wire.version", {}), identity],
        [wire, wire, unauthorized],
    )
    assert_loopback_exchanges(
        wrong_token, [("auth", {"token": "wrong"})], [unauthorized]
    )
    handshake = [
        ("auth", {"token": "test-secret"}),
        ("wire.version", {}),
        ("wire.version", {}),
        identity,
        ("rpc.catalog", {}),
    ]
    handshake_replies = [
        {"ok": True, "result": {}},
        wire,
        wire,
        {
            "ok": True,
            "result": {
                "holder": "rpc:agent",
                "locked": False,
                "online": True,
                "me": True,
            },
        },
        {"ok": True, "result": {"methods": catalog}},
    ]
    status_reads = [
        "state.has_project",
        "state.has_active_context",
        "state.has_soc",
        "tab.snapshot",
        "context.active",
        "device.list",
        "operation.active",
    ]
    status_replies = [
        {"ok": True, "result": overview_rpc(method, {})} for method in status_reads[:-1]
    ] + [
        {
            "ok": True,
            "result": {"operations": [{"op": 1, "tab": None, "kind": "device"}]},
        }
    ]
    assert_loopback_exchanges(
        valid_token,
        handshake + [(method, {}) for method in status_reads],
        handshake_replies + status_replies,
    )
    assert_loopback_exchanges(restarted, handshake, handshake_replies)


CATALOG = [
    {
        "method": "adapter.list",
        "description": "List adapters",
        "params": {"type": "object", "properties": {}},
        "timeout_seconds": 5.0,
        "exposure": "rpc",
        "tool_names": [],
        "operation_key": None,
    },
    {
        "method": "adapter.guide",
        "description": "Read a guide",
        "params": {
            "type": "object",
            "properties": {"adapter_name": {"type": "string"}},
            "required": ["adapter_name"],
        },
        "timeout_seconds": 5.0,
        "exposure": "tool",
        "tool_names": ["guide"],
        "operation_key": None,
    },
]


def overview_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    replies: dict[str, dict[str, Any]] = {
        "state.has_project": {"value": False},
        "state.has_context": {"value": False},
        "state.has_active_context": {"value": False},
        "state.has_soc": {"value": False},
        "tab.snapshot": {"tabs": []},
        "context.active": {"label": None},
        "device.list": {"devices": []},
        "predictor.info": {"loaded": False},
        "operation.active": {"operations": []},
        "state.hardware_gate": {},
        "run.running_tab": {"tab_id": None},
        "view.snapshot": {"active_tab_id": None},
    }
    return replies[method]


def test_connect_loads_catalog_and_rpc_tools_route_by_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    client.transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {"methods": CATALOG},
    }
    client.transport.replies["adapter.list"] = {
        "ok": True,
        "result": {"adapters": ["fake"]},
    }

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(client.transport)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    result = client.call("connect", {"port": 9912})
    assert result["port"] == 9912
    assert result["launched"] is False
    assert result["versions"]["wire"] == client.context.config.wire_version
    assert result["status"]
    assert client.transport.sent[:3] == [
        ("wire.version", {}),
        ("control.connect", {"label": "agent", "mode": "owner"}),
        ("rpc.catalog", {}),
    ]
    assert client.call("rpc_list", {"domain": "adapter"})["methods"] == [
        {"method": "adapter.list", "description": "List adapters", "tool_names": []},
        {
            "method": "adapter.guide",
            "description": "Read a guide",
            "tool_names": ["guide"],
        },
    ]
    assert client.call("rpc_describe", {"method": "adapter.list"})["params"] == {
        "type": "object",
        "properties": {},
    }
    assert client.call("rpc_call", {"method": "adapter.list", "params": {}}) == {
        "adapters": ["fake"]
    }
    with pytest.raises(RuntimeError) as error:
        client.call("rpc_describe", {"method": "rpc.catalog"})
    assert getattr(error.value, "reason", None) == "unknown_method"
    assert all(
        method not in {"adapter.guide", "rpc.catalog"}
        for method, _ in client.transport.sent[3:]
    )


def test_unexpected_gui_eof_reconnects_same_port_without_replaying_mutation(
    tmp_path: Path,
    disconnected: threading.Event,
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    catalog = {"methods": build_agent_catalog(METHOD_ENTRIES)}
    gui_a = LoopbackGui({**catalog, "wire": client.context.config.wire_version})
    client.context.bridge.set_transport(None)
    gui_b: LoopbackGui | None = None
    try:
        old = client.call("connect", {"port": gui_a.port})["status"]["running"][0]["op"]
        gui_a.close_on = "tab.edit_cfg"
        with pytest.raises((ConnectionError, OSError, RuntimeError)):
            client.call(
                "rpc_call",
                {
                    "method": "tab.edit_cfg",
                    "params": {
                        "tab_id": "t",
                        "expected": {"cfg_id": "cfg-t", "revision": "0"},
                        "edits": [],
                    },
                },
            )
        gui_a.stop()
        gui_b = LoopbackGui(
            {**catalog, "wire": client.context.config.wire_version}, gui_a.port
        )
        # Stopping the peer delivers EOF through the reader callback; 30s guards deadlock.
        assert disconnected.wait(30), "socket EOF was not delivered to the bridge"
        assert not client.context.bridge.is_connected
        with pytest.raises(RuntimeError) as error:
            client.call("wait", {"op": old})
        assert getattr(error.value, "reason", None) == "unknown_op"
        new = client.call("connect", {"port": gui_a.port})["status"]["running"][0]["op"]
        assert new != old
        with pytest.raises(RuntimeError) as error:
            client.call("cancel", {"op": old})
        assert getattr(error.value, "reason", None) == "unknown_op"
        for method, params in (
            ("tab.run_cancel", {}),
            ("analyze.cancel", {"tab_id": "t"}),
            ("device.cancel_operation", {"name": "bias"}),
        ):
            with pytest.raises(RuntimeError) as error:
                client.call("rpc_call", {"method": method, "params": params})
            assert getattr(error.value, "reason", None) == "unknown_method"
            assert method not in gui_b.sent
        assert client.call("wait", {"op": new})["status"] == "finished"
        assert client.call("cancel", {"op": new})["status"] == "finished"
        assert gui_a.sent.count("tab.edit_cfg") == 1
        assert "tab.edit_cfg" not in gui_b.sent
        assert gui_b.sent.count("operation.await") == 1
        assert gui_b.sent.count("operation.cancel") == 1
    finally:
        client.context.bridge.disconnect()
        gui_a.stop()
        if gui_b is not None:
            gui_b.stop()


def test_token_protected_gui_attach_rejects_bad_credentials_and_reauthenticates_after_restart(
    tmp_path: Path,
    disconnected: threading.Event,
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    catalog = {"methods": build_agent_catalog(METHOD_ENTRIES)}
    gui_a = LoopbackGui(
        {**catalog, "wire": client.context.config.wire_version}, token="test-secret"
    )
    gui_b: LoopbackGui | None = None
    try:
        with pytest.raises(RuntimeError) as exc_info:
            client.call("connect", {"port": gui_a.port})
        assert getattr(exc_info.value, "reason", None) == "unauthorized"
        with pytest.raises(RuntimeError) as exc_info:
            client.call("connect", {"port": gui_a.port, "token": "wrong"})
        assert getattr(exc_info.value, "reason", None) == "unauthorized"
        rejected_catalog_count = gui_a.sent.count("rpc.catalog")

        connected = client.call("connect", {"port": gui_a.port, "token": "test-secret"})
        assert connected["port"] == gui_a.port
        assert client.call("rpc_list", {"domain": "adapter"})["methods"]
        gui_a.stop()
        gui_b = LoopbackGui(
            {**catalog, "wire": client.context.config.wire_version},
            gui_a.port,
            token="test-secret",
        )
        # Stopping the peer delivers EOF through the reader callback; 30s guards deadlock.
        assert disconnected.wait(30), "socket EOF was not delivered to the bridge"
        assert not client.context.bridge.is_connected
        assert client.call("rpc_list", {"domain": "adapter"})["methods"]
        # Join the reply producer before reading its final delivery record.
        gui_b.stop()
        assert gui_b.sent.count("auth") == 1
        assert gui_b.sent.count("rpc.catalog") == 1
        assert rejected_catalog_count == 0
        assert_auth_transcript(
            gui_a.exchanges,
            gui_b.exchanges,
            wire_version=client.context.config.wire_version,
            catalog=catalog["methods"],
        )
    finally:
        client.context.bridge.disconnect()
        gui_a.stop()
        if gui_b is not None:
            gui_b.stop()


def test_failed_second_launch_preserves_credential_for_reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disconnected: threading.Event,
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: port == gui_a.port)
    client.context.bridge.set_transport(None)
    catalog = {"methods": build_agent_catalog(METHOD_ENTRIES)}
    gui_a = LoopbackGui(
        {**catalog, "wire": client.context.config.wire_version}, token="original"
    )
    gui_b: LoopbackGui | None = None
    try:
        client.call("connect", {"port": gui_a.port, "token": "original"})
        monkeypatch.setattr(
            type(client.context.bridge), "launched_gui", property(lambda _: True)
        )
        with pytest.raises(RuntimeError) as exc_info:
            client.call(
                "connect",
                {"port": gui_a.port + 1, "launch": "new", "token": "replacement"},
            )
        assert getattr(exc_info.value, "reason", None) == "busy"
        gui_a.stop()
        gui_b = LoopbackGui(
            {**catalog, "wire": client.context.config.wire_version},
            gui_a.port,
            token="original",
        )
        # Stopping the peer delivers EOF through the reader callback; 30s guards deadlock.
        assert disconnected.wait(30), "socket EOF was not delivered to the bridge"
        assert not client.context.bridge.is_connected
        assert client.call("rpc_list", {"domain": "adapter"})["methods"]
        assert gui_b.sent.count("auth") == 1
    finally:
        client.context.bridge.disconnect()
        gui_a.stop()
        if gui_b is not None:
            gui_b.stop()


def test_token_on_unprotected_gui_reports_auth_disabled(tmp_path: Path) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    gui = LoopbackGui(
        {
            "methods": build_agent_catalog(METHOD_ENTRIES),
            "wire": client.context.config.wire_version,
        }
    )
    try:
        with pytest.raises(RuntimeError) as exc_info:
            client.call("connect", {"port": gui.port, "token": "unneeded"})
        assert getattr(exc_info.value, "reason", None) == "auth_disabled"
        assert client.call("connect", {"port": gui.port})["port"] == gui.port
    finally:
        client.context.bridge.disconnect()
        gui.stop()


@pytest.mark.parametrize("token", ["", False, 23, []])
def test_connect_rejects_invalid_token_before_network(
    tmp_path: Path, token: Any
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    with pytest.raises(ValueError, match="token must be a non-empty string"):
        client.call("connect", {"port": 9912, "token": token})
    assert not client.context.bridge.is_connected


def test_explicit_port_switch_does_not_send_previous_gui_token(tmp_path: Path) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    catalog = {"methods": build_agent_catalog(METHOD_ENTRIES)}
    protected = LoopbackGui(
        {**catalog, "wire": client.context.config.wire_version}, token="test-secret"
    )
    unprotected = LoopbackGui({**catalog, "wire": client.context.config.wire_version})
    try:
        client.call("connect", {"port": protected.port, "token": "test-secret"})
        assert (
            client.call("connect", {"port": unprotected.port})["port"]
            == unprotected.port
        )
        assert "auth" not in unprotected.sent
        assert client.call("rpc_list", {"domain": "adapter"})["methods"]
    finally:
        client.context.bridge.disconnect()
        protected.stop()
        unprotected.stop()


def test_incompatible_wire_refuses_catalog_and_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    client.transport.replies["wire.version"] = {
        "ok": True,
        "result": {"wire_version": -1, "gui_version": 100},
    }

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(client.transport)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    with pytest.raises(RuntimeError, match="wire"):
        client.call("connect", {"port": 9912})
    assert client.transport.sent == [("wire.version", {})]
    assert not client.context.bridge.is_connected


def test_reconnect_replaces_catalog_without_replaying_previous_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    first = client.transport
    first.replies["rpc.catalog"] = {"ok": True, "result": {"methods": CATALOG}}
    first.replies["adapter.list"] = {"ok": True, "result": {"adapters": ["old"]}}
    client.context.bridge.set_transport(None)
    next_transport = type(first)(overview_rpc)
    next_transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {
            "methods": [{**CATALOG[0], "method": "entry.info"}],
        },
    }
    next_transport.replies["entry.info"] = {"ok": True, "result": {"chip": "new"}}
    transports = iter((first, next_transport))
    ports: list[int] = []

    def connect(port: int, token: str | None = None) -> str:
        ports.append(port)
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    client.call("connect", {"port": 9912})
    assert client.call("rpc_call", {"method": "adapter.list", "params": {}}) == {
        "adapters": ["old"]
    }
    client.context.bridge.disconnect()
    assert client.call("rpc_call", {"method": "entry.info", "params": {}}) == {
        "chip": "new"
    }
    with pytest.raises(RuntimeError) as error:
        client.call("rpc_call", {"method": "adapter.list"})
    assert getattr(error.value, "reason", None) == "unknown_method"
    assert [method for method, _ in first.sent].count("adapter.list") == 1
    assert all(method != "adapter.list" for method, _ in next_transport.sent)
    assert ports == [9912, 9912]


def test_discovered_gui_can_restart_on_a_different_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered = iter((9911, 9913))
    client = make_client(
        tmp_path,
        overview_rpc,
        resolve_connect_port=lambda config, requested: (
            requested if requested is not None else next(discovered)
        ),
        port_is_open=lambda port: True,
    )
    client.context.bridge.set_transport(None)
    first = client.transport
    first.replies["rpc.catalog"] = {"ok": True, "result": {"methods": CATALOG}}
    second = type(first)(overview_rpc)
    second.replies["rpc.catalog"] = {"ok": True, "result": {"methods": CATALOG}}
    second.replies["adapter.list"] = {"ok": True, "result": {"adapters": ["new"]}}
    transports = iter((first, second))
    ports: list[int] = []

    def connect(port: int, token: str | None = None) -> str:
        ports.append(port)
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    assert client.call("connect", {})["port"] == 9911
    client.context.bridge.disconnect()
    assert client.call("rpc_call", {"method": "adapter.list"}) == {
        "adapters": ["new"],
    }
    assert ports == [9911, 9913]


def test_ambiguous_mutation_timeout_does_not_replay_on_reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    first = client.transport
    second = type(first)(overview_rpc)
    second.replies["rpc.catalog"] = {
        "ok": True,
        "result": {
            "methods": [
                {**CATALOG[0], "method": "entry.info"},
            ]
        },
    }
    second.replies["entry.info"] = {"ok": True, "result": {"chip": "new"}}
    client.context.session.ensure_connected()
    real_send = client.context.bridge.send_rpc_raw
    attempted: list[str] = []

    def send(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        if method == "adapter.list":
            attempted.append(method)
            client.context.bridge.disconnect()
            raise GuiTransportTimeoutError(method, timeout_seconds)
        return real_send(method, params, timeout_seconds)

    def reconnect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(second)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", send)
    monkeypatch.setattr(client.context.bridge, "connect", reconnect)
    with pytest.raises(RuntimeError) as error:
        client.call("rpc_call", {"method": "adapter.list", "params": {}})
    assert getattr(error.value, "reason", None) == "gui_transport_timeout"
    assert client.call("rpc_call", {"method": "entry.info"}) == {"chip": "new"}
    assert attempted == ["adapter.list"]
    assert all(method != "adapter.list" for method, _ in second.sent)


def test_response_encoding_error_does_not_replay_mutation(tmp_path: Path) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    client.transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {"methods": [{**CATALOG[0], "method": "project.save"}]},
    }
    client.transport.replies["project.save"] = {
        "ok": False,
        "error": {
            "code": "internal",
            "reason": "response_encoding_failed",
            "message": "The request may have executed; inspect state before retrying.",
        },
    }
    with pytest.raises(RuntimeError) as error:
        client.call("rpc_call", {"method": "project.save"})
    assert getattr(error.value, "reason", None) == "response_encoding_failed"
    client.call("rpc_list", {})
    assert [method for method, _ in client.transport.sent].count("project.save") == 1


def test_connect_switches_an_explicit_port_and_expires_operation_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    first = client.transport
    first.replies["rpc.catalog"] = {
        "ok": True,
        "result": {
            "methods": CATALOG
            + [{**CATALOG[0], "method": "test.start", "operation_key": "tab:old"}]
        },
    }
    first.replies["test.start"] = {"ok": True, "result": {"operation_id": 43}}
    other = type(first)(overview_rpc)
    other.replies["rpc.catalog"] = {
        "ok": True,
        "result": {
            "methods": [
                {**CATALOG[0], "method": "entry.info"},
            ]
        },
    }
    ports: list[int] = []
    transports = iter((first, other))

    def connect(port: int, token: str | None = None) -> str:
        ports.append(port)
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    client.call("connect", {"port": 9911})
    handle = client.call("rpc_call", {"method": "test.start"})["handle"]
    assert client.context.session.operation_handles == {"tab:old": handle}
    client.call("connect", {"port": 9912})
    assert ports == [9911, 9912]
    assert not client.context.session.operation_handles
    assert client.call("rpc_list", {})["methods"] == [
        {"method": "entry.info", "description": "List adapters", "tool_names": []},
    ]


def test_connect_launch_modes_do_not_start_hardware_or_kill_an_existing_gui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: False)
    client.context.bridge.set_transport(None)
    client.transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {"methods": CATALOG},
    }
    calls: list[tuple[int, list[str] | None, str | None]] = []

    def launch(
        repo_root: Path,
        port: int,
        token: str | None = None,
        auto_connect: bool = True,
        extra_args: list[str] | None = None,
    ) -> str:
        calls.append((port, extra_args, token))
        client.context.bridge.set_transport(client.transport)
        return "launched"

    monkeypatch.setattr(client.context.bridge, "launch", launch)
    with pytest.raises(RuntimeError) as error:
        client.call("connect", {})
    assert getattr(error.value, "reason", None) == "no_gui"
    assert calls == []
    result = client.call(
        "connect", {"launch": "if_missing", "clean": True, "token": "test-secret"}
    )
    assert result["launched"] is True
    assert calls == [(8765, ["--clean"], "test-secret")]
    assert all(method != "soc.connect" for method, _ in client.transport.sent)

    busy = make_client(tmp_path, port_is_open=lambda port: True)
    busy.context.bridge.set_transport(None)
    with pytest.raises(RuntimeError) as error:
        busy.call("connect", {"launch": "new"})
    assert getattr(error.value, "reason", None) == "port_in_use"


def test_connect_busy_port_identifies_listener_and_port_parameter(
    tmp_path: Path,
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        client = make_client(tmp_path, port_is_open=lambda _: True)
        try:
            with pytest.raises(GuiRpcError) as error:
                client.call("connect", {"launch": "new", "port": port})
            assert error.value.reason == "port_in_use"
            assert f"pid={os.getpid()}" in str(error.value)
            assert "port parameter" in str(error.value)
            assert str(port) in str(error.value)
            assert client.transport.sent == []
        finally:
            client.context.session.close()


def test_connect_busy_port_retains_actionable_error_when_owner_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(kind: str) -> NoReturn:
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "net_connections", denied)
    client = make_client(tmp_path, port_is_open=lambda _: True)
    try:
        with pytest.raises(GuiRpcError) as error:
            client.call("connect", {"launch": "new"})
        assert error.value.reason == "port_in_use"
        assert "owner unknown" in str(error.value)
        assert "port parameter" in str(error.value)
        assert client.transport.sent == []
    finally:
        client.context.session.close()


@pytest.mark.parametrize("launch", ["new", "if_missing"])
def test_connect_refuses_second_port_while_launched_gui_is_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch: str
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: False)
    first = client.transport
    client.context.bridge.set_transport(None)
    launches: list[int] = []

    def fake_launch(
        repo_root: Path,
        port: int,
        token: str | None = None,
        auto_connect: bool = True,
        extra_args: list[str] | None = None,
    ) -> str:
        launches.append(port)
        if len(launches) == 1:
            client.context.bridge.set_transport(first)
            return "launched"
        return "GUI already running"

    monkeypatch.setattr(client.context.bridge, "launch", fake_launch)
    assert client.call("connect", {"port": 9911, "launch": "new"})["port"] == 9911
    monkeypatch.setattr(
        type(client.context.bridge), "launched_gui", property(lambda _: True)
    )
    with pytest.raises(RuntimeError) as exc_info:
        client.call("connect", {"port": 9912, "launch": launch})
    assert getattr(exc_info.value, "reason", None) == "busy"
    assert launches == [9911]
    assert client.context.bridge.is_connected
    assert client.call("connect", {"port": 9911})["port"] == 9911


@pytest.mark.parametrize(
    "catalog",
    [
        *(
            {
                "methods": [
                    {**CATALOG[0], "method": "rejected.entry"},
                    {**CATALOG[0], field: invalid},
                ]
            }
            for field in ("tool_names",)
            for invalid in (None, "not-a-list", [1], [""])
        ),
        *(
            {"methods": [{**CATALOG[0], "params": invalid}]}
            for invalid in (None, [], {}, {"type": "array"})
        ),
        {"methods": "invalid"},
        {"methods": [CATALOG[0], CATALOG[0]]},
    ],
)
def test_connect_rejects_malformed_or_duplicate_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    catalog: dict[str, Any],
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    client.transport.replies["rpc.catalog"] = {"ok": True, "result": catalog}

    def connect(port: int, token: str | None = None) -> str:
        client.transport.is_open = True
        client.context.bridge.set_transport(client.transport)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    with pytest.raises(RuntimeError, match="catalog") as error:
        client.call("connect", {"port": 9912})
    assert getattr(error.value, "reason", None) == "incompatible_wire"
    assert not client.context.bridge.is_connected

    client.transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {"methods": CATALOG},
    }
    client.call("connect", {"port": 9912})
    methods = client.call("rpc_list", {})["methods"]
    assert [entry["method"] for entry in methods] == [
        entry["method"] for entry in CATALOG
    ]
    with pytest.raises(RuntimeError) as missing:
        client.call("rpc_describe", {"method": "rejected.entry"})
    assert getattr(missing.value, "reason", None) == "unknown_method"
