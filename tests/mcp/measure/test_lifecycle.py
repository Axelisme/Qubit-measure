"""Measure lifecycle handlers attach and subscribe without taking over hardware."""

import socket
from pathlib import Path
from typing import Any

import pytest

from ._support import MeasureClient, make_client


def overview_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    replies: dict[str, dict[str, Any]] = {
        "state.has_project": {"value": False},
        "state.has_context": {"value": False},
        "state.has_active_context": {"value": False},
        "state.has_soc": {"value": False},
        "tab.snapshot": {"tabs": []},
        "context.active": {"label": None},
        "state.hardware_gate": {},
        "run.running_tab": {"tab_id": None},
        "view.snapshot": {"active_tab_id": None},
    }
    return replies[method]


def configure_events(client: MeasureClient) -> None:
    client.transport.replies.update(
        {
            "events.list": {
                "ok": True,
                "result": {"events": ["tab_added", "run_finished"]},
            },
            "events.subscribe": {"ok": True, "result": {}},
        }
    )


def test_lazy_attach_subscribes_before_forwarding_without_starting_soc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(
        tmp_path,
        overview_rpc,
        resolve_connect_port=lambda config, requested: 9911,
        port_is_open=lambda port: port == 9911,
    )
    client.context.bridge.set_transport(None)
    configure_events(client)
    connections: list[tuple[int, str | None]] = []

    def connect(port: int, token: str | None = None) -> str:
        connections.append((port, token))
        client.context.bridge.set_transport(client.transport)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    assert client.context.send_gui_rpc("state.has_soc", {}) == {"value": False}
    assert connections == [(9911, None)]
    assert client.transport.sent[:3] == [
        ("events.list", {}),
        ("events.subscribe", {"events": ["tab_added", "run_finished"]}),
        ("state.has_soc", {}),
    ]
    assert all(method != "soc.connect" for method, _ in client.transport.sent)
    client.context.send_gui_rpc("state.has_soc", {})
    assert connections == [(9911, None)]


@pytest.mark.parametrize(
    ("method", "reply", "message"),
    [
        ("events.list", {"ok": False}, "events.list failed"),
        (
            "events.list",
            {"ok": True, "result": {"events": "invalid"}},
            "invalid catalog",
        ),
        ("events.subscribe", {"ok": False}, "events.subscribe failed"),
    ],
)
def test_failed_subscription_closes_transport_and_clears_pending_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    reply: dict[str, Any],
    message: str,
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    configure_events(client)
    client.transport.replies[method] = reply

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(client.transport)
        client.context.session.deliver_event({"event": "tab_added"})
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    with pytest.raises(RuntimeError, match=message):
        client.context.send_gui_rpc("state.has_soc", {})
    assert not client.context.bridge.is_connected
    assert not client.transport.is_open
    assert client.context.session.drain_pending() == {"diagnostics": [], "events": []}


@pytest.mark.parametrize("listener_open", [False, True])
def test_lazy_attach_reports_no_gui_on_probe_or_connection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    listener_open: bool,
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: listener_open)
    client.context.bridge.set_transport(None)

    def connect(port: int, token: str | None = None) -> str:
        raise RuntimeError(f"No GUI is listening on 127.0.0.1:{port}")

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    with pytest.raises(
        RuntimeError, match="no running measure-gui found to attach to"
    ) as error:
        client.context.send_gui_rpc("state.has_soc", {})
    assert "gui_launch" in str(error.value)


@pytest.mark.parametrize("requested", [None, 9912])
def test_explicit_connect_uses_discovered_or_requested_port_and_folds_overview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: int | None,
) -> None:
    client = make_client(tmp_path, overview_rpc)
    client.context.bridge.set_transport(None)
    configure_events(client)
    connections: list[tuple[int, str | None]] = []

    def connect(port: int, token: str | None = None) -> str:
        connections.append((port, token))
        client.context.bridge.set_transport(client.transport)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    arguments: dict[str, Any] = {"token": "secret"}
    if requested is not None:
        arguments["port"] = requested
    result = client.call("gui_bridge_connect", arguments)
    assert connections == [(8765 if requested is None else requested, "secret")]
    assert result["note"] == "connected"
    assert result["overview"]["soc"] == {"connected": False, "is_mock": None}
    assert client.transport.sent[:2] == [
        ("events.list", {}),
        ("events.subscribe", {"events": ["tab_added", "run_finished"]}),
    ]


@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("auto_connect", [False, True])
def test_launch_forwards_clean_and_only_subscribes_when_attached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean: bool,
    auto_connect: bool,
) -> None:
    client = make_client(tmp_path, overview_rpc)
    client.context.bridge.set_transport(None)
    configure_events(client)
    calls: list[dict[str, Any]] = []

    def launch(
        repo_root: Path,
        port: int,
        token: str | None = None,
        auto_connect: bool = True,
        extra_args: list[str] | None = None,
    ) -> str:
        calls.append(
            {
                "port": port,
                "token": token,
                "auto_connect": auto_connect,
                "extra_args": extra_args,
            }
        )
        if auto_connect:
            client.context.bridge.set_transport(client.transport)
        return "launched"

    monkeypatch.setattr(client.context.bridge, "launch", launch)
    result = client.call(
        "gui_launch",
        {
            "port": 9912,
            "token": "secret",
            "clean": clean,
            "auto_connect": auto_connect,
        },
    )
    assert calls == [
        {
            "port": 9912,
            "token": "secret",
            "auto_connect": auto_connect,
            "extra_args": ["--clean"] if clean else None,
        }
    ]
    assert result["note"] == "launched"
    if auto_connect:
        assert result["overview"]["soc"]["connected"] is False
        assert client.transport.sent[0] == ("events.list", {})
    else:
        assert result == {"note": "launched"}
        assert client.transport.sent == []


def test_launch_rejects_an_occupied_port_before_starting_a_process(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with pytest.raises(RuntimeError, match="already in use"):
            client.call("gui_launch", {"port": listener.getsockname()[1]})
