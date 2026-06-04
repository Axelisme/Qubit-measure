"""gui_launch / gui_connect port-guard behaviour (MCP bridge).

These pin the fail-fast added so an agent can't mistake a stale GUI for a fresh
one: launch refuses an already-occupied port (instead of silently attaching to
whatever is there), and connect gives a clear error when nothing is listening.
"""

from __future__ import annotations

import socket

import pytest
from zcu_tools.gui.app.main.services.remote import mcp_server


@pytest.fixture
def busy_port():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    yield srv.getsockname()[1]
    srv.close()


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # closed → nothing listening
    return port


def test_port_is_open_detects_listener(busy_port):
    assert mcp_server._port_is_open(busy_port) is True
    assert mcp_server._port_is_open(_free_port()) is False


def test_launch_refuses_occupied_port(busy_port):
    # A listener already owns the port → launching there must fail fast, not
    # connect to the foreign process.
    with pytest.raises(RuntimeError, match="already in use"):
        mcp_server.tool_gui_launch({"port": busy_port})


class _FakeProc:
    pid = 4242
    stderr = None

    def poll(self):  # alive
        return None


def _capture_launch_cmd(monkeypatch, arguments) -> list[str]:
    """Run tool_gui_launch with all side effects stubbed, return the spawned cmd."""
    captured: dict[str, list[str]] = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    # _port_is_open is consulted twice with opposite expectations: the
    # pre-flight wants it free (False → proceed to spawn), the post-spawn
    # readiness wait wants it open (True → ready). Return False once, then True.
    calls = {"n": 0}

    def fake_port_is_open(_port):
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(mcp_server, "_port_is_open", fake_port_is_open)
    monkeypatch.setattr(mcp_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mcp_server, "_write_pid_file", lambda _pid: None)
    monkeypatch.setattr(mcp_server, "_GUI_PROC", None)
    try:
        # auto_connect=False so we never open a real socket; _port_is_open=True
        # makes the readiness wait return immediately.
        mcp_server.tool_gui_launch({**arguments, "auto_connect": False})
    finally:
        monkeypatch.setattr(mcp_server, "_GUI_PROC", None)
    return captured["cmd"]


def test_launch_clean_flag_adds_clean_arg(monkeypatch):
    cmd = _capture_launch_cmd(monkeypatch, {"port": 8799, "clean": True})
    assert "--clean" in cmd


def test_launch_without_clean_omits_clean_arg(monkeypatch):
    cmd = _capture_launch_cmd(monkeypatch, {"port": 8799})
    assert "--clean" not in cmd


def test_connect_errors_clearly_when_no_gui():
    with pytest.raises(RuntimeError, match="No GUI is listening"):
        mcp_server.tool_gui_connect({"port": _free_port()})


def test_connect_port_defaults_to_8765():
    # Missing port no longer raises a "missing argument" error; it defaults to
    # 8765. (Skip if something is actually listening on 8765 — e.g. a live GUI
    # during interactive testing — so the assertion stays about the default, not
    # the environment.)
    if mcp_server._port_is_open(8765):
        pytest.skip("port 8765 is in use (live GUI); cannot assert the no-GUI path")
    with pytest.raises(RuntimeError, match="No GUI is listening on 127.0.0.1:8765"):
        mcp_server.tool_gui_connect({})
