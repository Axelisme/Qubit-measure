"""gui_launch / gui_connect port-guard behaviour (MCP bridge).

These pin the fail-fast added so an agent can't mistake a stale GUI for a fresh
one: launch refuses an already-occupied port (instead of silently attaching to
whatever is there), and connect gives a clear error when nothing is listening.
"""

from __future__ import annotations

import socket

import pytest
from zcu_tools.gui.services.remote import mcp_server


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
