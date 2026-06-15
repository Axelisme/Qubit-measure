"""Tests for McpBridge internals.

``launched_gui`` — the cleanup-on-exit ownership guard.
``launched_gui`` distinguishes a GUI this bridge *launched* (``gui_launch`` sets
``_proc``) from one it merely *attached* to (lazy auto-connect leaves ``_proc``
None). The exit-cleanup path guards on it so an attach-only server never stops a
GUI another process owns — the bug being that closing the external-terminal
agent killed the user's GUI via the shared pid-file fallback in ``stop()``.

``_port_is_open`` — the fast-fail probe used by ``_ensure_connected``.
Used before a full TCP connect in ``_ensure_connected`` so that a cold start
(no GUI listening) returns an actionable error in ~0.5s instead of hanging
~30s until the socket timeout fires.
"""

from __future__ import annotations

import socket
from pathlib import Path

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig, _port_is_open


def _config(tmp_path: Path) -> MCPBridgeConfig:
    return MCPBridgeConfig(
        tool_prefix="test_",
        server_display_name="test-control",
        server_instructions="",
        app_name="test",
        app_slug="test",
        default_port=18765,
        mcp_version=1,
        wire_version=1,
        pid_file=tmp_path / "test_gui.pid",
        log_file=tmp_path / "test_gui.log",
        run_script_name="run_test_gui.py",
    )


class _FakeProc:
    """Minimal subprocess.Popen stand-in: only ``poll`` is exercised here."""

    def __init__(self, *, alive: bool) -> None:
        self._alive = alive

    def poll(self) -> int | None:
        return None if self._alive else 0


def test_launched_gui_false_when_attached_only(tmp_path: Path) -> None:
    # Lazy auto-connect attaches without launching -> _proc stays None -> not ours.
    bridge = McpBridge(_config(tmp_path))
    assert bridge._proc is None
    assert bridge.launched_gui is False


def test_launched_gui_true_when_we_launched_a_live_proc(tmp_path: Path) -> None:
    bridge = McpBridge(_config(tmp_path))
    bridge._proc = _FakeProc(alive=True)  # type: ignore[assignment]
    assert bridge.launched_gui is True


def test_launched_gui_false_when_our_proc_exited(tmp_path: Path) -> None:
    # We launched it but it already exited -> nothing live to stop.
    bridge = McpBridge(_config(tmp_path))
    bridge._proc = _FakeProc(alive=False)  # type: ignore[assignment]
    assert bridge.launched_gui is False


def test_launched_gui_ignores_shared_pid_file(tmp_path: Path) -> None:
    # The bug: an attach-only bridge whose (shared) pid file points at a GUI that
    # another process launched must still report launched_gui False, so the
    # exit-cleanup path skips stop() and leaves that GUI alone. Writing the pid
    # file must NOT flip the verdict — only our own live _proc counts.
    cfg = _config(tmp_path)
    cfg.pid_file.write_text("4242")
    bridge = McpBridge(cfg)
    assert bridge._read_pid_file() == 4242  # pid file is readable...
    assert bridge.launched_gui is False  # ...but launched_gui ignores it


# ---------------------------------------------------------------------------
# _port_is_open — fast-fail probe for _ensure_connected (BUG-3 fix)
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to port 0 to get a free port number, then release it.

    There is a brief TOCTOU window between release and the test assertion, but
    for a loopback-only test this is acceptable (no service binds the port in CI).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_port_is_open_returns_false_for_closed_port() -> None:
    # _ensure_connected calls _port_is_open before the full TCP connect so a
    # cold start (no GUI on the port) fails fast (~0.5s) instead of hanging ~30s.
    port = _find_free_port()
    # Nothing is listening on the port — probe must return False immediately.
    assert _port_is_open(port) is False


def test_port_is_open_returns_true_for_listening_port() -> None:
    # Sanity-check the positive path: a real listening socket is detected.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = int(srv.getsockname()[1])
        assert _port_is_open(port) is True
