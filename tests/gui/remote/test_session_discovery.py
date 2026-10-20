"""Tests for the per-app GUI session discovery helper.

Covers the write->read round-trip, stale self-healing (dead pid / closed socket),
malformed-file tolerance, and the connect-port resolution used by the MCP connect
tools (explicit port wins, else discovery, else convention default).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from zcu_tools.gui.remote import session_discovery as sd

# tolerated(capability): Real listeners and TCP connections verify session-port
# liveness and stale-session cleanup after close. A recording socket would replace
# the external seam under test.
pytestmark = pytest.mark.requires_loopback


@pytest.fixture
def session_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate discovery through the public cache-root environment override."""
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    return tmp_path / "sessions"


def test_session_directory_uses_shared_cache_root(session_root: Path) -> None:
    assert sd.session_dir() == session_root


@pytest.mark.parametrize("override", ["", "relative/cache"])
def test_session_directory_rejects_invalid_cache_override(
    override: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", override)
    with pytest.raises(ValueError, match="ZCU_TOOLS_CACHE_DIR"):
        sd.session_dir()


def _open_listener() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def test_write_read_round_trip(session_root: Path) -> None:
    """A live session (this process + an open socket) reads back its fields."""
    listener, port = _open_listener()
    try:
        sd.write_session(
            "measure",
            port,
            pid=os.getpid(),
            host="127.0.0.1",
            wire_version=23,
            started="2026-06-13T00:00:00+00:00",
        )
        entry = sd.read_session("measure")
        assert entry is not None
        assert entry["app"] == "measure"
        assert entry["port"] == port
        assert entry["pid"] == os.getpid()
        assert entry["host"] == "127.0.0.1"
        assert entry["wire_version"] == 23
    finally:
        listener.close()


def test_read_missing_returns_none(session_root: Path) -> None:
    assert sd.read_session("measure") is None


@pytest.mark.requires_subprocess
def test_stale_dead_pid_is_swept(session_root: Path) -> None:
    """A live socket but a dead pid is stale: read returns None and deletes file."""
    listener, port = _open_listener()
    try:
        # The pass-only child exits without consulting the liveness implementation.
        with subprocess.Popen([sys.executable, "-c", "pass"]) as child:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                raise
        dead_pid = child.pid
        sd.write_session(
            "fluxdep",
            port,
            pid=dead_pid,
            host="127.0.0.1",
            wire_version=1,
            started="2026-06-13T00:00:00+00:00",
        )
        assert sd.read_session("fluxdep") is None
        assert not (session_root / "fluxdep.json").exists(), (
            "stale file must be deleted on read"
        )
    finally:
        listener.close()


def test_stale_closed_socket_is_swept(session_root: Path) -> None:
    """A live pid but an unreachable socket is stale: read returns None + deletes."""
    listener, port = _open_listener()
    listener.close()  # nothing listens on this port now
    sd.write_session(
        "dispersive",
        port,
        pid=os.getpid(),
        host="127.0.0.1",
        wire_version=1,
        started="2026-06-13T00:00:00+00:00",
    )
    assert sd.read_session("dispersive") is None
    assert not (session_root / "dispersive.json").exists()


def test_corrupt_file_is_swept(session_root: Path) -> None:
    """A non-JSON file is treated as no session and removed."""
    session_root.mkdir(parents=True, exist_ok=True)
    (session_root / "measure.json").write_text("{ not json", encoding="utf-8")
    assert sd.read_session("measure") is None
    assert not (session_root / "measure.json").exists()


def test_malformed_entry_is_swept(session_root: Path) -> None:
    """Valid JSON missing required fields is rejected and removed."""
    session_root.mkdir(parents=True, exist_ok=True)
    (session_root / "measure.json").write_text(
        json.dumps({"app": "measure", "port": 8765}), encoding="utf-8"
    )
    assert sd.read_session("measure") is None
    assert not (session_root / "measure.json").exists()


def test_clear_session_is_idempotent(session_root: Path) -> None:
    sd.clear_session("measure")  # no file yet — must not raise
    sd.write_session(
        "measure",
        8765,
        pid=os.getpid(),
        host="127.0.0.1",
        wire_version=1,
        started="2026-06-13T00:00:00+00:00",
    )
    sd.clear_session("measure")
    assert sd.read_session("measure") is None


# ---------------------------------------------------------------------------
# resolve_connect_port: explicit > discovery > convention default
# ---------------------------------------------------------------------------


def _bridge_config(app_slug: str, default_port: int):
    from zcu_tools.mcp.core.bridge import MCPBridgeConfig

    return MCPBridgeConfig(
        app_name=app_slug,
        app_slug=app_slug,
        tool_prefix=f"{app_slug}_",
        default_port=default_port,
        mcp_version=1,
        wire_version=1,
        server_display_name=f"{app_slug}-control",
        server_instructions="",
        pid_file=Path("/tmp/_unused.pid"),
        log_file=Path("/tmp/_unused.log"),
        run_script_name=f"run_{app_slug}_gui.py",
    )


def test_resolve_explicit_port_wins(session_root: Path) -> None:
    from zcu_tools.mcp.core.bridge import resolve_connect_port

    cfg = _bridge_config("measure", 8765)
    assert resolve_connect_port(cfg, 9999) == 9999


def test_resolve_uses_discovery_when_omitted(session_root: Path) -> None:
    from zcu_tools.mcp.core.bridge import resolve_connect_port

    listener, port = _open_listener()
    try:
        sd.write_session(
            "measure",
            port,
            pid=os.getpid(),
            host="127.0.0.1",
            wire_version=1,
            started="2026-06-13T00:00:00+00:00",
        )
        cfg = _bridge_config("measure", 8765)
        assert resolve_connect_port(cfg, None) == port
    finally:
        listener.close()


def test_resolve_falls_back_to_default_without_discovery(
    session_root: Path,
) -> None:
    from zcu_tools.mcp.core.bridge import resolve_connect_port

    cfg = _bridge_config("measure", 8765)
    assert resolve_connect_port(cfg, None) == 8765
