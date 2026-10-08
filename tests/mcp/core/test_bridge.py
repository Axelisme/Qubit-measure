"""McpBridge ownership, transport failure, and framing contracts."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from zcu_tools.gui.remote.framing import MAX_LINE_BYTES, encode_line
from zcu_tools.mcp.core.bridge import (
    GuiMessageTooLargeError,
    GuiTransportTimeoutError,
    McpBridge,
    MCPBridgeConfig,
    SocketTransport,
)


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


class _SilentTransport:
    """Transport fake that records requests but never delivers replies."""

    def __init__(self) -> None:
        self.closed = False
        self.sent: list[dict[str, Any]] = []
        self._on_closed: Callable[[Exception | None], None] | None = None

    def attach(self, deliver_reply, deliver_event, on_closed) -> None:
        del deliver_reply, deliver_event
        self._on_closed = on_closed

    @property
    def is_open(self) -> bool:
        return not self.closed

    def send_line(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True


class _ObservedTransport(_SilentTransport):
    """In-process wire peer with explicit request/reply and EOF barriers."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: Queue[dict[str, Any]] = Queue()
        self.release_send = Event()
        self._reply: Callable[[dict[str, Any]], None] | None = None

    def attach(self, deliver_reply, deliver_event, on_closed) -> None:
        super().attach(deliver_reply, deliver_event, on_closed)
        self._reply = deliver_reply

    def send_line(self, payload: dict[str, Any]) -> None:
        super().send_line(payload)
        self.requests.put(payload)
        if payload["method"] == "failing.send":
            assert self.release_send.wait(timeout=5)
            raise ConnectionError("send failed")

    def reply(self, request: dict[str, Any]) -> None:
        assert self._reply is not None
        self._reply({"id": request["id"], "ok": True, "result": request["method"]})

    def unexpected_close(self) -> None:
        self.close()
        assert self._on_closed is not None
        self._on_closed(ConnectionError("connection lost"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-liveness seam")
@pytest.mark.parametrize("alive", [False, True])
@pytest.mark.parametrize("stale_owned_process", [False, True])
def test_wait_for_gui_exit_observes_reply_pid_without_termination(
    tmp_path, monkeypatch, alive, stale_owned_process
):
    config = _config(tmp_path)
    config.pid_file.write_text("9999")
    bridge = McpBridge(config)
    if stale_owned_process:
        exited = Mock(pid=1234)
        exited.poll.return_value = 0
        exited.wait.side_effect = AssertionError("old process handle is not this GUI")
        monkeypatch.setattr(bridge, "_proc", exited)
    clock = [0.0]
    probes = []

    def probe(pid, signal):
        probes.append((pid, signal))
        assert signal == 0
        if not alive:
            raise ProcessLookupError

    def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr("zcu_tools.mcp.core.bridge.os.kill", probe)
    monkeypatch.setattr("zcu_tools.mcp.core.bridge.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("zcu_tools.mcp.core.bridge.time.sleep", sleep)
    assert bridge.wait_for_gui_exit(1234, timeout=0.4) is (not alive)
    assert probes and all(pid == 1234 for pid, _ in probes)


@pytest.mark.parametrize("times_out", [False, True])
def test_wait_for_owned_gui_uses_process_wait_without_termination(
    tmp_path, monkeypatch, times_out
):
    from subprocess import TimeoutExpired

    bridge = McpBridge(_config(tmp_path))
    process = Mock(pid=1234)
    process.poll.return_value = None
    process.wait.side_effect = TimeoutExpired("gui", 2.0) if times_out else None
    monkeypatch.setattr(bridge, "_proc", process)
    assert bridge.wait_for_gui_exit(1234, timeout=2.0) is (not times_out)
    process.wait.assert_called_once_with(timeout=2.0)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_launched_gui_false_when_attached_only(tmp_path: Path) -> None:
    # Lazy auto-connect attaches without launching -> _proc stays None -> not ours.
    bridge = McpBridge(_config(tmp_path))
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
    assert bridge.launched_gui is False


class _ChildStderr:
    """Popen stand-in whose child writes ``output`` to the stderr it receives.

    A real child blocks once an unread pipe fills, so the stand-in only accepts
    a writable file handle.
    """

    def __init__(self, proc: MagicMock) -> None:
        self.proc = proc
        self.output = b""

    def __call__(self, cmd: list[str], **kwargs: Any) -> MagicMock:
        del cmd
        stderr = kwargs["stderr"]
        if stderr in (None, subprocess.PIPE):
            raise AssertionError(f"GUI stderr must go to a file, got {stderr!r}")
        stderr.write(self.output)
        return self.proc


@pytest.fixture
def launch_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[McpBridge, _ChildStderr, Mock, Mock]:
    config = _config(tmp_path)
    script = tmp_path / "scripts" / config.run_script_name
    script.parent.mkdir()
    script.touch()
    bridge = McpBridge(config)
    proc = MagicMock(pid=4242)
    proc.poll.return_value = None
    child = _ChildStderr(proc)
    probe = Mock(side_effect=[False, True])
    connect = Mock(return_value="connected")
    monkeypatch.setattr("zcu_tools.mcp.core.bridge.subprocess.Popen", child)
    monkeypatch.setattr("zcu_tools.mcp.core.bridge.port_is_open", probe)
    monkeypatch.setattr(bridge, "connect", connect)
    return bridge, child, probe, connect


@pytest.mark.parametrize("auto_connect", [False, True])
def test_launch_ready_preserves_connection_choice(
    launch_bridge: tuple[McpBridge, _ChildStderr, Mock, Mock],
    tmp_path: Path,
    *,
    auto_connect: bool,
) -> None:
    bridge, _, _, connect = launch_bridge
    result = bridge.launch(tmp_path, 18765, "credential", auto_connect=auto_connect)
    assert "listening on port 18765" in result
    assert bridge.launched_gui
    if auto_connect:
        connect.assert_called_once_with(18765, "credential")
    else:
        connect.assert_not_called()


def test_launch_keeps_gui_stderr_in_file_without_blocking(
    launch_bridge: tuple[McpBridge, _ChildStderr, Mock, Mock], tmp_path: Path
) -> None:
    bridge, child, _, _ = launch_bridge
    stderr_file = bridge.config.stderr_file
    stderr_file.write_bytes(b"previous launch")
    child.output = b"warning\n" * 32768  # 256 KiB, well past a pipe buffer
    result = bridge.launch(tmp_path, 18765, auto_connect=False)
    assert str(stderr_file) in result
    assert stderr_file.read_bytes() == child.output
    assert bridge.launched_gui


@pytest.mark.parametrize("stderr", [b"", b"noise\n" * 10 + b"startup details\n"])
def test_launch_reports_early_exit_without_connecting(
    launch_bridge: tuple[McpBridge, _ChildStderr, Mock, Mock],
    tmp_path: Path,
    stderr: bytes,
) -> None:
    bridge, child, _, connect = launch_bridge
    child.proc.poll.return_value = 7
    child.output = stderr
    with pytest.raises(RuntimeError, match="returncode=7") as error:
        bridge.launch(tmp_path, 18765)
    message = str(error.value)
    assert str(bridge.config.stderr_file) in message
    if stderr:
        assert message.endswith("noise\nnoise\nnoise\nnoise\nstartup details")
    assert not bridge.launched_gui
    connect.assert_not_called()


def test_launched_child_can_fill_stderr_after_parent_returns(tmp_path: Path) -> None:
    """A real child must not stall when startup warnings exceed pipe capacity."""
    config = _config(tmp_path)
    script = tmp_path / "scripts" / config.run_script_name
    script.parent.mkdir()
    script.write_text(
        "import socket, sys, time\n"
        "from pathlib import Path\n"
        "port = int(sys.argv[sys.argv.index('--control-port') + 1])\n"
        "with socket.socket() as server:\n"
        "    server.bind(('127.0.0.1', port))\n"
        "    server.listen()\n"
        "    deadline = time.monotonic() + 10\n"
        "    while not Path('release').exists():\n"
        "        if time.monotonic() > deadline: raise TimeoutError('release')\n"
        "        time.sleep(0.01)\n"
        "    sys.stderr.write('warning\\n' * 262144)\n"
        "    sys.stderr.flush()\n"
        "    Path('completed').touch()\n",
        encoding="utf8",
    )
    bridge = McpBridge(config)
    try:
        bridge.launch(tmp_path, _find_free_port(), auto_connect=False)
        pid = int(config.pid_file.read_text())
        (tmp_path / "release").touch()
        assert bridge.wait_for_gui_exit(pid, timeout=5)
        assert (tmp_path / "completed").exists()
    finally:
        if bridge.launched_gui:
            bridge.stop()


def test_launch_timeout_retains_process_without_claiming_connection(
    launch_bridge: tuple[McpBridge, _ChildStderr, Mock, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _, _, connect = launch_bridge
    monkeypatch.setattr(
        "zcu_tools.mcp.core.bridge.time.monotonic", Mock(side_effect=[0, 16])
    )
    result = bridge.launch(tmp_path, 18765)
    assert "not yet reachable" in result
    assert bridge.launched_gui
    connect.assert_not_called()


def test_launch_rejects_occupied_port_without_owning_process(
    launch_bridge: tuple[McpBridge, _ChildStderr, Mock, Mock], tmp_path: Path
) -> None:
    bridge, _, probe, connect = launch_bridge
    probe.side_effect = None
    probe.return_value = True
    with pytest.raises(RuntimeError, match="already in use"):
        bridge.launch(tmp_path, 18765)
    assert not bridge.launched_gui
    connect.assert_not_called()


def test_launch_tolerates_unwritable_pid_file(
    launch_bridge: tuple[McpBridge, _ChildStderr, Mock, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _, _, _ = launch_bridge
    monkeypatch.setattr(
        Path, "write_text", Mock(side_effect=PermissionError("read only"))
    )
    result = bridge.launch(tmp_path, 18765, auto_connect=False)
    assert "listening on port 18765" in result
    assert bridge.launched_gui


# ---------------------------------------------------------------------------
# Public socket connection outcomes
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to port 0 to get a free port number, then release it.

    There is a brief TOCTOU window between release and the test assertion, but
    for a loopback-only test this is acceptable (no service binds the port in CI).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.requires_loopback
def test_socket_transport_rejects_closed_port() -> None:
    transport = SocketTransport("connection-test")
    try:
        with pytest.raises(ConnectionRefusedError):
            transport.open(_find_free_port())
        assert not transport.is_open
    finally:
        transport.close()


@pytest.mark.requires_loopback
def test_socket_transport_connects_to_listening_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        transport = SocketTransport("connection-test")
        try:
            transport.open(srv.getsockname()[1])
            assert transport.is_open
        finally:
            transport.close()


def test_send_rpc_raw_timeout_closes_transport(tmp_path: Path) -> None:
    transport = _SilentTransport()
    bridge = McpBridge(_config(tmp_path), transport=transport)

    with pytest.raises(GuiTransportTimeoutError) as exc_info:
        bridge.send_rpc_raw("slow.method", {}, 0.001)

    assert exc_info.value.method == "slow.method"
    assert transport.closed is True
    assert bridge.is_connected is False
    assert transport.sent[0]["method"] == "slow.method"


def test_disconnect_releases_all_pending_rpc_without_close_callback(
    tmp_path: Path,
) -> None:
    transport = _ObservedTransport()
    bridge = McpBridge(_config(tmp_path), transport=transport)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [
            pool.submit(bridge.send_rpc_raw, f"slow.{i}", {}, 5) for i in range(2)
        ]
        requests = [transport.requests.get(timeout=1) for _ in pending]
        try:
            assert bridge.disconnect() == "Disconnected from GUI."
            for result in pending:
                with pytest.raises(RuntimeError, match="[Dd]isconnect"):
                    result.result(timeout=1)
            assert transport.closed
            assert not bridge.is_connected
            assert bridge.disconnect() == "Not connected."
            with pytest.raises(RuntimeError, match="not connected"):
                bridge.send_rpc_raw("after.disconnect", {}, 5)
        finally:
            for request in requests:
                transport.reply(request)
            bridge.disconnect()


def test_disconnect_does_not_overwrite_a_delivered_reply(tmp_path: Path) -> None:
    transport = _ObservedTransport()
    bridge = McpBridge(_config(tmp_path), transport=transport)
    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = pool.submit(bridge.send_rpc_raw, "finished", {}, 5)
        finished_request = transport.requests.get(timeout=1)
        waiting = pool.submit(bridge.send_rpc_raw, "waiting", {}, 5)
        waiting_request = transport.requests.get(timeout=1)
        try:
            transport.reply(finished_request)
            bridge.disconnect()
            assert completed.result(timeout=1)["result"] == "finished"
            with pytest.raises(RuntimeError, match="[Dd]isconnect"):
                waiting.result(timeout=1)
        finally:
            transport.reply(waiting_request)
            bridge.disconnect()


@pytest.mark.parametrize("cause", ["eof", "timeout", "send_failure"])
def test_transport_failure_releases_other_pending_rpc(
    tmp_path: Path, cause: str
) -> None:
    transport = _ObservedTransport()
    bridge = McpBridge(_config(tmp_path), transport=transport)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(bridge.send_rpc_raw, "waiting", {}, 5)
        waiting_request = transport.requests.get(timeout=1)
        try:
            if cause == "eof":
                transport.unexpected_close()
            elif cause == "timeout":
                with pytest.raises(GuiTransportTimeoutError):
                    bridge.send_rpc_raw("timed.out", {}, 0)
            else:
                transport.release_send.set()
                with pytest.raises(ConnectionError, match="send failed"):
                    bridge.send_rpc_raw("failing.send", {}, 5)
            with pytest.raises((RuntimeError, ConnectionError)):
                waiting.result(timeout=1)
            assert not bridge.is_connected
            assert transport.closed
        finally:
            transport.reply(waiting_request)
            bridge.disconnect()


def test_transport_replacement_settles_old_rpc_and_ignores_retired_callbacks(
    tmp_path: Path,
) -> None:
    old = _ObservedTransport()
    bridge = McpBridge(_config(tmp_path), transport=old)
    new = _ObservedTransport()
    with ThreadPoolExecutor(max_workers=2) as pool:
        old_rpc = pool.submit(bridge.send_rpc_raw, "old", {}, 5)
        old_request = old.requests.get(timeout=1)
        new_request = None
        try:
            bridge.set_transport(new)
            with pytest.raises(RuntimeError, match="[Dd]isconnect"):
                old_rpc.result(timeout=1)
            new_rpc = pool.submit(bridge.send_rpc_raw, "new", {}, 5)
            new_request = new.requests.get(timeout=1)
            old.unexpected_close()
            old.reply(old_request)
            assert not new_rpc.done()
            new.reply(new_request)
            assert new_rpc.result(timeout=1)["result"] == "new"
            assert bridge.is_connected
        finally:
            old.reply(old_request)
            old.close()
            if new_request is not None:
                new.reply(new_request)
            bridge.disconnect()


def test_retired_send_failure_does_not_disconnect_replacement(tmp_path: Path) -> None:
    old = _ObservedTransport()
    bridge = McpBridge(_config(tmp_path), transport=old)
    new = _ObservedTransport()
    with ThreadPoolExecutor(max_workers=2) as pool:
        old_rpc = pool.submit(bridge.send_rpc_raw, "failing.send", {}, 5)
        old.requests.get(timeout=1)
        new_request = None
        try:
            bridge.set_transport(new)
            new_rpc = pool.submit(bridge.send_rpc_raw, "new", {}, 5)
            new_request = new.requests.get(timeout=1)
            old.release_send.set()
            with pytest.raises(ConnectionError, match="send failed"):
                old_rpc.result(timeout=1)
            old.unexpected_close()
            assert not new_rpc.done()
            assert bridge.is_connected
            new.reply(new_request)
            assert new_rpc.result(timeout=1)["result"] == "new"
        finally:
            old.release_send.set()
            old.close()
            if new_request is not None:
                new.reply(new_request)
            bridge.disconnect()


@pytest.mark.parametrize(
    ("wire_version", "gui_version", "mismatch"),
    [
        (1, 7, False),
        (100, 1, True),
        (1, 999, False),
    ],
)
def test_version_note_compares_wire_but_only_reports_gui_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wire_version: int,
    gui_version: int,
    mismatch: bool,
) -> None:
    bridge = McpBridge(_config(tmp_path))

    def send_rpc_raw(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "result": {"wire_version": wire_version, "gui_version": gui_version},
        }

    monkeypatch.setattr(bridge, "send_rpc_raw", send_rpc_raw)
    note = bridge.wire_version_note()
    if mismatch:
        assert "WIRE VERSION MISMATCH" in note
    else:
        assert "MISMATCH" not in note
        assert "wire v1" in note
        assert f"gui code v{gui_version}" in note
        assert "mcp code v1" in note


def _attach_peer(bridge: McpBridge, listener: socket.socket) -> socket.socket:
    transport = SocketTransport("frame-test")
    bridge.set_transport(transport)
    transport.open(listener.getsockname()[1])
    peer, _ = listener.accept()
    peer.settimeout(3)
    return peer


@pytest.fixture
def wire_bridge(
    tmp_path: Path,
) -> Iterator[tuple[McpBridge, socket.socket, socket.socket]]:
    bridge = McpBridge(_config(tmp_path))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        try:
            with _attach_peer(bridge, listener) as peer:
                yield bridge, listener, peer
        finally:
            bridge.disconnect()


def _read_request(peer: socket.socket) -> dict[str, Any]:
    with peer.makefile("rb") as reader:
        return json.loads(reader.readline(MAX_LINE_BYTES + 2))


def _assert_small_rpc(bridge: McpBridge, peer: socket.socket) -> None:
    with ThreadPoolExecutor(max_workers=1) as worker:
        reply = worker.submit(bridge.send_rpc_raw, "small", {}, 3)
        request = _read_request(peer)
        assert request["method"] == "small"
        peer.sendall(encode_line({"id": request["id"], "ok": True, "result": 42}))
        assert reply.result(timeout=3)["result"] == 42


@pytest.mark.requires_loopback
def test_oversized_request_is_not_sent_and_connection_remains_usable(
    wire_bridge: tuple[McpBridge, socket.socket, socket.socket],
) -> None:
    bridge, _, peer = wire_bridge
    # UTF-8 bytes, not character count. This exceeds the budget only after encoding.
    with pytest.raises(GuiMessageTooLargeError, match="request.*not sent") as caught:
        bridge.send_rpc_raw("mutate", {"value": "é" * (MAX_LINE_BYTES // 2)}, 3)
    assert caught.value.reason == "message_too_large"
    assert bridge.is_connected
    # The first actual wire request is small; mutate was not partially transmitted.
    _assert_small_rpc(bridge, peer)


@pytest.mark.requires_loopback
def test_exact_limit_response_and_following_frames_remain_usable(
    wire_bridge: tuple[McpBridge, socket.socket, socket.socket],
) -> None:
    bridge, _, peer = wire_bridge
    with ThreadPoolExecutor(max_workers=1) as worker:
        reply = worker.submit(bridge.send_rpc_raw, "large", {}, 5)
        request = _read_request(peer)
        response = {"id": request["id"], "ok": True, "result": ""}
        overhead = len(encode_line(response)) - 1
        response["result"] = "x" * (MAX_LINE_BYTES - overhead)
        peer.sendall(
            encode_line(response) + encode_line({"event": "notice", "payload": {}})
        )
        assert reply.result(timeout=5) == response
    assert bridge.is_connected
    _assert_small_rpc(bridge, peer)


@pytest.mark.requires_loopback
@pytest.mark.parametrize("terminated", [False, True])
def test_oversized_response_fails_explicitly_and_reconnect_does_not_replay(
    wire_bridge: tuple[McpBridge, socket.socket, socket.socket], terminated: bool
) -> None:
    bridge, listener, peer = wire_bridge
    with ThreadPoolExecutor(max_workers=1) as worker:
        reply = worker.submit(bridge.send_rpc_raw, "mutate", {}, 5)
        request = _read_request(peer)
        assert request["method"] == "mutate"
        # Exercise a coalesced preceding frame as well as a fragmented oversized frame.
        prefix = encode_line({"event": "notice", "payload": {}})
        data = prefix + json.dumps(
            {"id": request["id"], "ok": True, "result": "x" * MAX_LINE_BYTES}
        ).encode("utf-8")
        if terminated:
            data += b"\n"
        with suppress(BrokenPipeError, ConnectionResetError):
            peer.sendall(data)
        with pytest.raises(
            GuiMessageTooLargeError, match="response.*may have executed"
        ):
            reply.result(timeout=5)
    assert not bridge.is_connected
    bridge.disconnect()
    with _attach_peer(bridge, listener) as recovered:
        _assert_small_rpc(bridge, recovered)
