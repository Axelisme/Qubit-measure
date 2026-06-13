"""Headless integration smoke test for the agent supervisor's file-based IPC.

Unlike ``services/test_agent_supervisor.py`` (pure-logic, fully mocked), this
module spawns a *real* child subprocess (a fake stand-in for ``claude``, run via
``sys.executable``) and drives the *real* ``run_supervisor_loop`` in a background
thread. The point is to exercise the parts that no mock can reproduce:

  - a real ``subprocess.Popen`` with real OS pipes,
  - the real ``_io_loop`` (a blocking-``read1`` reader thread + a separate
    spool-poll main loop),
  - real pipe back-pressure / timing.

Commit ``11790d6b`` fixed a starvation bug where a blocking stdout ``read1`` on
the main loop starved spool delivery; that class of bug is only reachable with a
real subprocess producing real stdout while a spool command is pending. Test (c)
below is the regression guard for it.

Linux-only: relies on POSIX subprocess timing and ``start_new_session``-free
in-thread spawning. The fake-child stream-json wire format mirrors the
empirically-verified claude v2.1.177 schema documented in
``task_plans/agent_assistant/findings.md`` (output = one complete JSON object per
line, NOT deltas; input envelope = ``{"type":"user","message":{"role":"user",
"content":"<text>"}}``).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from zcu_tools.gui.app.main.services import agent_supervisor as sup
from zcu_tools.gui.app.main.services.agent_supervisor import (
    run_supervisor_loop,
    write_spool_message,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="smoke test relies on POSIX subprocess timing/signal semantics",
)

# ---------------------------------------------------------------------------
# Fake child script — a minimal stream-json stand-in for ``claude``
# ---------------------------------------------------------------------------

# Written to ``tmp_path`` and executed via ``sys.executable``. Behaviour:
#   - emits one ``system/init`` frame at startup,
#   - reads stdin line-by-line; for each ``{"type":"user",...}`` envelope it
#     echoes back an ``assistant`` text frame ``echo:<content>``,
#   - if ``ZCU_FAKE_FLOOD=1`` it continuously emits filler ``assistant`` frames
#     from a background thread (the starvation stressor for test (c)),
#   - exits cleanly on the sentinel input ``__STOP__`` or on stdin EOF.
#
# CRITICAL: every stdout write is explicitly flushed. Without flush the parent's
# tail never sees the bytes and the whole test would pass vacuously.
_FAKE_CHILD_SOURCE = r"""
import json
import os
import sys
import threading
import time


def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _flood():
    # Continuously emit filler frames to stress the parent's reader thread.
    # This is what makes a blocking main-loop stdout read starve the spool poll.
    while True:
        _emit(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "filler"}],
                },
            }
        )
        time.sleep(0.001)


def main():
    # Startup init frame (schema per findings.md).
    _emit(
        {
            "type": "system",
            "subtype": "init",
            "session_id": "fake-session-0001",
            "apiKeySource": "none",
            "model": "fake-model",
            "permissionMode": "default",
            "mcp_servers": [],
        }
    )

    if os.environ.get("ZCU_FAKE_FLOOD") == "1":
        threading.Thread(target=_flood, daemon=True).start()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            env = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if env.get("type") != "user":
            continue
        content = env.get("message", {}).get("content", "")
        if content == "__STOP__":
            break
        _emit(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "echo:" + str(content)}],
                },
            }
        )


if __name__ == "__main__":
    main()
"""


@pytest.fixture
def fake_child(tmp_path: Path) -> Path:
    """Write the fake-child script and return its path."""
    path = tmp_path / "fake_claude.py"
    path.write_text(_FAKE_CHILD_SOURCE, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Polling helpers (generous timeout; robust under ``pytest -n auto``)
# ---------------------------------------------------------------------------

# Generous because a real subprocess + thread scheduling under parallel test
# load can be slow; the supervisor's own spool poll runs every 0.1s.
_DEADLINE_S = 10.0
_POLL_S = 0.02


def _read_log_lines(log_path: Path) -> list[dict]:
    """Read ``log.ndjson`` and return parsed JSON objects (best-effort).

    Re-reads the whole file each call; a partially-written trailing line is
    tolerated by skipping any line that does not parse.
    """
    if not log_path.exists():
        return []
    objs: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return objs


def _wait_until(predicate: Callable[[], bool], *, what: str) -> None:
    """Poll ``predicate`` until true or the deadline elapses; else fail."""
    deadline = time.monotonic() + _DEADLINE_S
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(_POLL_S)
    pytest.fail(f"timed out after {_DEADLINE_S}s waiting for: {what}")


def _assistant_texts(log_path: Path) -> list[str]:
    """Extract all assistant text blocks currently in the log."""
    texts: list[str] = []
    for obj in _read_log_lines(log_path):
        if obj.get("type") != "assistant":
            continue
        for block in obj.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
    return texts


def _has_init(log_path: Path) -> bool:
    return any(
        obj.get("type") == "system" and obj.get("subtype") == "init"
        for obj in _read_log_lines(log_path)
    )


# ---------------------------------------------------------------------------
# Supervisor harness — runs the real run_supervisor_loop in a background thread
# ---------------------------------------------------------------------------


class _SupervisorRun:
    """Run ``run_supervisor_loop`` in a thread with ``build_claude_argv`` patched.

    ``build_claude_argv`` is patched (in the *supervisor* module's namespace,
    where it is imported and looked up) to launch the fake child via
    ``sys.executable`` instead of the real ``claude`` binary. ``session_id`` is
    intentionally left ``None`` so the loop never touches the real
    ``~/.cache/.../agent_sessions`` registry.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        fake_child: Path,
        flood: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.session_dir = session_dir
        self.log_path = session_dir / "log.ndjson"
        self.spool_dir = session_dir / "spool"
        self._fake_child = fake_child
        self._flood = flood
        self._monkeypatch = monkeypatch
        self._thread: threading.Thread | None = None
        self.exit_code: int | None = None

    def __enter__(self) -> _SupervisorRun:
        env_flag = "1" if self._flood else "0"

        def _fake_argv(task: str, mcp_config_path: str, *args: object) -> list[str]:
            # ``-u`` forces unbuffered stdio so the parent's tail sees output
            # promptly; the fake child also flushes explicitly as a belt-and-braces.
            return [
                sys.executable,
                "-u",
                str(self._fake_child),
            ]

        self._monkeypatch.setattr(sup, "build_claude_argv", _fake_argv)
        # The flood flag is read by the child from the environment; set it on the
        # supervisor process so the inherited env carries it to the child.
        self._monkeypatch.setenv("ZCU_FAKE_FLOOD", env_flag)

        def _target() -> None:
            self.exit_code = run_supervisor_loop(
                session_dir=self.session_dir,
                task="initial task",
                repo_root=str(self.session_dir),  # any valid dir; only used for mcp cfg
                session_id=None,  # no registry side effects
                _spawn_claude=True,  # real Popen + real _io_loop
            )

        self._thread = threading.Thread(target=_target, name="supervisor-under-test")
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        # Best-effort clean shutdown: send the sentinel so the child breaks its
        # loop, then join the supervisor thread.
        try:
            if self.spool_dir.exists():
                write_spool_message(self.spool_dir, "__STOP__")
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=_DEADLINE_S)

    def join(self, timeout: float) -> bool:
        assert self._thread is not None
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()


# ---------------------------------------------------------------------------
# (a) init frame reaches the log
# ---------------------------------------------------------------------------


def test_smoke_init_frame_appears_in_log(
    tmp_path: Path, fake_child: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    with _SupervisorRun(
        session_dir=session_dir,
        fake_child=fake_child,
        flood=False,
        monkeypatch=monkeypatch,
    ) as run:
        _wait_until(
            lambda: _has_init(run.log_path),
            what="system/init frame in log.ndjson",
        )


# ---------------------------------------------------------------------------
# (b) spool → stdin → assistant echo round-trip
# ---------------------------------------------------------------------------


def test_smoke_spool_roundtrip(
    tmp_path: Path, fake_child: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    with _SupervisorRun(
        session_dir=session_dir,
        fake_child=fake_child,
        flood=False,
        monkeypatch=monkeypatch,
    ) as run:
        # Wait for the child to be up (init seen) before injecting a command.
        _wait_until(lambda: _has_init(run.log_path), what="init before spool")

        spool_file = write_spool_message(run.spool_dir, "ping-roundtrip")

        # The supervisor must consume (delete) the spool file...
        _wait_until(
            lambda: not spool_file.exists(),
            what="spool file consumed (deleted) by supervisor",
        )
        # ...and the child's echo must land in the log.
        _wait_until(
            lambda: "echo:ping-roundtrip" in _assistant_texts(run.log_path),
            what="assistant echo frame for the spooled command",
        )


# ---------------------------------------------------------------------------
# (c) starvation regression guard for commit 11790d6b (the core test)
# ---------------------------------------------------------------------------


def test_smoke_spool_not_starved_by_heavy_stdout(
    tmp_path: Path, fake_child: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for ``11790d6b``: a spool command must still be delivered and
    echoed promptly even while the child floods stdout.

    Before ``11790d6b`` the blocking ``read1`` on the supervisor's *main* loop
    starved the spool poll, so a command written while the child was producing
    stdout never reached the child's stdin. With the reader-thread / main-loop
    split, the heavy stdout is drained on a separate thread and the spool poll
    keeps running, so the echo arrives.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    with _SupervisorRun(
        session_dir=session_dir,
        fake_child=fake_child,
        flood=True,  # child continuously floods stdout
        monkeypatch=monkeypatch,
    ) as run:
        _wait_until(lambda: _has_init(run.log_path), what="init before spool (flood)")

        # Inject a command in the middle of the stdout flood.
        spool_file = write_spool_message(run.spool_dir, "ping-under-load")

        _wait_until(
            lambda: not spool_file.exists(),
            what="spool consumed despite heavy stdout (11790d6b guard)",
        )
        _wait_until(
            lambda: "echo:ping-under-load" in _assistant_texts(run.log_path),
            what="echo delivered despite heavy stdout (11790d6b guard)",
        )


# ---------------------------------------------------------------------------
# (d) clean shutdown — sentinel ends the child, supervisor thread joins
# ---------------------------------------------------------------------------


def test_smoke_clean_shutdown(
    tmp_path: Path, fake_child: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    with _SupervisorRun(
        session_dir=session_dir,
        fake_child=fake_child,
        flood=False,
        monkeypatch=monkeypatch,
    ) as run:
        _wait_until(lambda: _has_init(run.log_path), what="init before shutdown")

        # Sentinel makes the child break its read loop and exit (EOF on its
        # stdout → reader thread sees EOF → _io_loop exits → thread returns).
        write_spool_message(run.spool_dir, "__STOP__")

        assert run.join(timeout=_DEADLINE_S), (
            "supervisor thread did not join after the child exited on sentinel"
        )
        # The loop returns the child's exit code (0 for a clean fake-child exit).
        assert run.exit_code == 0


# ---------------------------------------------------------------------------
# (stretch, intentionally omitted) IndependentAgentSession liveness end-to-end
# ---------------------------------------------------------------------------
#
# The ``7fd2f6b9`` fix has two halves: (1) is_running() stays True across turns
# while the supervisor pid is alive, and (2) it flips to stopped once the pid
# dies. Driving this end-to-end against THIS smoke harness cannot validate half
# (2) honestly: ``run_supervisor_loop`` runs in a thread of *this* test process,
# so the "supervisor pid" the session would probe is the test process itself —
# always alive, so the pid-death transition is unreachable here. Wiring a real
# *detached* supervisor (so there is a killable pid) routes argv resolution into
# a child process where ``build_claude_argv`` is re-imported, putting it past the
# monkeypatch's process boundary. Both halves are already covered by the mocked
# unit tests in ``services/test_independent_agent_session.py`` (I13 pid-death →
# stopped; the "attached idle session still running" regression). Adding a
# misleading half-test here would be worse than omitting it.
