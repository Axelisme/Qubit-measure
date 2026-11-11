"""Git-runner helpers and OS boundary doubles shared by run-heavy contracts."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest
from run_heavy import main


def run(repo: Path, argv: list[str]) -> int:
    """Run main against the disposable repository's ample-memory fixture."""
    return main(argv, meminfo_path=repo / ".git" / "meminfo")


@contextmanager
def observed_group(
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    ready: Path,
    on_ready: Callable[[float | None], None] | None = None,
) -> Generator[dict[str, int]]:
    """Observe a child's FIFO handshake and always stop/reap its owned group."""
    os.mkfifo(ready)
    ready_fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    original_wait = subprocess.Popen.wait
    original_killpg = os.killpg
    children: list[subprocess.Popen[bytes]] = []
    pids: dict[str, int] = {}

    def wait(child: subprocess.Popen[bytes], timeout: float | None = None) -> int:
        if child.args == command and not children:
            children.append(child)
            # The child publishes its fork; this watchdog only detects a stuck handshake.
            assert select.select([ready_fd], [], [], 30)[0], (
                "child never reached readiness"
            )
            pids.update(json.loads(os.read(ready_fd, 4096)))
            if on_ready is not None:
                on_ready(timeout)
        return original_wait(child, timeout=timeout)

    with monkeypatch.context() as boundary:
        boundary.setattr(subprocess.Popen, "wait", wait)
        try:
            yield pids
        finally:
            os.close(ready_fd)
            for child in children:
                with suppress(ProcessLookupError):
                    original_killpg(child.pid, signal.SIGKILL)
                original_wait(child, timeout=30)


def group_command(ready: Path, leader_action: str, preamble: str = "") -> list[str]:
    """Build a two-process group whose leader publishes both PIDs to a FIFO."""
    source = (
        "import json, os, signal\n" + preamble + "worker = os.fork()\n"
        "if worker == 0:\n"
        "    signal.pause()\n"
        "else:\n"
        f"    with open({str(ready)!r}, 'w') as stream:\n"
        "        print(json.dumps({'leader': os.getpid(), 'worker': worker}), file=stream, flush=True)\n"
        f"    {leader_action}\n"
    )
    return [sys.executable, "-c", source]


def assert_group_stopped(pids: dict[str, int]) -> None:
    """Check that every observed process is gone or stopped as a zombie."""
    assert set(pids) == {"leader", "worker"}
    for pid in pids.values():
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] in {
            "Z",
            "X",
        }


@contextmanager
def unreapable_leader(
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    is_stuck: Callable[[], bool],
) -> Generator[None]:
    """Model a child that cannot be reaped after KILL, without leaking a real process."""
    original_wait = subprocess.Popen.wait
    original_poll = subprocess.Popen.poll

    def poll(child: subprocess.Popen[bytes]) -> int | None:
        if is_stuck() and child.args == command:
            return None
        return original_poll(child)

    def wait(child: subprocess.Popen[bytes], timeout: float | None = None) -> int:
        if is_stuck() and child.args == command:
            assert timeout is not None, "SIGKILL leader reap must have a deadline"
            raise subprocess.TimeoutExpired(command, timeout)
        return original_wait(child, timeout=timeout)

    with monkeypatch.context() as boundary:
        boundary.setattr(subprocess.Popen, "wait", wait)
        boundary.setattr(subprocess.Popen, "poll", poll)
        yield


@contextmanager
def held_slots(
    lock_directory: Path, slots: list[int]
) -> Generator[Callable[[int], None]]:
    """Hold real flocks in a peer process until explicit release handshakes."""
    lock_directory.mkdir(parents=True, exist_ok=True)
    source = (
        "import fcntl, pathlib, sys\n"
        f"directory = pathlib.Path({str(lock_directory)!r})\n"
        f"locks = {{slot: (directory / f'slot-{{slot}}.lock').open('a+b') for slot in {slots!r}}}\n"
        "for stream in locks.values():\n"
        "    fcntl.flock(stream, fcntl.LOCK_EX)\n"
        "print('ready', flush=True)\n"
        "for line in sys.stdin:\n"
        "    locks.pop(int(line)).close()\n"
        "    print('released', flush=True)\n"
    )
    with subprocess.Popen(
        [sys.executable, "-c", source],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as peer:
        assert peer.stdin is not None and peer.stdout is not None
        writer, reader = peer.stdin, peer.stdout

        def receive(expected: str) -> None:
            # Every read has a peer write; the timeout only catches a broken handshake.
            assert select.select([reader], [], [], 30)[0], "lock peer did not respond"
            assert reader.readline().strip() == expected

        def release(slot: int) -> None:
            writer.write(f"{slot}\n")
            writer.flush()
            receive("released")

        try:
            receive("ready")
            yield release
        finally:
            writer.close()
            peer.wait(timeout=30)
