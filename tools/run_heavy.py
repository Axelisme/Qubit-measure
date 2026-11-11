"""Run a bounded Linux command and emit a JSON receipt."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

SLOT_COUNT = 2
TERMINATION_GRACE_SECONDS = 10.0
CHILD_ENV = {
    "QT_QPA_PLATFORM": "offscreen",
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class _Invocation:
    log: Path
    timeout: float
    command: list[str]


def _parse(argv: list[str] | None) -> _Invocation:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("timeout", type=float)
    raw = sys.argv[1:] if argv is None else argv
    if "--" not in raw:
        parser.parse_args(raw)
        parser.error("command must follow --")
    separator = raw.index("--")
    args = parser.parse_args(raw[:separator])
    command = raw[separator + 1 :]
    if not command:
        parser.error("a command is required after --")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be a positive finite number of seconds")
    return _Invocation(log=args.log.resolve(), timeout=args.timeout, command=command)


def _spawn(invocation: _Invocation, cpus: list[int]) -> subprocess.Popen[bytes]:
    """Start in a new process group, restoring the caller's thread affinity."""
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        if os.sched_getaffinity(0) != set(cpus):
            raise OSError("the cpuset clipped the requested slot CPU affinity")
        with invocation.log.open("wb") as log:
            return subprocess.Popen(
                invocation.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **CHILD_ENV},
                start_new_session=True,
            )
    finally:
        os.sched_setaffinity(0, affinity)


@contextmanager
def _forwarding_child(
    invocation: _Invocation, cpus: list[int]
) -> Generator[subprocess.Popen[bytes]]:
    """Forward INT/TERM for the child's lifetime, including signals during spawn."""
    child: subprocess.Popen[bytes] | None = None
    pending: list[int] = []

    def forward(signum: int, _frame: FrameType | None) -> None:
        if child is None:
            pending.append(signum)
        else:
            _signal_group(child.pid, signum)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, forward)
        child = _spawn(invocation, cpus)
        for sig in pending:
            _signal_group(child.pid, sig)
        yield child
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _signal_group(pgid: int, signum: int) -> None:
    """Signal the owned group; a group that has already exited needs no signal."""
    with suppress(ProcessLookupError):
        os.killpg(pgid, signum)


def _group_running(pgid: int) -> bool:
    """Return whether /proc contains a non-zombie member of this process group."""
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
            return True
    return False


def _stop_group(child: subprocess.Popen[bytes]) -> None:
    """TERM the whole group, KILL survivors after 10 seconds, and reap the leader."""
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    _signal_group(child.pid, signal.SIGTERM)
    while _group_running(child.pid):
        child.poll()
        if time.monotonic() >= deadline:
            _signal_group(child.pid, signal.SIGKILL)
            break
        time.sleep(0.05)
    child.wait()
    # Orphan descendants can become zombies under an unreaping PID 1.
    while _group_running(child.pid):
        time.sleep(0.01)


def _execute(invocation: _Invocation) -> int:
    cwd = Path.cwd()
    count = os.cpu_count()
    if count is None or count < SLOT_COUNT:
        raise ValueError(f"at least {SLOT_COUNT} machine CPUs are required")
    cpus = list(range(count // SLOT_COUNT))
    metadata = {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"], text=True
            ).strip()
        ),
    }
    started_utc = datetime.now(UTC).isoformat()
    started = time.monotonic()
    with _forwarding_child(invocation, cpus) as child:
        timed_out = False
        try:
            code = child.wait(timeout=invocation.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_group(child)
            code = 124
        residual_killed = _group_running(child.pid)
        if residual_killed:
            _stop_group(child)
    if code < 0:
        code = 128 - code
    print(
        json.dumps(
            {
                "command": invocation.command,
                "cpus": cpus,
                "env": CHILD_ENV,
                "slot": 0,
                "cwd": str(cwd),
                **metadata,
                "wait_seconds": 0.0,
                "started_utc": started_utc,
                "finished_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "exit_code": code,
                "timed_out": timed_out,
                "residual_killed": residual_killed,
                "log": str(invocation.log),
            }
        )
    )
    return code


def main(argv: list[str] | None = None) -> int:
    """Run `--log PATH TIMEOUT -- COMMAND...` from the caller's Git worktree.

    Call on the main thread, on Linux only. Slot 0 uses the first half of the
    machine CPUs; at least two CPUs are required. The child receives fixed
    headless/single-thread settings and its combined output replaces PATH.
    Return its shell status (128 + signal for signal death), or 124 on timeout.
    Timeout and residual processes get TERM, then KILL after a 10-second grace.
    INT/TERM received by the runner are forwarded to the entire child group.
    Once the child starts, print one JSON receipt with Git identity and outcome.
    Prelaunch errors print their reason to stderr, return 2 and print no receipt.
    """
    if sys.platform != "linux":
        print("run_heavy: only Linux is supported", file=sys.stderr)
        return 2
    try:
        invocation = _parse(argv)
    except SystemExit as error:
        return int(error.code or 0)
    try:
        return _execute(invocation)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"run_heavy: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
