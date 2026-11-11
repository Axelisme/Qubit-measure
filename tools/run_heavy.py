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
MINIMUM_AVAILABLE_BYTES = 8_000_000_000
QUEUE_POLL_SECONDS = 0.25
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
    wait_limit: float | None


def _parse(argv: list[str] | None) -> _Invocation:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--wait-limit", type=float)
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
    if args.wait_limit is not None and (
        not math.isfinite(args.wait_limit) or args.wait_limit < 0
    ):
        parser.error("wait limit must be a non-negative finite number of seconds")
    return _Invocation(
        log=args.log.resolve(),
        timeout=args.timeout,
        command=command,
        wait_limit=args.wait_limit,
    )


def _spawn(
    invocation: _Invocation, cpus: list[int], lock_fd: int
) -> subprocess.Popen[bytes]:
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
                pass_fds=(lock_fd,),
            )
    finally:
        os.sched_setaffinity(0, affinity)


@contextmanager
def _forwarding_child(
    invocation: _Invocation, cpus: list[int], lock_fd: int
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
        child = _spawn(invocation, cpus, lock_fd)
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


def _memory_available(meminfo_path: Path) -> bool:
    """Read MemAvailable in procfs kB; missing or malformed data is an error."""
    for line in meminfo_path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "MemAvailable:":
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdecimal():
                raise ValueError("invalid MemAvailable in meminfo")
            return int(fields[1]) * 1024 >= MINIMUM_AVAILABLE_BYTES
    raise ValueError("MemAvailable is missing from meminfo")


class _WaitLimitExpired(Exception):
    """Admission ended without starting a command."""


@contextmanager
def _acquired_slot(
    invocation: _Invocation, meminfo_path: Path, lock_directory: Path
) -> Generator[tuple[int, int, float]]:
    """Yield (slot index, inherited lock fd, waited seconds) after admission.

    Close our descriptor on exit, without unlocking copies inherited by children.
    Raise _WaitLimitExpired if admission exceeds the optional wait limit.
    """
    import fcntl

    lock_directory.mkdir(parents=True, exist_ok=True)
    queued = time.monotonic()
    reported = ""
    while True:
        if (
            reported
            and invocation.wait_limit is not None
            and time.monotonic() - queued >= invocation.wait_limit
        ):
            raise _WaitLimitExpired(f"wait limit expired: {reported}")
        reason = "MemAvailable < 8 GB"
        if _memory_available(meminfo_path):
            for slot in range(SLOT_COUNT):
                with (lock_directory / f"slot-{slot}.lock").open("a+b") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    # Close only: LOCK_UN would also unlock inherited descriptions.
                    yield slot, lock.fileno(), time.monotonic() - queued
                    return
            reason = "all CPU slots are occupied"
        if reason != reported:
            print(f"run_heavy: waiting: {reason}", file=sys.stderr)
            reported = reason
        if invocation.wait_limit is not None:
            remaining = invocation.wait_limit - (time.monotonic() - queued)
            if remaining <= 0:
                raise _WaitLimitExpired(f"wait limit expired: {reason}")
            time.sleep(min(QUEUE_POLL_SECONDS, remaining))
        else:
            time.sleep(QUEUE_POLL_SECONDS)


def _execute(
    invocation: _Invocation, meminfo_path: Path, lock_directory: Path | None
) -> int:
    cwd = Path.cwd()
    count = os.cpu_count()
    if count is None or count < SLOT_COUNT:
        raise ValueError(f"at least {SLOT_COUNT} machine CPUs are required")
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
    if lock_directory is None:
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip()
        lock_directory = Path(common) / "run-heavy"
    with _acquired_slot(invocation, meminfo_path, lock_directory) as (
        slot,
        lock_fd,
        wait_seconds,
    ):
        width = count // SLOT_COUNT
        cpus = list(range(slot * width, (slot + 1) * width))
        started_utc = datetime.now(UTC).isoformat()
        started = time.monotonic()
        with _forwarding_child(invocation, cpus, lock_fd) as child:
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
                    "slot": slot,
                    "cwd": str(cwd),
                    **metadata,
                    "wait_seconds": wait_seconds,
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


def main(
    argv: list[str] | None = None,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    lock_directory: Path | None = None,
) -> int:
    """Run `--log PATH [--wait-limit SECONDS] TIMEOUT -- COMMAND...` in cwd.

    Call on the main thread, on Linux only, within a committed Git worktree.
    Wait for MemAvailable >= 8 GB and one of two repository-wide CPU slots.
    Each slot uses cpu_count // 2 consecutive machine CPUs, ignoring an odd last
    CPU. At least two CPUs and affinity access to the acquired slot are required.
    The child receives fixed headless/single-thread settings and its combined
    output replaces PATH. It inherits the flock so runner death cannot free a
    slot while its command or inheriting descendants still run.
    meminfo_path reads procfs-style memory data (default /proc/meminfo).
    lock_directory overrides the default <absolute git common dir>/run-heavy.
    Both keyword-only paths permit isolated tests; neither is a CLI option.
    A non-negative finite wait limit bounds admission; expiry returns 75 with
    a stderr reason and no receipt. Missing MemAvailable fails with status 2.
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
        return _execute(invocation, meminfo_path, lock_directory)
    except _WaitLimitExpired as error:
        print(f"run_heavy: {error}", file=sys.stderr)
        return 75
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"run_heavy: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
