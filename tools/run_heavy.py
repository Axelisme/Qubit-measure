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
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

SLOT_COUNT = 2
MINIMUM_AVAILABLE_BYTES = 8_000_000_000
QUEUE_POLL_SECONDS = 0.25
TERMINATION_GRACE_SECONDS = 10.0
KILL_GRACE_SECONDS = 10.0
TERMINATION_POLL_SECONDS = 0.05
KILL_POLL_SECONDS = 0.01
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
    exclusive: bool


def _parse(argv: list[str] | None) -> _Invocation:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="replace this file with combined command stdout/stderr",
    )
    parser.add_argument(
        "--wait-limit",
        type=float,
        help="maximum admission wait in seconds (default: unlimited)",
    )
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="reserve both CPU slots before starting the command",
    )
    parser.add_argument(
        "timeout",
        type=float,
        help="positive execution timeout in seconds, excluding queue time",
    )
    raw = sys.argv[1:] if argv is None else argv
    separator = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:separator])
    if separator == len(raw):
        parser.error("command must follow --")
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
        exclusive=args.exclusive,
    )


@contextmanager
def _spawn(
    invocation: _Invocation, cpus: list[int], lock_fds: tuple[int, ...]
) -> Generator[subprocess.Popen[bytes]]:
    """Yield immediately after launch; restore affinity and close the log on exit."""
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        if os.sched_getaffinity(0) != set(cpus):
            raise OSError("the cpuset clipped the requested slot CPU affinity")
        with invocation.log.open("wb") as log:
            child = subprocess.Popen(
                invocation.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **CHILD_ENV},
                start_new_session=True,
                pass_fds=lock_fds,
            )
            yield child
    finally:
        os.sched_setaffinity(0, affinity)


@contextmanager
def _forwarding_child(
    invocation: _Invocation, cpus: list[int], lock_fds: tuple[int, ...]
) -> Generator[tuple[subprocess.Popen[bytes], list[int]]]:
    """Yield the launched child and signals queued during spawn; restore handlers."""
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
        with _spawn(invocation, cpus, lock_fds) as child:
            yield child, pending
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
    """TERM then KILL the group, bounding both phases and polling to reap the leader.

    Stop waiting after KILL_GRACE_SECONDS even if the kernel still reports a
    live group member or cannot reap the leader, so a receipt can be emitted.
    """
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    _signal_group(child.pid, signal.SIGTERM)
    while True:
        leader_stopped = child.poll() is not None
        if not _group_running(child.pid) and leader_stopped:
            return
        if time.monotonic() >= deadline:
            _signal_group(child.pid, signal.SIGKILL)
            break
        time.sleep(TERMINATION_POLL_SECONDS)

    deadline = time.monotonic() + KILL_GRACE_SECONDS
    while True:
        leader_stopped = child.poll() is not None
        if not _group_running(child.pid) and leader_stopped:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                "run_heavy: SIGKILL cleanup wait expired; group may still exist",
                file=sys.stderr,
            )
            return
        # Zombies do not keep a group running, but D-state survivors can outlast KILL.
        time.sleep(min(KILL_POLL_SECONDS, remaining))


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
) -> Generator[tuple[int, tuple[int, ...], float]]:
    """Yield (first slot, inherited lock fds, waited seconds) after admission.

    Close our descriptor on exit, without unlocking copies inherited by children.
    Raise _WaitLimitExpired if admission exceeds the optional wait limit.
    """
    # Linux-only: defer import so other platforms can report the CLI platform error.
    import fcntl

    lock_directory.mkdir(parents=True, exist_ok=True)
    queued = time.monotonic()
    deadline = None if invocation.wait_limit is None else queued + invocation.wait_limit
    attempted = False
    last_reported: str | None = None
    while True:
        if attempted and deadline is not None and time.monotonic() >= deadline:
            raise _WaitLimitExpired(f"wait limit expired: {last_reported}")
        attempted = True
        reason = f"MemAvailable < {MINIMUM_AVAILABLE_BYTES / 1_000_000_000:g} GB"
        if _memory_available(meminfo_path):
            candidates = (
                [list(range(SLOT_COUNT))]
                if invocation.exclusive
                else [[slot] for slot in range(SLOT_COUNT)]
            )
            for slots in candidates:
                with ExitStack() as reservation:
                    locks = []
                    for slot in slots:
                        lock = reservation.enter_context(
                            (lock_directory / f"slot-{slot}.lock").open("a+b")
                        )
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            break
                        locks.append(lock.fileno())
                    else:
                        # Close only: LOCK_UN would also unlock inherited descriptions.
                        yield slots[0], tuple(locks), time.monotonic() - queued
                        return
                # Failed attempts close every partial lock before polling again.
            reason = (
                "exclusive access requires all CPU slots to be free"
                if invocation.exclusive
                else "all CPU slots are occupied"
            )
        if reason != last_reported:
            print(f"run_heavy: waiting: {reason}", file=sys.stderr)
            last_reported = reason
        delay = QUEUE_POLL_SECONDS
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - time.monotonic()))
        time.sleep(delay)


def _git_metadata() -> dict[str, str | bool]:
    """Read HEAD, its committed tree and tracked/untracked dirtiness in cwd."""
    return {
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


def _execute(
    invocation: _Invocation, meminfo_path: Path, lock_directory: Path | None
) -> int:
    resources = ExitStack()
    try:
        cwd = Path.cwd()
        count = os.cpu_count()
        if count is None or count < SLOT_COUNT:
            raise ValueError(f"at least {SLOT_COUNT} machine CPUs are required")
        _git_metadata()  # Fail fast before waiting; receipt identity is read at admission.
        if lock_directory is None:
            common = subprocess.check_output(
                ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                text=True,
            ).strip()
            lock_directory = Path(common) / "run-heavy"
        slot, lock_fds, wait_seconds = resources.enter_context(
            _acquired_slot(invocation, meminfo_path, lock_directory)
        )
        metadata = _git_metadata()
        width = count // SLOT_COUNT
        cpus = (
            list(range(SLOT_COUNT * width))
            if invocation.exclusive
            else list(range(slot * width, (slot + 1) * width))
        )
        started_utc = datetime.now(UTC).isoformat()
        started = time.monotonic()
        child, pending = resources.enter_context(
            _forwarding_child(invocation, cpus, lock_fds)
        )
    except _WaitLimitExpired as error:
        resources.close()
        print(f"run_heavy: {error}", file=sys.stderr)
        return 75
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        resources.close()
        print(f"run_heavy: {error}", file=sys.stderr)
        return 2

    timed_out = False
    residual_killed = False
    try:
        with resources:
            for sig in pending:
                _signal_group(child.pid, sig)
            try:
                code = child.wait(timeout=invocation.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_group(child)
                code = 124
            if not timed_out:
                residual_killed = _group_running(child.pid)
                if residual_killed:
                    _stop_group(child)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"run_heavy: post-launch failure: {error}", file=sys.stderr)
        code = 70
        try:
            _signal_group(child.pid, signal.SIGKILL)
            child.wait(timeout=KILL_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired) as cleanup_error:
            print(f"run_heavy: cleanup failed: {cleanup_error}", file=sys.stderr)
    if code < 0:
        # Popen encodes signal death as a negative signal number.
        code = 128 + abs(code)
    print(
        json.dumps(
            {
                "command": invocation.command,
                "cpus": cpus,
                "env": CHILD_ENV,
                "slot": None if invocation.exclusive else slot,
                "exclusive": invocation.exclusive,
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
    """Run `--log PATH [--wait-limit SECONDS] [--exclusive] TIMEOUT -- COMMAND...`.

    Call on the main thread, on Linux only, within a committed Git worktree.
    Run in cwd after MemAvailable >= 8 GB and a repository-wide CPU slot is free.
    --exclusive waits for both slots, releasing partial reservations before retry.
    Each slot uses cpu_count // 2 consecutive machine CPUs, ignoring an odd last
    CPU. Exclusive commands use the union of both slots' CPUs. At least two CPUs
    and affinity access to all reserved CPUs are required.
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
    After KILL, wait at most another 10 seconds for the leader and group; report
    survivors to stderr and still emit the receipt with timeout/residual flags.
    Zombies and dead processes do not count as running group members.
    INT/TERM received by the runner are forwarded to the entire child group.
    Check Git identity before queuing, then reread HEAD/tree/dirty after admission.
    Once the child starts, print one JSON receipt with that identity and outcome.
    exclusive is a bool; slot is null for exclusive commands, otherwise its index.
    cpus lists the command's affinity. Children inherit every reserved slot lock.
    Post-launch OS, value or Git-command failures report to stderr and return 70
    with a receipt, after best-effort group cleanup. Receipt-output failures
    propagate to the caller because no receipt can be written on that path.
    Prelaunch errors print their reason to stderr, return 2 and print no receipt.
    """
    if sys.platform != "linux":
        print("run_heavy: only Linux is supported", file=sys.stderr)
        return 2
    try:
        invocation = _parse(argv)
    except SystemExit as error:
        return int(error.code or 0)
    return _execute(invocation, meminfo_path, lock_directory)


if __name__ == "__main__":
    sys.exit(main())
