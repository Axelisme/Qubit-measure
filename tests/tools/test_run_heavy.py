"""Run-heavy CLI contracts using small children in disposable Git repositories."""

from __future__ import annotations

import json
import os
import runpy
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType

import pytest
from run_heavy import main

from tests.tools._run_heavy_support import (
    assert_group_stopped,
    forwarding_group_command,
    group_command,
    held_slots,
    observed_group,
    run,
    unreapable_leader,
)

pytestmark = pytest.mark.requires_subprocess


def test_help_returns_success_without_launching_a_command() -> None:
    assert main(["-h"]) == 0


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
            "--allow-empty",
        ],
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("initial")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "tracked file",
        ],
        check=True,
    )
    (tmp_path / ".git" / "meminfo").write_text("MemAvailable: 999999999 kB\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_command_receives_slot_environment_and_combined_log(
    repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    monkeypatch.setenv("QT_QPA_PLATFORM", "caller-platform")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    log = repo / "child.log"
    source = (
        "import json, os, sys; "
        "print(json.dumps({'cpus': sorted(os.sched_getaffinity(0)), "
        "'env': {k: os.environ[k] for k in "
        "['QT_QPA_PLATFORM', 'MPLBACKEND', 'OMP_NUM_THREADS', "
        "'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']}, "
        "'cwd': os.getcwd()}), flush=True); "
        "print('child stderr', file=sys.stderr, flush=True)"
    )
    command = [sys.executable, "-c", source]

    assert run(repo, ["--log", str(log), "30", "--", *command]) == 0

    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    lines = log.read_text().splitlines()
    child = json.loads(lines[0])
    expected_env = {
        "QT_QPA_PLATFORM": "offscreen",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    assert child == {
        "cpus": [0],
        "env": expected_env,
        "cwd": str(repo),
    }
    assert lines[1] == "child stderr"
    assert receipt["command"] == command
    assert receipt["cpus"] == child["cpus"]
    assert receipt["env"] == expected_env
    assert receipt["slot"] == 0
    assert captured.err == ""
    assert os.environ["QT_QPA_PLATFORM"] == "caller-platform"
    assert os.environ["OMP_NUM_THREADS"] == "99"


@pytest.mark.parametrize(
    "state", ["clean", "staged", "unstaged", "untracked", "linked"]
)
def test_receipt_identifies_callers_worktree_and_child_exit(
    repo: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    cwd = repo
    if state == "linked":
        cwd = repo / "linked"
        subprocess.run(["git", "worktree", "add", "-qb", "other", str(cwd)], check=True)
        monkeypatch.chdir(cwd)
    elif state == "untracked":
        subprocess.run(["git", "config", "status.showUntrackedFiles", "no"], check=True)
        (repo / "new.txt").write_text("untracked")
    elif state != "clean":
        (repo / "tracked.txt").write_text("changed")
        if state == "staged":
            subprocess.run(["git", "add", "tracked.txt"], check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    log = repo.parent / (repo.name + ".log")
    command = [sys.executable, "-c", "raise SystemExit(7)"]

    assert run(repo, ["--log", str(log), "30", "--", *command]) == 7

    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    receipt = json.loads(output)
    assert set(receipt) == {
        "command",
        "cwd",
        "head",
        "tree",
        "dirty",
        "slot",
        "cpus",
        "env",
        "wait_seconds",
        "started_utc",
        "finished_utc",
        "elapsed_seconds",
        "exit_code",
        "timed_out",
        "residual_killed",
        "log",
        "exclusive",
    }
    assert receipt["command"] == command
    assert receipt["cwd"] == str(cwd)
    assert receipt["head"] == head
    assert receipt["tree"] == tree
    assert receipt["dirty"] is (state in {"staged", "unstaged", "untracked"})
    assert receipt["wait_seconds"] >= 0
    assert receipt["elapsed_seconds"] >= 0
    assert datetime.fromisoformat(receipt["started_utc"]).utcoffset() == timedelta(0)
    assert datetime.fromisoformat(receipt["finished_utc"]).utcoffset() == timedelta(0)
    assert receipt["exit_code"] == 7
    assert receipt["timed_out"] is False
    assert receipt["residual_killed"] is False
    assert receipt["log"] == str(log)


@pytest.mark.parametrize("dirty_after_admission", [False, True])
def test_receipt_refreshes_git_identity_after_queue_admission(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dirty_after_admission: bool,
) -> None:
    meminfo = repo / ".git" / "meminfo"
    meminfo.write_text("MemAvailable: 1 kB\n")
    if not dirty_after_admission:
        (repo / "tracked.txt").write_text("queued edit")
    queued_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    original_sleep = time.sleep
    admitted = False

    def change_worktree_while_queued(seconds: float) -> None:
        nonlocal admitted
        if admitted:
            # Popen.wait also sleeps; only the admission sleep changes the worktree.
            original_sleep(seconds)
            return
        admitted = True
        (repo / "tracked.txt").write_text("admitted commit")
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qam",
                "commit while queued",
            ],
            check=True,
        )
        if dirty_after_admission:
            (repo / "tracked.txt").write_text("admitted edit")
        meminfo.write_text("MemAvailable: 999999999 kB\n")

    monkeypatch.setattr(time, "sleep", change_worktree_while_queued)
    assert (
        run(
            repo,
            [
                "--log",
                str(repo / ".git" / "child.log"),
                "30",
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    assert head != queued_head
    assert receipt["head"] == head
    assert receipt["tree"] == tree
    assert receipt["dirty"] is dirty_after_admission


@pytest.mark.parametrize(
    "failure",
    [
        "timeout-zero",
        "timeout-negative",
        "timeout-nan",
        "timeout-inf",
        "timeout-text",
        "missing-command",
        "missing-log",
        "missing-separator",
    ],
)
def test_invalid_usage_never_starts_a_child(
    repo: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    log = repo / "child.log"
    timeout = {
        "timeout-zero": "0",
        "timeout-negative": "-1",
        "timeout-nan": "nan",
        "timeout-inf": "inf",
        "timeout-text": "not-a-number",
    }.get(failure, "30")
    marker = repo / "child-started"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]
    args = ["--log", str(log), timeout, "--", *command]
    if failure == "missing-command":
        args = ["--log", str(log), timeout, "--"]
    elif failure == "missing-log":
        args = [timeout, "--", *command]
    elif failure == "missing-separator":
        args.remove("--")

    assert run(repo, args) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.strip()
    assert not marker.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "not-git",
        "non-linux",
        "one-cpu",
        "unknown-cpus",
        "log-directory",
        "affinity",
        "affinity-clipped",
        "missing-executable",
        "not-executable",
    ],
)
def test_prelaunch_errors_return_two_without_receipt(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    log = repo / "child.log"
    marker = repo / "child-started"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]
    if failure == "not-git":
        outside = repo / "outside"
        outside.mkdir()
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(repo))
        monkeypatch.chdir(outside)
    elif failure == "non-linux":
        monkeypatch.setattr(sys, "platform", "darwin")
    elif failure in {"one-cpu", "unknown-cpus"}:
        monkeypatch.setattr(
            os, "cpu_count", lambda: 1 if failure == "one-cpu" else None
        )
    elif failure == "log-directory":
        log = repo
    elif failure == "affinity":

        def forbidden_affinity(pid: int, mask: object) -> None:
            raise PermissionError("affinity forbidden by cpuset")

        monkeypatch.setattr(os, "sched_setaffinity", forbidden_affinity)
    elif failure == "affinity-clipped":
        monkeypatch.setattr(os, "cpu_count", lambda: 4)
        original_setaffinity = os.sched_setaffinity

        def clipped_affinity(pid: int, mask: list[int] | set[int]) -> None:
            original_setaffinity(pid, {0} if set(mask) == {0, 1} else mask)

        monkeypatch.setattr(os, "sched_setaffinity", clipped_affinity)
    elif failure == "missing-executable":
        command = [str(repo / "missing-command")]
    elif failure == "not-executable":
        executable = repo / "not-executable"
        executable.write_text("not executable")
        command = [str(executable)]

    assert run(repo, ["--log", str(log), "30", "--", *command]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.strip()
    assert not marker.exists()


@pytest.mark.parametrize(
    "failure", ["group-scan", "affinity-restore", "handler-restore"]
)
def test_postlaunch_cleanup_failure_still_emits_receipt(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    if failure == "group-scan":
        original_iterdir = Path.iterdir

        def scan(path: Path) -> Generator[Path]:
            if path == Path("/proc"):
                raise PermissionError("cleanup unavailable")
            yield from original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", scan)
    elif failure == "affinity-restore":
        original_affinity = os.sched_setaffinity
        calls: list[set[int]] = []

        def restore_affinity(pid: int, mask: list[int] | set[int]) -> None:
            original_affinity(pid, mask)
            calls.append(set(mask))
            # The runner sets the slot CPUs, launches, then restores the caller's
            # mask; fail the restore by call order, since the caller may already
            # be pinned to the same CPUs as the slot.
            if len(calls) == 2:
                raise OSError("cleanup unavailable")

        monkeypatch.setattr(os, "sched_setaffinity", restore_affinity)
    else:
        original_signal = signal.signal
        handler = signal.getsignal(signal.SIGTERM)

        def restore_handler(
            signum: int,
            action: Callable[[int, FrameType | None], object] | int | None,
        ) -> Callable[[int, FrameType | None], object] | int | None:
            result = original_signal(signum, action)
            # Fail on the last restoration, after both real handlers are restored.
            if signum == signal.SIGTERM and action == handler:
                raise ValueError("cleanup unavailable")
            return result

        monkeypatch.setattr(signal, "signal", restore_handler)

    command = [sys.executable, "-c", "print('started')"]
    log = repo / "child.log"
    assert run(repo, ["--log", str(log), "30", "--", *command]) == 70
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert receipt["exit_code"] == 70
    assert receipt["command"] == command
    assert "cleanup unavailable" in captured.err
    assert log.read_text() == "started\n"


def test_receipt_output_failure_is_not_a_prelaunch_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable_receipt(value: object) -> str:
        raise ValueError("receipt output unavailable")

    monkeypatch.setattr(json, "dumps", unavailable_receipt)
    log = repo / "child.log"
    with pytest.raises(ValueError, match="receipt output unavailable"):
        run(
            repo,
            ["--log", str(log), "30", "--", sys.executable, "-c", "print('started')"],
        )
    assert log.read_text() == "started\n"
    assert capsys.readouterr().err == ""


def test_non_linux_entrypoint_reports_platform_without_loading_fcntl(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = Path(__file__).resolve().parents[2] / "tools" / "run_heavy.py"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "fcntl", None)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(tool), run_name="__main__")
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "only Linux" in output.err


def test_slot_uses_machine_cpus_even_when_caller_is_pinned(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 5)
    original = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, {1})
        log = repo / "child.log"
        assert (
            run(
                repo,
                [
                    "--log",
                    str(log),
                    "30",
                    "--",
                    sys.executable,
                    "-c",
                    "import os; print(sorted(os.sched_getaffinity(0)))",
                ],
            )
            == 0
        )
        assert os.sched_getaffinity(0) == {1}
    finally:
        os.sched_setaffinity(0, original)
    assert json.loads(log.read_text()) == [0, 1]
    assert json.loads(capsys.readouterr().out)["cpus"] == [0, 1]


@pytest.mark.parametrize("ignore_term", [False, True])
def test_timeout_stops_the_whole_group_and_escalates_if_needed(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ignore_term: bool,
) -> None:
    ready = repo / "ready"
    preamble = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else ""
    command = group_command(ready, "signal.pause()", preamble)
    delivered: list[int] = []
    original_killpg = os.killpg
    original_sleep = time.sleep
    clock = [100.0]

    def expire(timeout: float | None) -> None:
        assert timeout == 1.25
        raise subprocess.TimeoutExpired(command, 1.25)

    def killpg(pgid: int, signum: int) -> None:
        delivered.append(signum)
        original_killpg(pgid, signum)

    def advance(seconds: float) -> None:
        # Skip the TERM grace on the fake clock, but yield for actual signal delivery.
        clock[0] += max(seconds, 11.0)
        original_sleep(0)

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", advance)
    with observed_group(monkeypatch, command, ready, expire) as pids:
        assert (
            run(repo, ["--log", str(repo / "child.log"), "1.25", "--", *command]) == 124
        )
        receipt = json.loads(capsys.readouterr().out)
        assert receipt["exit_code"] == 124
        assert receipt["timed_out"] is True
        assert receipt["residual_killed"] is False
        assert signal.SIGTERM in delivered
        if ignore_term:
            assert signal.SIGKILL in delivered
        assert_group_stopped(pids)


@pytest.mark.parametrize("timed_out", [False, True])
@pytest.mark.parametrize("stuck_leader", [False, True])
def test_sigkill_survivor_does_not_block_receipt_forever(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stuck_leader: bool,
    timed_out: bool,
) -> None:
    ready = repo / "ready"
    command = group_command(
        ready,
        "signal.pause()" if timed_out else "raise SystemExit(7)",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n",
    )
    original_killpg = os.killpg
    original_iterdir = Path.iterdir
    original_read_text = Path.read_text
    original_sleep = time.sleep
    clock = [100.0]
    killed = False
    cleaning = False
    survivor = Path("/proc/999999999")

    def expire(_timeout: float | None) -> None:
        if timed_out:
            raise subprocess.TimeoutExpired(command, 1)

    def killpg(pgid: int, signum: int) -> None:
        nonlocal killed, cleaning
        original_killpg(pgid, signum)
        cleaning = True
        if signum == signal.SIGKILL:
            killed = True

    def proc_entries(path: Path) -> Generator[Path]:
        yield from original_iterdir(path)
        if path == Path("/proc") and killed:
            yield survivor

    def proc_status(path: Path) -> str:
        if path == survivor / "stat":
            return f"999999999 (unkillable) D 1 {pids['leader']} 0\n"
        return original_read_text(path)

    def advance(seconds: float) -> None:
        if not cleaning:
            # Let the leader finish normally before accelerating cleanup deadlines.
            original_sleep(seconds)
            return
        # Jump over the TERM grace without waiting in real time; yield for real SIGKILL.
        clock[0] += max(seconds, 11.0)
        original_sleep(0)
        assert clock[0] < 300, "SIGKILL cleanup exceeded its bounded wait"

    with (
        observed_group(monkeypatch, command, ready, expire) as pids,
        monkeypatch.context() as boundary,
        unreapable_leader(monkeypatch, command, lambda: killed and stuck_leader),
    ):
        boundary.setattr(os, "killpg", killpg)
        boundary.setattr(Path, "iterdir", proc_entries)
        boundary.setattr(Path, "read_text", proc_status)
        boundary.setattr(time, "monotonic", lambda: clock[0])
        boundary.setattr(time, "sleep", advance)
        expected_code = 124 if timed_out else 7
        assert (
            run(repo, ["--log", str(repo / "child.log"), "1", "--", *command])
            == expected_code
        )
        receipt = json.loads(capsys.readouterr().out)
        assert killed
        assert receipt["exit_code"] == expected_code
        assert receipt["timed_out"] is timed_out
        assert receipt["residual_killed"] is (not timed_out)


def test_exited_leader_leaves_no_running_group_members(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ready = repo / "ready"
    command = group_command(ready, "raise SystemExit(7)")
    with observed_group(monkeypatch, command, ready) as pids:
        assert run(repo, ["--log", str(repo / "child.log"), "30", "--", *command]) == 7
        receipt = json.loads(capsys.readouterr().out)
        assert receipt["exit_code"] == 7
        assert receipt["timed_out"] is False
        assert receipt["residual_killed"] is True
        assert_group_stopped(pids)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_runner_forwards_signal_to_group_and_restores_handlers(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    signum: int,
) -> None:
    ready = repo / "ready"
    command = forwarding_group_command(ready)

    def unhandled(signum: int, frame: object) -> None:
        pytest.fail(f"runner did not install forwarding for signal {signum}")

    previous = signal.signal(signum, unhandled)
    log = repo / "child.log"
    try:
        with observed_group(
            monkeypatch, command, ready, lambda _timeout: os.kill(os.getpid(), signum)
        ) as pids:
            assert run(repo, ["--log", str(log), "30", "--", *command]) == 128 + signum
            receipt = json.loads(capsys.readouterr().out)
            assert receipt["exit_code"] == 128 + signum
            assert receipt["timed_out"] is False
            assert {
                tuple(map(int, line.split())) for line in log.read_text().splitlines()
            } == {(pid, signum) for pid in pids.values()}
            assert_group_stopped(pids)
        assert signal.getsignal(signum) is unhandled
    finally:
        signal.signal(signum, previous)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
def test_signal_death_uses_shell_exit_status(
    repo: Path,
    capsys: pytest.CaptureFixture[str],
    signum: int,
) -> None:
    command = [
        sys.executable,
        "-c",
        f"import os, signal; os.kill(os.getpid(), {signum})",
    ]
    assert (
        run(repo, ["--log", str(repo / "child.log"), "30", "--", *command])
        == 128 + signum
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["exit_code"] == 128 + signum
    assert receipt["timed_out"] is False


@pytest.mark.parametrize(
    "contents", ["MemFree: 999999999 kB\n", "MemAvailable: broken kB\n"]
)
def test_unreadable_available_memory_fails_before_launch(
    repo: Path, capsys: pytest.CaptureFixture[str], contents: str
) -> None:
    meminfo = repo / ".git" / "meminfo"
    meminfo.write_text(contents)
    log = repo / "child.log"
    assert (
        main(
            ["--log", str(log), "30", "--", sys.executable, "-c", "print('started')"],
            meminfo_path=meminfo,
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "MemAvailable" in captured.err
    assert not log.exists()


def test_low_memory_wait_reports_once_then_starts_when_memory_recovers(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    meminfo = repo / ".git" / "meminfo"
    meminfo.write_text("MemAvailable: 7812499 kB\n")
    log = repo / "child.log"
    clock = [100.0]
    waits: list[str] = []
    original_sleep = time.sleep

    def memory_recovers(_seconds: float) -> None:
        if len(waits) == 2:
            # Popen.wait shares time.sleep; admission is done, so only yield to the child.
            original_sleep(0)
            return
        output = capsys.readouterr()
        assert output.out == ""
        assert not log.exists()
        waits.append(output.err)
        clock[0] += 2.0
        if len(waits) == 2:
            meminfo.write_text("MemAvailable: 7812500 kB\n")

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", memory_recovers)
    assert (
        main(
            ["--log", str(log), "30", "--", sys.executable, "-c", "print('started')"],
            meminfo_path=meminfo,
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert len(waits) == 2
    assert "MemAvailable" in waits[0]
    assert waits[1] == ""
    assert receipt["wait_seconds"] == 4.0
    assert log.read_text() == "started\n"


@pytest.mark.parametrize("limit", ["0", "2.5"])
def test_wait_limit_expires_without_receipt_or_child(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    limit: str,
) -> None:
    meminfo = repo / ".git" / "meminfo"
    meminfo.write_text("MemAvailable: 1 kB\n")
    log = repo / "child.log"
    clock = [100.0]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", advance)
    assert (
        main(
            [
                "--log",
                str(log),
                "--wait-limit",
                limit,
                "30",
                "--",
                sys.executable,
                "-c",
                "print('started')",
            ],
            meminfo_path=meminfo,
        )
        == 75
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "wait limit" in captured.err
    assert "MemAvailable" in captured.err
    assert not log.exists()


def test_linked_worktree_uses_other_slot_while_peer_holds_first(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    linked = repo / "linked"
    subprocess.run(["git", "worktree", "add", "-qb", "peer", str(linked)], check=True)
    monkeypatch.chdir(linked)
    monkeypatch.setattr(os, "cpu_count", lambda: 5)
    log = repo / "child.log"
    args = [
        "--log",
        str(log),
        "30",
        "--",
        sys.executable,
        "-c",
        "import os; print(sorted(os.sched_getaffinity(0)))",
    ]
    with held_slots(repo / ".git" / "run-heavy", [0]):
        for _ in range(2):
            assert run(repo, args) == 0
            receipt = json.loads(capsys.readouterr().out)
            assert receipt["slot"] == 1
            assert receipt["cpus"] == [2, 3]
            assert json.loads(log.read_text()) == [2, 3]


def test_full_slots_wait_for_release_and_report_only_reason_changes(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    meminfo = repo / ".git" / "meminfo"
    log = repo / "child.log"
    clock = [100.0]
    waits: list[str] = []
    original_sleep = time.sleep
    with held_slots(repo / ".git" / "run-heavy", [0, 1]) as release:

        def advance(_seconds: float) -> None:
            if len(waits) == 4:
                # Popen.wait shares time.sleep; keep admission observations unchanged.
                original_sleep(0)
                return
            output = capsys.readouterr()
            assert output.out == ""
            assert not log.exists()
            waits.append(output.err)
            clock[0] += 2.0
            if len(waits) == 2:
                meminfo.write_text("MemAvailable: 1 kB\n")
                release(1)
            elif len(waits) == 4:
                meminfo.write_text("MemAvailable: 999999999 kB\n")

        with monkeypatch.context() as boundary:
            boundary.setattr(time, "monotonic", lambda: clock[0])
            boundary.setattr(time, "sleep", advance)
            assert (
                run(
                    repo,
                    [
                        "--log",
                        str(log),
                        "30",
                        "--",
                        sys.executable,
                        "-c",
                        "print('started')",
                    ],
                )
                == 0
            )

    receipt = json.loads(capsys.readouterr().out)
    assert len(waits) == 4
    assert "slot" in waits[0]
    assert waits[1] == ""
    assert "MemAvailable" in waits[2]
    assert waits[3] == ""
    assert receipt["slot"] == 1
    assert receipt["wait_seconds"] == 8.0
    assert log.read_text() == "started\n"


def test_crashed_runner_keeps_slot_until_inheriting_descendant_exits(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ready = repo / "ready"
    release = repo / "release"
    os.mkfifo(ready)
    os.mkfifo(release)
    ready_fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    release_fd = os.open(release, os.O_RDWR | os.O_NONBLOCK)
    lock_directory = repo / ".git" / "injected-locks"
    meminfo = repo / ".git" / "meminfo"
    source = (
        "import json, os, signal\n"
        "reader, writer = os.pipe()\n"
        "worker = os.fork()\n"
        "if worker == 0:\n"
        "    os.close(writer)\n"
        f"    with open({str(ready)!r}, 'w') as stream:\n"
        "        print(json.dumps({'leader': os.getppid(), 'worker': os.getpid()}), file=stream, flush=True)\n"
        "    os.read(reader, 1)\n"
        f"    with open({str(ready)!r}, 'w') as stream:\n"
        "        print('orphan', file=stream, flush=True)\n"
        f"    with open({str(release)!r}, 'rb') as stream:\n"
        "        stream.read(1)\n"
        "    os._exit(0)\n"
        "else:\n"
        "    os.close(reader)\n"
        "    signal.pause()\n"
    )
    args = ["--log", str(repo / "owner.log"), "30", "--", sys.executable, "-c", source]
    tools = Path(__file__).resolve().parents[2] / "tools"
    runner_source = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(tools)!r}); from run_heavy import main; "
        f"sys.exit(main({args!r}, meminfo_path=Path({str(meminfo)!r}), "
        f"lock_directory=Path({str(lock_directory)!r})))"
    )
    pids: dict[str, int] = {}

    def receive() -> bytes:
        # The worker writes once at startup and once on pipe EOF after its leader dies.
        assert select.select([ready_fd], [], [], 30)[0], "worker handshake missing"
        return os.read(ready_fd, 4096).strip()

    def assert_waiting() -> None:
        assert (
            main(
                [
                    "--log",
                    str(repo / "contender.log"),
                    "--wait-limit",
                    "0",
                    "30",
                    "--",
                    sys.executable,
                    "-c",
                    "print('started')",
                ],
                meminfo_path=meminfo,
                lock_directory=lock_directory,
            )
            == 75
        )
        output = capsys.readouterr()
        assert output.out == ""
        assert "slot" in output.err

    try:
        with (
            held_slots(lock_directory, [1]),
            subprocess.Popen(
                [sys.executable, "-c", runner_source],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as runner,
        ):
            try:
                pids.update(json.loads(receive()))
                runner.kill()
                runner.wait(timeout=30)
                assert_waiting()
                os.kill(pids["leader"], signal.SIGKILL)
                assert receive() == b"orphan"
                assert_waiting()
                os.write(release_fd, b"x")
                # Admission joins the worker's explicit release through the real flock.
                assert (
                    main(
                        [
                            "--log",
                            str(repo / "contender.log"),
                            "--wait-limit",
                            "30",
                            "30",
                            "--",
                            sys.executable,
                            "-c",
                            "print('started')",
                        ],
                        meminfo_path=meminfo,
                        lock_directory=lock_directory,
                    )
                    == 0
                )
                assert json.loads(capsys.readouterr().out)["slot"] == 0
            finally:
                if runner.poll() is None:
                    runner.kill()
                runner.wait(timeout=30)
                for pid in pids.values():
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
    finally:
        for fd in [ready_fd, release_fd]:
            os.close(fd)
