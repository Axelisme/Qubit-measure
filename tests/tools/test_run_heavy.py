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
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from run_heavy import main

pytestmark = pytest.mark.requires_subprocess


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


def run(repo: Path, argv: list[str]) -> int:
    """Run main against the disposable repository's ample-memory fixture."""
    return main(argv, meminfo_path=repo / ".git" / "meminfo")


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
    command = group_command(
        ready,
        "signal.pause()",
        (
            "def stopped(signum, frame):\n"
            "    print(os.getpid(), signum, flush=True)\n"
            "    raise SystemExit(128 + signum)\n"
            "signal.signal(signal.SIGINT, stopped)\n"
            "signal.signal(signal.SIGTERM, stopped)\n"
        ),
    )

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
