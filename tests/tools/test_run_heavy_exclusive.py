"""Exclusive admission and receipts at the run-heavy CLI/process boundary."""

from __future__ import annotations

import fcntl
import json
import os
import select
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
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
    (tmp_path / ".git" / "meminfo").write_text("MemAvailable: 999999999 kB\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("exclusive", [False, True])
def test_receipt_and_child_affinity_identify_reserved_capacity(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exclusive: bool,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 5)
    original = os.sched_getaffinity(0)
    log = repo / "child.log"
    try:
        os.sched_setaffinity(0, {1})
        assert (
            main(
                [
                    "--log",
                    str(log),
                    *(["--exclusive"] if exclusive else []),
                    "10",
                    "--",
                    sys.executable,
                    "-c",
                    "import os; print(sorted(os.sched_getaffinity(0)))",
                ],
                meminfo_path=repo / ".git" / "meminfo",
            )
            == 0
        )
        assert os.sched_getaffinity(0) == {1}
    finally:
        os.sched_setaffinity(0, original)

    expected = [0, 1, 2, 3] if exclusive else [0, 1]
    receipt = json.loads(capsys.readouterr().out)
    assert json.loads(log.read_text()) == expected
    assert receipt["cpus"] == expected
    assert receipt["slot"] == (None if exclusive else 0)
    assert receipt["exclusive"] is exclusive


@contextmanager
def exclusive_runner(repo: Path, source: str) -> Generator[subprocess.Popen[str]]:
    """Run an exclusive peer with isolated locks; terminate it on failed assertions."""
    tools = Path(__file__).resolve().parents[2] / "tools"
    args = [
        "--exclusive",
        "--log",
        str(repo / "peer.log"),
        "30",
        "--",
        sys.executable,
        "-c",
        source,
    ]
    runner_source = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(tools)!r}); from run_heavy import main; "
        f"sys.exit(main({args!r}, "
        f"meminfo_path=Path({str(repo / '.git' / 'meminfo')!r})))"
    )
    with subprocess.Popen(
        [sys.executable, "-c", runner_source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as runner:
        try:
            yield runner
        finally:
            if runner.poll() is None:
                runner.terminate()
            # Termination is forwarded to the child group, then the runner exits.
            runner.wait(timeout=30)


@pytest.mark.parametrize("exclusive", [False, True])
def test_request_waits_until_exclusive_command_finishes(
    repo: Path, capsys: pytest.CaptureFixture[str], exclusive: bool
) -> None:
    ready = repo / "ready"
    release = repo / "release"
    os.mkfifo(ready)
    os.mkfifo(release)
    ready_fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    release_fd = os.open(release, os.O_RDWR | os.O_NONBLOCK)
    source = (
        f"with open({str(ready)!r}, 'w') as stream: stream.write('ready')\n"
        f"with open({str(release)!r}, 'rb') as stream: stream.read(1)\n"
    )
    args = [
        "--log",
        str(repo / "contender.log"),
        *(["--exclusive"] if exclusive else []),
        "--wait-limit",
        "0",
        "30",
        "--",
        sys.executable,
        "-c",
        "print('normal')",
    ]
    try:
        with exclusive_runner(repo, source) as runner:
            try:
                # The child writes readiness before blocking on our explicit release.
                assert select.select([ready_fd], [], [], 30)[0], "peer did not start"
                assert os.read(ready_fd, 4096) == b"ready"
                assert main(args, meminfo_path=repo / ".git" / "meminfo") == 75
                output = capsys.readouterr()
                assert output.out == ""
                assert "slot" in output.err
                assert not (repo / "contender.log").exists()
            finally:
                os.write(release_fd, b"x")
            # Release lets the child finish; communicate joins both child and runner.
            stdout, stderr = runner.communicate(timeout=30)
            assert runner.returncode == 0, stderr
            assert json.loads(stdout)["exclusive"] is True
        assert main(args, meminfo_path=repo / ".git" / "meminfo") == 0
        assert json.loads(capsys.readouterr().out)["exclusive"] is exclusive
        assert (repo / "contender.log").read_text() == "normal\n"
    finally:
        os.close(ready_fd)
        os.close(release_fd)


def test_partial_reservation_is_released_before_waiting_then_completes(
    repo: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = repo / ".git" / "run-heavy"
    directory.mkdir()
    observed_release = False
    with (directory / "slot-1.lock").open("a+b") as occupied:
        fcntl.flock(occupied, fcntl.LOCK_EX)
        original_write = sys.stderr.write

        def report_waiting(message: str) -> int:
            nonlocal observed_release
            if "waiting:" in message:
                # A waiting report follows a failed attempt, before its next retry.
                # The same-thread observer cannot race another reservation attempt.
                with (directory / "slot-0.lock").open("a+b") as available:
                    fcntl.flock(available, fcntl.LOCK_EX | fcntl.LOCK_NB)
                observed_release = True
                occupied.close()
            return original_write(message)

        monkeypatch.setattr(sys.stderr, "write", report_waiting)
        assert (
            main(
                [
                    "--exclusive",
                    "--log",
                    str(repo / "contender.log"),
                    "30",
                    "--",
                    sys.executable,
                    "-c",
                    "print('contender')",
                ],
                meminfo_path=repo / ".git" / "meminfo",
            )
            == 0
        )
    assert observed_release
    assert json.loads(capsys.readouterr().out)["exclusive"] is True
    assert (repo / "contender.log").read_text() == "contender\n"
