"""Exercise basetemp ownership at the pytest session/process boundary."""

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = [pytest.mark.requires_subprocess, pytest.mark.uses_wall_clock]


@pytest.fixture
def checkout(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the real pytest configuration in a small synthetic checkout."""
    repo = Path(__file__).resolve().parents[1]
    tests = pytester.path / "tests"
    tests.mkdir()
    for name in (
        "__init__.py",
        "conftest.py",
        "_cache_environment.py",
        "_temp_directory_guard.py",
    ):
        shutil.copyfile(repo / "tests" / name, tests / name)
    shutil.copyfile(repo / "pyproject.toml", pytester.path / "pyproject.toml")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(repo), str(repo / "lib"))))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    return tests


@contextmanager
def running_session(pytester: pytest.Pytester, label: str):
    """Hold a child session at its test's stdin barrier; always release and join."""
    with (pytester.path / f"{label}.log").open("w") as output:
        process = pytester.popen(
            [sys.executable, "-m", "pytest", "-q", "-s", "-n", "0"],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            yield process
        finally:
            try:
                process.communicate("release\n", timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise


def ready_marker(pytester: pytest.Pytester, process: subprocess.Popen[str]) -> Path:
    """Wait for the child's published tmp_path, failing if it exits or hangs."""
    ready = pytester.path / f"{process.pid}.ready"
    deadline = time.monotonic() + 30
    # The child publishes after writing its marker, then waits for our release.
    while not ready.exists():
        assert process.poll() is None, "child exited before publishing tmp_path"
        assert time.monotonic() < deadline, "child did not reach the stdin barrier"
        time.sleep(0.01)
    return Path(ready.read_text())


def test_concurrent_sessions_own_and_clean_only_their_basetemp(
    pytester: pytest.Pytester, checkout: Path
) -> None:
    (checkout / "test_barrier.py").write_text(
        """
import os
import sys


def test_barrier(tmp_path, pytestconfig):
    marker = tmp_path / 'marker'
    marker.write_text('still here')
    ready = pytestconfig.rootpath / f'{os.getpid()}.ready'
    pending = ready.with_suffix('.pending')
    pending.write_text(str(marker))
    pending.replace(ready)
    assert sys.stdin.readline().strip() == 'release'
    assert marker.read_text() == 'still here'
"""
    )
    with running_session(pytester, "second") as second:
        second_marker = ready_marker(pytester, second)
        second_base = second_marker.parent.parent
        with running_session(pytester, "first") as first:
            first_marker = ready_marker(pytester, first)
            first_base = first_marker.parent.parent
            assert first_base != second_base
            assert (
                first_base.parent == second_base.parent == pytester.path / ".pytest-tmp"
            )
            assert first_marker.read_text() == second_marker.read_text() == "still here"
        assert first.returncode == 0, (pytester.path / "first.log").read_text()
        assert not first_base.exists()
        assert second_marker.read_text() == "still here"
    assert second.returncode == 0, (pytester.path / "second.log").read_text()
    assert not second_base.exists()


def test_explicit_basetemp_keeps_pytest_cleanup_semantics(
    pytester: pytest.Pytester, checkout: Path
) -> None:
    chosen = pytester.path / "chosen"
    chosen.mkdir()
    stale = chosen / "stale"
    stale.write_text("previous run")
    (checkout / "test_explicit.py").write_text(
        """
def test_explicit(tmp_path, tmp_path_factory, pytestconfig):
    base = tmp_path_factory.getbasetemp()
    assert base == pytestconfig.rootpath / 'chosen'
    assert not (base / 'stale').exists()
    (tmp_path / 'marker').write_text('retained')
"""
    )
    result = pytester.runpytest_subprocess("-q", "-n", "0", f"--basetemp={chosen}")
    result.assert_outcomes(passed=1)
    assert {marker.read_text() for marker in chosen.glob("test_*/marker")} == {
        "retained"
    }


def test_xdist_workers_share_the_invocation_root(
    pytester: pytest.Pytester, checkout: Path
) -> None:
    (checkout / "test_workers.py").write_text(
        """
import json


def test_worker(tmp_path, tmp_path_factory, pytestconfig, worker_id):
    marker = tmp_path / 'marker'
    marker.write_text(worker_id)
    report = pytestconfig.rootpath / f'{worker_id}.json'
    report.write_text(json.dumps({
        'base': str(tmp_path_factory.getbasetemp()),
        'marker': marker.read_text(),
    }))
"""
    )
    # runpytest_subprocess adds --basetemp itself, masking default allocation.
    result = pytester.run(
        sys.executable, "-m", "pytest", "-q", "-n", "2", "--dist=each", timeout=30
    )
    result.assert_outcomes(passed=2)
    reports = [
        json.loads((pytester.path / f"gw{i}.json").read_text()) for i in range(2)
    ]
    bases = [Path(report["base"]) for report in reports]
    assert bases[0] != bases[1]
    assert bases[0].parent == bases[1].parent
    invocation_root = bases[0].parent
    assert invocation_root.parent == pytester.path / ".pytest-tmp"
    assert [report["marker"] for report in reports] == ["gw0", "gw1"]
    assert not invocation_root.exists()
