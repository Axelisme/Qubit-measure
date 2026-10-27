"""Exercise opt-in failure logging through real pytest sessions."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def checkout(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the real conftest in a synthetic checkout with mixed outcomes."""
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
    (tests / "test_outcomes.py").write_text(
        """
import pytest


@pytest.fixture
def broken_setup():
    raise RuntimeError('setup failure')


def test_call_failure():
    assert False, 'call failure'


def test_setup_failure(broken_setup):
    pass


def test_pass():
    pass
""",
        encoding="utf-8",
    )
    return tests


def test_failure_log_records_failed_phases(
    pytester: pytest.Pytester, checkout: Path
) -> None:
    log = pytester.path / "pytest-failures.jsonl"
    result = pytester.runpytest_inprocess("-q", "-n", "0", "--failure-log", str(log))

    result.assert_outcomes(failed=1, errors=1, passed=1)
    assert_failed_phases(log)


def assert_failed_phases(log: Path) -> None:
    """Check that the log contains exactly the two synthetic failures."""
    records = [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert sorted((record["nodeid"], record["when"]) for record in records) == [
        ("tests/test_outcomes.py::test_call_failure", "call"),
        ("tests/test_outcomes.py::test_setup_failure", "setup"),
    ]
    for record in records:
        assert record["outcome"] == "failed"
        assert record["traceback"].strip()
        assert datetime.fromisoformat(record["timestamp"]).utcoffset() is not None


def test_without_failure_log_creates_no_jsonl(
    pytester: pytest.Pytester, checkout: Path
) -> None:
    result = pytester.runpytest_inprocess("-q", "-n", "0")

    result.assert_outcomes(failed=1, errors=1, passed=1)
    assert not list(pytester.path.rglob("*.jsonl"))


@pytest.mark.requires_subprocess
def test_xdist_failure_log_records_each_failure_once(
    pytester: pytest.Pytester, checkout: Path
) -> None:
    log = pytester.path / "pytest-failures.jsonl"
    # Real workers verify that only the controller writes received reports.
    result = pytester.runpytest_subprocess("-q", "-n", "2", "--failure-log", str(log))

    result.assert_outcomes(failed=1, errors=1, passed=1)
    assert_failed_phases(log)
