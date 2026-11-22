"""Register failure logging before pytest discovers collection paths."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


class _FailureLog:
    """Append failed reports to a caller-owned path; propagate write failures."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Flush one JSONL record per failed test phase or xdist crash report."""
        if not report.failed:
            return
        record = {
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "traceback": report.longreprtext,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add opt-in JSONL failure logging to the suite's pytest CLI."""
    parser.addoption(
        "--failure-log",
        type=Path,
        metavar="PATH",
        help="Append failed test reports immediately to PATH as JSONL.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Enable logging only in the controller or serial session when requested.

    The log's parent must exist. Write errors propagate when a report fails.
    """
    failure_log = config.getoption("--failure-log")
    if failure_log is not None and not hasattr(config, "workerinput"):
        config.pluginmanager.register(_FailureLog(failure_log), "failure-log")
