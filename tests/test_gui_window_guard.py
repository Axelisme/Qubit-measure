"""Verify GUI resource ownership through pytest's public test lifecycle."""

import shutil
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication

pytest_plugins = ["pytester"]


class TeardownReports:
    """Observe reports, then isolate deliberately leaked synthetic windows."""

    def __init__(self) -> None:
        self.reports: dict[str, pytest.TestReport] = {}
        self.visible: dict[str, tuple[str, ...]] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "teardown":
            return
        self.reports[report.nodeid] = report
        windows = [
            window
            for window in QApplication.topLevelWidgets()
            if type(window).__name__ == "LeakedWindow"
        ]
        self.visible[report.nodeid] = tuple(
            type(window).__name__ for window in windows if window.isVisible()
        )
        # These windows belong to the synthetic tests. The guard has already
        # reported failure and must leave them visible until this observation.
        for window in windows:
            window.close()
            window.deleteLater()


def test_gui_window_guard_runs_after_fixture_cleanup(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = Path(__file__).resolve().parents[1]
    gui = pytester.path / "gui"
    gui.mkdir()
    shutil.copyfile(repo / "tests/gui/conftest.py", gui / "conftest.py")
    pytester.makeini("[pytest]\naddopts = --import-mode=importlib\n")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    (gui / "test_windows.py").write_text(
        """
import pytest
from qtpy.QtWidgets import QWidget

windows = []

class LeakedWindow(QWidget):
    pass

@pytest.fixture
def owned_window(qapp):
    window = QWidget()
    windows.append(window)
    window.show()
    yield window
    window.close()
    window.deleteLater()

def test_fixture_closes_after_call(owned_window):
    assert owned_window.isVisible()

def test_hidden_window(qapp):
    windows.append(QWidget())

def test_closed_window(qapp):
    window = QWidget()
    windows.append(window)
    window.show()
    window.close()

@pytest.fixture
def leaked_window(qapp, request):
    window = LeakedWindow()
    windows.append(window)
    window.show()
    yield window
    if request.param:
        raise RuntimeError('original fixture teardown error')

@pytest.mark.parametrize('leaked_window', [False, True], indirect=True)
def test_leaked_window(leaked_window):
    assert leaked_window.isVisible()
""",
        encoding="utf-8",
    )
    reports = TeardownReports()
    try:
        result = pytester.runpytest_inprocess("-q", plugins=[reports])
        result.assert_outcomes(passed=5, errors=2)
        for name in (
            "test_fixture_closes_after_call",
            "test_hidden_window",
            "test_closed_window",
        ):
            assert reports.reports[f"gui/test_windows.py::{name}"].passed
        for parameter in ("False", "True"):
            nodeid = f"gui/test_windows.py::test_leaked_window[{parameter}]"
            leaked = reports.reports[nodeid]
            assert leaked.failed
            assert "LeakedWindow" in str(leaked.longrepr)
            assert nodeid in str(leaked.longrepr)
            assert reports.visible[nodeid] == ("LeakedWindow",)
        assert "original fixture teardown error" in str(
            reports.reports["gui/test_windows.py::test_leaked_window[True]"].longrepr
        )
    finally:
        # The synthetic owner isolates its deliberate leaks, not the guard.
        for window in QApplication.topLevelWidgets():
            window.close()
            window.deleteLater()
