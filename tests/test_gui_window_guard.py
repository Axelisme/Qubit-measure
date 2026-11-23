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
            if type(window).__name__
            in {"LeakedWindow", "DeleteOnlyDialog", "UnownedDialog", "UnownedRow"}
        ]
        self.visible[report.nodeid] = tuple(
            type(window).__name__ for window in windows if window.isVisible()
        )
        # These windows belong to the synthetic tests. The guard has already
        # reported failure and must leave them visible until this observation.
        for window in windows:
            window.close()
            window.deleteLater()


def test_control_factory_leaves_unowned_windows_for_guard(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only explicitly owned views may retire before the teardown report."""
    repo = Path(__file__).resolve().parents[1]
    gui = pytester.path / "gui"
    ui = gui / "ui"
    ui.mkdir(parents=True)
    shutil.copyfile(repo / "tests/gui/conftest.py", gui / "conftest.py")
    shutil.copyfile(repo / "tests/gui/app/measure/ui/conftest.py", ui / "conftest.py")
    pytester.makeini("[pytest]\naddopts = --import-mode=importlib\n")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    (ui / "test_unowned.py").write_text(
        """
import pytest
from qtpy.QtWidgets import QDialog, QWidget

windows = []

class UnownedDialog(QDialog):
    pass

class UnownedRow(QWidget):
    pass

@pytest.mark.parametrize('window_type', [UnownedDialog, UnownedRow])
def test_unowned_window(browser_host, window_type):
    window = window_type()
    window.setObjectName('guard.synthetic.unowned')
    windows.append(window)
    window.show()
    assert window.isVisible()
""",
        encoding="utf-8",
    )
    reports = TeardownReports()
    try:
        result = pytester.runpytest_inprocess("-q", plugins=[reports])
        result.assert_outcomes(passed=2, errors=2)
        for window_type in ("UnownedDialog", "UnownedRow"):
            nodeid = f"gui/ui/test_unowned.py::test_unowned_window[{window_type}]"
            report = reports.reports[nodeid]
            assert report.failed
            assert window_type in str(report.longrepr)
            assert nodeid in str(report.longrepr)
            assert reports.visible[nodeid] == (window_type,)
    finally:
        for window in QApplication.topLevelWidgets():
            if window.objectName() == "guard.synthetic.unowned":
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
from qtpy.QtWidgets import QDialog, QWidget

windows = []

class LeakedWindow(QWidget):
    pass

class DeleteOnlyDialog(QDialog):
    pass

@pytest.fixture
def owned_window(qapp):
    window = QWidget()
    window.setObjectName('guard.synthetic.owned')
    windows.append(window)
    window.show()
    yield window
    window.close()
    window.deleteLater()

def test_fixture_closes_after_call(owned_window):
    assert owned_window.isVisible()

def test_hidden_window(qapp):
    window = QWidget()
    window.setObjectName('guard.synthetic.hidden')
    windows.append(window)

def test_closed_window(qapp):
    window = QWidget()
    window.setObjectName('guard.synthetic.closed')
    windows.append(window)
    window.show()
    window.close()

@pytest.fixture
def delete_only_dialog(qapp):
    dialog = DeleteOnlyDialog()
    dialog.setObjectName('guard.synthetic.delete-only')
    windows.append(dialog)
    dialog.show()
    yield dialog
    dialog.deleteLater()

def test_delete_only_dialog(delete_only_dialog):
    assert delete_only_dialog.isVisible()

@pytest.fixture
def leaked_window(qapp, request):
    window = LeakedWindow()
    window.setObjectName('guard.synthetic.leaked')
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
        result.assert_outcomes(passed=6, errors=3)
        for name in (
            "test_fixture_closes_after_call",
            "test_hidden_window",
            "test_closed_window",
        ):
            assert reports.reports[f"gui/test_windows.py::{name}"].passed
        nodeid = "gui/test_windows.py::test_delete_only_dialog"
        dialog_report = reports.reports[nodeid]
        assert dialog_report.failed
        assert "DeleteOnlyDialog" in str(dialog_report.longrepr)
        assert nodeid in str(dialog_report.longrepr)
        assert reports.visible[nodeid] == ("DeleteOnlyDialog",)
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
            if window.objectName().startswith("guard.synthetic."):
                window.close()
                window.deleteLater()


def test_gui_qapp_rejects_non_qapplication_instance(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GUI qapp fast-fails instead of letting Qt abort on a bare QCoreApplication."""

    class BareCoreApp:
        pass

    repo = Path(__file__).resolve().parents[1]
    gui = pytester.path / "gui"
    gui.mkdir()
    shutil.copyfile(repo / "tests/gui/conftest.py", gui / "conftest.py")
    pytester.makeini("[pytest]\naddopts = --import-mode=importlib\n")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setattr(QApplication, "instance", staticmethod(BareCoreApp))
    (gui / "test_qapp_guard.py").write_text(
        """
def test_needs_qapp(qapp):
    assert qapp is not None
""",
        encoding="utf-8",
    )
    result = pytester.runpytest_inprocess("-q")
    result.assert_outcomes(errors=1)
    assert "GUI tests require a QApplication, got BareCoreApp" in result.stdout.str()
