"""Results browsing through visible controls over synthetic Database content."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.services.results_index import scan_results
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.control_identity import ControlIdentity

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure.ui._main_window_fakes import apply_window_defaults, mock_ctrl


@pytest.fixture
def browser_host(
    qapp: QApplication,
    tmp_path: Path,
    main_window_control_factory: Callable[[], ControlIdentity],
) -> Iterator[MainWindow]:
    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.query_results.return_value = scan_results(tmp_path)
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter())
    window.show()
    qapp.processEvents()
    yield window
    window.deleteLater()
    qapp.processEvents()


def test_menu_opens_independent_results_window(
    browser_host: MainWindow, tmp_path: Path,
) -> None:
    ui = ComponentDriver(browser_host)
    assert "Results ▾" in tuple(w.text for w in ui.visible().widgets)
    ui.select_menu("Results ▾", ("Results browser…",))
    assert ui.observe(object_name="results.browser").visible
    assert ui.observe(object_name="results.focus").text == "All results · 0 runs"
    ui.screenshot(tmp_path / "empty-browser.png")
