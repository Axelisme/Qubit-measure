"""Form-local cfg diagnostics through real measure owners and visible inputs."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.events.run import RunStartedPayload
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import SampleAdapter
from tests.gui.widgets.form._text_input import commit_text


@pytest.fixture
def diagnostic_window(
    qapp: QApplication,
    tmp_path: Path,
) -> Iterator[tuple[Controller, MainWindow]]:
    registry = Registry()
    registry.register("sample", SampleAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
    )
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    ctrl.setup_control.new_entry(EntryRequest("diagnostics"))
    ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    window.show()
    qapp.processEvents()
    try:
        yield ctrl, window
    finally:
        ctrl.close()
        window.deleteLater()
        qapp.processEvents()


def open_tab(ctrl: Controller, window: MainWindow) -> ExpTabWidget:
    tab_id = ctrl.new_tab("sample")
    return next(
        tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == tab_id
    )


def reps_input(tab: ExpTabWidget) -> QLineEdit:
    return next(
        line
        for line in tab.findChildren(QLineEdit)
        if line.objectName() == "form.input.reps"
    )


def visible_status(window: MainWindow) -> str:
    status_bar = window.statusBar()
    assert status_bar is not None
    return "\n".join(
        label.text() for label in status_bar.findChildren(QLabel) if label.isVisible()
    )


def test_invalid_edit_disables_run_and_repair_restores_it(
    diagnostic_window: tuple[Controller, MainWindow],
) -> None:
    ctrl, window = diagnostic_window
    tab = open_tab(ctrl, window)
    assert tab.run_btn.isEnabled()
    ui = ComponentDriver(window)
    initial_status = ui.status().text
    ui.enter_text(object_name="form.input.reps", value="not-an-integer")
    assert not ui.observe("Run").enabled
    notice = ui.observe(object_name="run.cfg_error")
    assert notice.visible and "reps:" in notice.text
    assert ui.status().text == initial_status
    assert "Config invalid: reps:" in tab.run_btn.toolTip()
    started: list[RunStartedPayload] = []
    subscription = ctrl.get_bus().subscribe(RunStartedPayload, started.append)
    tab.run_btn.click()
    subscription.unsubscribe()
    assert started == []
    assert ctrl.get_running_tab_id() is None
    commit_text(reps_input(tab), "12")
    assert tab.run_btn.isEnabled()


def test_valid_refresh_preserves_unrelated_status(
    diagnostic_window: tuple[Controller, MainWindow],
) -> None:
    ctrl, window = diagnostic_window
    tab = open_tab(ctrl, window)
    commit_text(reps_input(tab), "not-an-integer")
    assert "Config invalid: reps:" in tab.run_btn.toolTip()
    window.show_status_message("Save completed")
    commit_text(reps_input(tab), "12")
    assert "Save completed" in visible_status(window)
    window.show_status_message("Device setup completed")
    commit_text(reps_input(tab), "13")
    assert "Device setup completed" in visible_status(window)


def test_cfg_changes_and_tab_switches_preserve_unrelated_status(
    diagnostic_window: tuple[Controller, MainWindow],
) -> None:
    ctrl, window = diagnostic_window
    first = open_tab(ctrl, window)
    second = open_tab(ctrl, window)
    window.show_status_message("Device setup completed")
    commit_text(reps_input(first), "not-an-integer")
    assert "Device setup completed" in visible_status(window)
    window.select_tab_widget(first.tab_id)
    assert "Config invalid: reps:" in first.run_btn.toolTip()
    window.select_tab_widget(second.tab_id)
    assert "Device setup completed" in visible_status(window)
    commit_text(reps_input(second), "also-invalid")
    assert "Config invalid: reps:" in second.run_btn.toolTip()
    commit_text(reps_input(first), "12")
    window.select_tab_widget(first.tab_id)
    assert "Device setup completed" in visible_status(window)
