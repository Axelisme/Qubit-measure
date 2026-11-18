"""Form-local cfg diagnostics through real measure owners and visible inputs."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

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
from tests.program._profiles import load_soccfg


@pytest.fixture
def diagnostic_window(
    qapp: QApplication,
    tmp_path: Path,
) -> Iterator[tuple[Controller, MainWindow]]:
    registry = Registry()
    registry.register("sample", SampleAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=MagicMock(), soccfg=load_soccfg())),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
    )
    soccfg = load_soccfg()
    ctrl.session_services.soc_connection.install_soc(soccfg, soccfg)
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    ctrl.entry_control.new_entry(EntryRequest("diagnostics"))
    ctrl.entry_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    window.show()
    qapp.processEvents()
    try:
        yield ctrl, window
    finally:
        ctrl.close()
        window.dispose()
        qapp.processEvents()


def open_tab(ctrl: Controller, window: MainWindow) -> ExpTabWidget:
    tab_id = ctrl.tab_control.new_tab("sample")
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
    assert notice.visible and notice.text == "Config invalid: 1 field"
    assert ui.status().text == initial_status
    assert tab.run_btn.toolTip() == "Fix the fields marked in red"
    started: list[RunStartedPayload] = []
    subscription = ctrl.get_bus().subscribe(RunStartedPayload, started.append)
    tab.run_btn.click()
    subscription.unsubscribe()
    assert started == []
    assert ctrl.tab_control.get_running_tab_id() is None
    commit_text(reps_input(tab), "12")
    assert tab.run_btn.isEnabled()


def test_run_counts_local_invalid_paths_and_repairs_without_repeating_errors(
    diagnostic_window: tuple[Controller, MainWindow], tmp_path: Path
) -> None:
    ctrl, window = diagnostic_window
    open_tab(ctrl, window)
    ui = ComponentDriver(window)
    ui.resize_window(1500, 1100)
    initial_status = ui.status().text
    if ui.observe(object_name="form.fold.sweep").text.startswith("▸"):
        ui.click(object_name="form.fold.sweep")
    ui.screenshot(tmp_path / "00-valid.png")
    for path, value in (("sweep.start", "6000"), ("sweep.stop", "6100")):
        ui.enter_text(object_name=f"form.input.{path}", value=value)
    ui.screenshot(tmp_path / "01-filled.png")

    for index, path in enumerate(("sweep.start", "sweep.stop"), start=1):
        ui.enter_text(object_name=f"form.input.{path}", value="")
        ui.screenshot(tmp_path / f"02-invalid-{index}.png")
        suffix = "field" if index == 1 else "fields"
        assert ui.observe(object_name="run.cfg_error").text == (
            f"Config invalid: {index} {suffix}"
        )
        assert not ui.observe("Run").enabled
        assert ui.status().text == initial_status
    errors = [
        widget
        for widget in ui.visible().widgets
        if widget.object_name.startswith("form.error.")
    ]
    assert {widget.object_name for widget in errors} == {
        "form.error.sweep.start",
        "form.error.sweep.stop",
    }
    assert all(widget.text for widget in errors)
    assert not ui.observe(object_name="form.error.summary").visible
    assert ui.observe("Run").tooltip == "Fix the fields marked in red"
    assert all(widget.text not in ui.observe("Run").tooltip for widget in errors)

    ui.enter_text(object_name="form.input.sweep.start", value="6000")
    ui.screenshot(tmp_path / "03-one-left.png")
    assert ui.observe(object_name="run.cfg_error").text == "Config invalid: 1 field"
    assert not ui.observe(object_name="form.error.sweep.start").visible
    assert ui.observe(object_name="form.error.sweep.stop").visible
    assert not ui.observe("Run").enabled
    assert ui.status().text == initial_status

    ui.enter_text(object_name="form.input.sweep.stop", value="6100")
    ui.screenshot(tmp_path / "04-repaired.png")
    assert ui.observe(object_name="run.cfg_error").text == ""
    assert not ui.observe(object_name="run.cfg_error").visible
    assert not ui.observe(object_name="form.error.sweep.stop").visible
    assert ui.observe("Run").enabled
    assert ui.status().text == initial_status


def test_run_merges_resource_and_local_errors_by_canonical_path(
    diagnostic_window: tuple[Controller, MainWindow], tmp_path: Path
) -> None:
    ctrl, window = diagnostic_window
    open_tab(ctrl, window)
    ui = ComponentDriver(window)
    ui.resize_window(1500, 1100)
    if ui.observe(object_name="form.fold.sweep").text.startswith("▸"):
        ui.click(object_name="form.fold.sweep")
    ui.enter_text(object_name="form.input.sweep.n", value="0")
    ui.screenshot(tmp_path / "00-resource-error.png")
    assert ui.observe(object_name="run.cfg_error").text == "Config invalid: 1 field"
    assert ui.observe("Run").tooltip == "Fix the fields marked in red"

    # The resource still rejects n=0 while its replacement text is unparseable.
    ui.enter_text(object_name="form.input.sweep.n", value="-")
    ui.screenshot(tmp_path / "01-same-path-local-error.png")
    assert ui.observe(object_name="run.cfg_error").text == "Config invalid: 1 field"
    ui.enter_text(object_name="form.input.sweep.stop", value="")
    ui.screenshot(tmp_path / "02-resource-and-local-errors.png")
    assert ui.observe(object_name="run.cfg_error").text == "Config invalid: 2 fields"
    assert not ui.observe("Run").enabled


def test_valid_refresh_preserves_unrelated_status(
    diagnostic_window: tuple[Controller, MainWindow],
) -> None:
    ctrl, window = diagnostic_window
    tab = open_tab(ctrl, window)
    commit_text(reps_input(tab), "not-an-integer")
    assert tab.run_btn.toolTip() == "Fix the fields marked in red"
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
    assert first.run_btn.toolTip() == "Fix the fields marked in red"
    window.select_tab_widget(second.tab_id)
    assert "Device setup completed" in visible_status(window)
    commit_text(reps_input(second), "also-invalid")
    assert second.run_btn.toolTip() == "Fix the fields marked in red"
    commit_text(reps_input(first), "12")
    window.select_tab_widget(first.tab_id)
    assert "Device setup completed" in visible_status(window)
