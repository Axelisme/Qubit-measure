"""Workflow hosting through caller registrations and visible mouse/keyboard input."""

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import Done, InitEnv, Step, WorkflowEnv, workflow
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import SampleAdapter


class Plan(FormModel):
    cycles: int = 1


class Tunables(FormModel):
    gain: float = 0.25


class NativePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cycles: int


class NativeTunables(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    gain: float


def _init(env: InitEnv[None], plan: NativePlan) -> int:
    return 0


@workflow("synthetic", plan=NativePlan, tunables=NativeTunables, init=_init)
def _waiting(
    env: WorkflowEnv[None], plan: NativePlan, tun: NativeTunables, state: int
) -> Step[int, int]:
    # Public wait effect lets Pause and Stop end the segment without elapsed-time success.
    yield from env.wait_until(env.started_at + timedelta(days=1))
    return Done()


@pytest.fixture
def workflow_window(
    qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[Controller, MainWindow]]:
    registry = Registry()
    registry.register("experiment", SampleAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
        progress_transport=DirectProgressTransport(),
    )
    ctrl.workflow_control().configure(
        (
            WorkflowAttachment(
                step=_waiting,
                plan_model=Plan,
                knobs_model=Tunables,
                plan_to_cfg=lambda form, env, snapshot: NativePlan(cycles=form.cycles),
                knobs_to_cfg=lambda form: NativeTunables(gain=form.gain),
                seed=lambda plan, form: None,
            ),
        )
    )
    ctrl.setup_control.new_entry(EntryRequest("workflow-ui"))
    ctrl.setup_control.new_point(PointCreateRequest("source", "", CommandOrigin("gui")))
    board = white_noise()
    ctrl.install_soc(board.soc, board.soccfg)
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    window.show()
    qapp.processEvents()
    try:
        yield ctrl, window
    finally:
        ctrl.close()
        window.deleteLater()
        qapp.processEvents()


def _ready(driver: ComponentDriver) -> None:
    assert any(w.object_name == "workflow.toggle" for w in driver.visible().widgets), (
        "MainWindow must expose the single Experiment/Workflow toggle"
    )


def _shot(driver: ComponentDriver, root: Path, step: str) -> None:
    driver.screenshot(root / f"{step}.png")


def _text(driver: ComponentDriver, name: str) -> str:
    return next(w.text for w in driver.visible().widgets if w.object_name == name)


def _open(driver: ComponentDriver) -> None:
    driver.select_menu("New Tab ▾", ("synthetic",))


def _status(driver: ComponentDriver, text: str) -> bool:
    return any(
        w.object_name == "workflow.status" and w.text == text
        for w in driver.visible().widgets
    )


def test_toggle_same_name_tabs_forms_and_idle_close(workflow_window, tmp_path):
    _, window = workflow_window
    driver = ComponentDriver(window)
    _ready(driver)
    _shot(driver, tmp_path, "01-experiments")
    driver.click(object_name="workflow.toggle")
    _shot(driver, tmp_path, "02-empty-workflows")
    _open(driver)
    _shot(driver, tmp_path, "03-first-tab")
    assert driver.tabs("synthetic (1)").current == "synthetic (1)"
    assert driver.tabs("Plan").captions == ("Plan", "Tunables", "Ask")
    driver.enter_text("Cycles", "7")
    _shot(driver, tmp_path, "04-plan-edited")
    _open(driver)
    _shot(driver, tmp_path, "05-second-tab")
    assert driver.tabs("synthetic (2)").captions == ("synthetic (1)", "synthetic (2)")
    assert _text(driver, "form.input.cycles") == "1"
    driver.select_tab("synthetic (1)")
    _shot(driver, tmp_path, "06-independent-plan")
    assert _text(driver, "form.input.cycles") == "7"
    driver.select_tab("Tunables")
    _shot(driver, tmp_path, "07-tunables")
    assert _text(driver, "form.input.gain") == "0.25"
    driver.select_tab("Ask")
    _shot(driver, tmp_path, "08-ask")
    assert driver.observe(object_name="workflow.ask").text == "No question pending."
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "09-one-closed")
    assert driver.tabs("synthetic (2)").captions == ("synthetic (2)",)
    driver.click_tooltip("Close synthetic (2)")
    _shot(driver, tmp_path, "10-last-closed")
    _open(driver)
    _shot(driver, tmp_path, "11-reopened")
    assert driver.observe("Start").enabled
    driver.click(object_name="workflow.toggle")
    _shot(driver, tmp_path, "12-experiments-again")
    assert driver.observe(object_name="empty_tabs_hint").visible


def test_running_paused_close_lock_and_badge_navigation(workflow_window, tmp_path):
    _, window = workflow_window
    driver = ComponentDriver(window)
    _ready(driver)
    driver.select_menu("New Tab ▾", ("experiment",))
    _shot(driver, tmp_path, "01-experiment")
    driver.click(object_name="workflow.toggle")
    _open(driver)
    _open(driver)
    driver.select_tab("synthetic (1)")
    driver.click("Start")
    qt_until(lambda: _status(driver, "running"), label="workflow accepted")
    _shot(driver, tmp_path, "02-running")
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "03-running-close-refused")
    assert driver.tabs("synthetic (1)").captions == ("synthetic (1)", "synthetic (2)")
    driver.select_tab("synthetic (2)")
    _shot(driver, tmp_path, "04-other-locked")
    assert not driver.observe("Start").enabled
    driver.click(object_name="workflow.toggle")
    _shot(driver, tmp_path, "05-experiment-locked")
    assert not driver.observe("Run").enabled
    driver.click(object_name="workflow.badge")
    _shot(driver, tmp_path, "06-badge-navigation")
    assert driver.tabs("synthetic (1)").current == "synthetic (1)"
    driver.click("Pause")
    qt_until(lambda: _status(driver, "paused"), label="paused segment delivered")
    _shot(driver, tmp_path, "07-paused")
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "08-paused-close-refused")
    assert driver.tabs("synthetic (1)").captions == ("synthetic (1)", "synthetic (2)")
    driver.select_tab("Tunables")
    driver.enter_text("Gain", "0.5")
    driver.click("Apply")
    _shot(driver, tmp_path, "09-tunables-applied")
    assert driver.observe(object_name="workflow.message").text == "Applied revision 1"
    driver.click("Resume")
    qt_until(lambda: _status(driver, "running"), label="resumed")
    _shot(driver, tmp_path, "10-resumed")
    driver.click("Stop")
    qt_until(lambda: _status(driver, "stopped"), label="stop delivered")
    _shot(driver, tmp_path, "11-stopped")
    driver.click_tooltip("Close synthetic (1)")
    _shot(driver, tmp_path, "12-closed-after-stop")
    assert driver.tabs("synthetic (2)").captions == ("synthetic (2)",)
    assert driver.observe("Start").enabled


def test_external_facet_open_is_projected_without_gui_selection(
    workflow_window, tmp_path
):
    ctrl, window = workflow_window
    driver = ComponentDriver(window)
    _ready(driver)
    # This is an external-caller seam test, not a replacement for the driven scenarios above.
    ctrl.workflow_control().open("synthetic")
    _shot(driver, tmp_path, "01-external-open")
    assert driver.tabs("synthetic (1)").current == "synthetic (1)"
    ctrl.workflow_control().open("synthetic")
    _shot(driver, tmp_path, "02-external-second-open")
    assert driver.tabs("synthetic (2)").current == "synthetic (2)"
    driver.click_tooltip("Close synthetic (2)")
    _shot(driver, tmp_path, "03-external-tab-closed-locally")
    assert driver.tabs("synthetic (1)").captions == ("synthetic (1)",)
