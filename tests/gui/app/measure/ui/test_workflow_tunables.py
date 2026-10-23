"""Tunables draft commands and lifecycle presentation through visible input."""

from datetime import timedelta
from pathlib import Path
from threading import Event

import pytest
from zcu_tools.experiment.workflows import Live, WorkflowEnv, workflow
from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment

from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._workflow_window import (
    NativePlan,
    NativeTunables,
    Plan,
    Tunables,
)


@pytest.fixture
def stopping_window(workflow_window):
    ctrl, window = workflow_window
    release_cleanup = Event()

    @workflow("synthetic", "Synthetic retained cleanup")
    async def cleanup_wait(
        env: WorkflowEnv[None], plan: NativePlan, tun: Live[NativeTunables]
    ) -> None:
        try:
            await env.wait_until(env.started_at + timedelta(days=1))
        finally:
            # Keep stopping observable until fixture teardown releases author cleanup.
            release_cleanup.wait()

    ctrl.workflow_control().configure(
        (
            WorkflowAttachment(
                script=cleanup_wait,
                plan_model=Plan,
                knobs_model=Tunables,
                plan_to_cfg=lambda form, env, snapshot: NativePlan(cycles=form.cycles),
                knobs_to_cfg=lambda form: NativeTunables(gain=form.gain),
                seed=lambda plan, form: None,
            ),
        )
    )
    try:
        yield window
    finally:
        release_cleanup.set()


def _shot(driver: ComponentDriver, root: Path, step: str) -> None:
    driver.screenshot(root / f"{step}.png")


def _open(driver: ComponentDriver, evidence: Path) -> None:
    driver.select_menu("New Tab ▾", ("synthetic",))
    driver.screenshot(evidence)


def _status(driver: ComponentDriver, text: str) -> bool:
    return any(
        widget.object_name == "workflow.status" and widget.text == text
        for widget in driver.visible().widgets
    )


def test_stopping_locks_tunables_edits_and_submission(stopping_window, tmp_path):
    driver = ComponentDriver(stopping_window)
    driver.click(object_name="workflow.toggle")
    _open(driver, tmp_path / "stop-01-open.png")
    driver.click("Start")
    qt_until(lambda: _status(driver, "running"), label="cleanup scenario accepted")
    driver.select_tab("Tunables")
    driver.enter_text(value="0.5", object_name="form.input.gain")
    _shot(driver, tmp_path, "stop-02-running-editable")
    assert driver.observe(object_name="form.input.gain").text == "0.5"
    assert driver.observe("Apply").enabled
    assert driver.observe("Discard").enabled
    driver.click("Pause")
    qt_until(lambda: _status(driver, "paused"), label="cleanup scenario paused")
    driver.enter_text(value="0.75", object_name="form.input.gain")
    _shot(driver, tmp_path, "stop-03-paused-editable")
    assert driver.observe(object_name="form.input.gain").text == "0.75"
    driver.click("Stop")
    qt_until(lambda: _status(driver, "stopping"), label="author cleanup retained")
    _shot(driver, tmp_path, "stop-04-stopping")
    assert not driver.observe(object_name="workflow.pause_hint").visible
    driver.enter_text(value="0.9", object_name="form.input.gain")
    _shot(driver, tmp_path, "stop-05-edit-refused")
    assert driver.observe(object_name="form.input.gain").text == "0.75"
    assert not driver.observe("Apply").enabled
    assert not driver.observe("Discard").enabled


def test_pause_selects_tunables_and_clears_hint_on_resume_or_stop(
    workflow_window, tmp_path
):
    _, window = workflow_window
    driver = ComponentDriver(window)
    driver.click(object_name="workflow.toggle")
    _open(driver, tmp_path / "pause-01-open.png")
    driver.click("Start")
    qt_until(lambda: _status(driver, "running"), label="pause scenario accepted")
    _shot(driver, tmp_path, "pause-02-running")
    for action in ("Resume", "Stop"):
        driver.select_tab("Plan")
        _shot(driver, tmp_path, f"pause-03-plan-before-{action}")
        driver.click("Pause")
        qt_until(lambda: _status(driver, "paused"), label="native pause gate")
        _shot(driver, tmp_path, f"pause-04-paused-before-{action}")
        assert driver.tabs("Tunables").current == "Tunables"
        assert driver.observe(object_name="workflow.pause_hint").visible
        driver.select_tab("Plan")
        _shot(driver, tmp_path, f"pause-05-plan-while-paused-before-{action}")
        assert driver.tabs("Plan").current == "Plan"
        driver.select_tab("Tunables")
        _shot(driver, tmp_path, f"pause-06-tunables-before-{action}")
        driver.click(action)
        qt_until(
            lambda expected="running" if action == "Resume" else "cancelled": _status(
                driver, expected
            ),
            label=f"{action} delivered",
        )
        _shot(driver, tmp_path, f"pause-07-after-{action}")
        assert not driver.observe(object_name="workflow.pause_hint").visible


def test_discard_restores_latest_applied_tunables(workflow_window, tmp_path):
    _, window = workflow_window
    driver = ComponentDriver(window)
    driver.click(object_name="workflow.toggle")
    _open(driver, tmp_path / "01-open.png")
    driver.click("Start")
    qt_until(lambda: _status(driver, "running"), label="tunables run accepted")
    driver.select_tab("Tunables")
    _shot(driver, tmp_path, "02-clean")
    assert not driver.observe("Discard").enabled
    driver.enter_text(value="0.5", object_name="form.input.gain")
    _shot(driver, tmp_path, "03-edited")
    assert driver.observe("Discard").enabled
    driver.click("Apply")
    _shot(driver, tmp_path, "04-applied")
    assert driver.observe(object_name="workflow.message").text == "Applied revision 1"
    assert not driver.observe("Discard").enabled
    driver.enter_text(value="0.75", object_name="form.input.gain")
    _shot(driver, tmp_path, "05-draft")
    driver.click("Discard")
    _shot(driver, tmp_path, "06-discarded")
    assert driver.observe(object_name="form.input.gain").text == "0.5"
    assert not driver.observe("Discard").enabled
    driver.enter_text(value="not-a-number", object_name="form.input.gain")
    _shot(driver, tmp_path, "07-invalid-draft")
    assert driver.observe("Discard").enabled
    driver.click("Discard")
    _shot(driver, tmp_path, "08-invalid-discarded")
    assert driver.observe(object_name="form.input.gain").text == "0.5"
    assert not driver.observe("Discard").enabled
    driver.click("Stop")
    qt_until(lambda: _status(driver, "cancelled"), label="tunables run stopped")
    _shot(driver, tmp_path, "09-stopped")
