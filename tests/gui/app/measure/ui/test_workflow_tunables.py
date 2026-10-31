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
    toggle_tab_family,
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
    toggle_tab_family(driver)
    _open(driver, tmp_path / "stop-01-open.png")
    driver.click("Start")
    qt_until(lambda: _status(driver, "running"), label="cleanup scenario accepted")
    driver.select_tab("Tunables")
    driver.enter_text(value="0.5", object_name="form.input.gain")
    _shot(driver, tmp_path, "stop-02-running-editable")
    assert driver.observe(object_name="form.input.gain").text == "0.5"
    assert driver.observe("Apply").enabled
    assert driver.observe("Discard").enabled
    driver.click("Stop")
    qt_until(lambda: _status(driver, "stopping"), label="author cleanup retained")
    _shot(driver, tmp_path, "stop-04-stopping")
    driver.enter_text(value="0.9", object_name="form.input.gain")
    _shot(driver, tmp_path, "stop-05-edit-refused")
    assert driver.observe(object_name="form.input.gain").text == "0.5"
    assert not driver.observe("Apply").enabled
    assert not driver.observe("Discard").enabled


def test_discard_restores_latest_applied_tunables(workflow_window, tmp_path):
    _, window = workflow_window
    driver = ComponentDriver(window)
    toggle_tab_family(driver)
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
