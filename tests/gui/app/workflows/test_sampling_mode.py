"""Registered synthetic workflow mementos retain Step expressions, Prepare resets."""

from pathlib import Path

from zcu_tools.cfg_model import ConfigBase, Range
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.sweep import SweepCfg

from tests.gui._user_input import ComponentDriver

from ._synthetic import Context, Plan, TunablesForm, sample_step


class SamplingPlanCfg(ConfigBase):
    sweep: SweepCfg


def test_workflow_step_expression_memento_and_prepare(qapp, tmp_path: Path, attachment):
    form = Form(SamplingPlanCfg, {"sweep": Range.between(0, 1, 5)})
    registration = WorkflowAttachment(
        step=sample_step, plan_model=form.model, knobs_model=TunablesForm,
        plan_to_cfg=lambda value, env, snapshot: Plan(snapshot=snapshot),
        knobs_to_cfg=TunablesForm.to_cfg, seed=lambda plan, value: Context(1),
    )
    _, harness, _ = attachment
    window = WorkflowWindow(harness.state, harness.services, harness.owner,
                            templates={}, alias_presets=None, registrations=(registration,))
    window.resize(1100, 900)
    window.show()
    qapp.processEvents()
    driver = ComponentDriver(window)
    try:
        driver.invoke_context_action(object_name="form.input.sweep.n", action="Switch to Step")
        driver.invoke_context_action(object_name="form.input.sweep.start", action="Switch to Center / Span")
        driver.invoke_context_action(object_name="form.input.sweep.step", action="Expression")
        driver.enter_text(object_name="form.input.sweep.step", value="0.1 * 3")
        driver.screenshot(tmp_path / "01-workflow-center-step.png")
        assert "0.45" in driver.observe(object_name="form.value.sweep.center").text
        saved = window.capture_persisted_state()
        driver.click("Prepare forms")
        driver = ComponentDriver(window.plan_panel)
        driver.screenshot(tmp_path / "02-workflow-prepare.png")
        assert driver.observe(object_name="form.input.sweep.n").text == "5"
        window.restore_persisted_state(saved)
        driver = ComponentDriver(window.plan_panel)
        driver.screenshot(tmp_path / "03-workflow-restored-step.png")
        assert driver.observe(object_name="form.input.sweep.step").text == "0.1 * 3"
        assert "0.45" in driver.observe(object_name="form.value.sweep.center").text
        assert window.capture_persisted_state() == saved
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
