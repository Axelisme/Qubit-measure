"""Registered workflow forms share sweep commands and persisted active input."""

from pathlib import Path

from zcu_tools.cfg_model import ConfigBase, Range
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.sweep import SweepCfg

from tests.gui._user_input import ComponentDriver

from ._start_fakes import StartHarness
from ._synthetic import Context, Plan, TunablesForm, sample_step


class SweepPlanCfg(ConfigBase):
    sweep: SweepCfg


def test_registered_workflow_sweep_restores_center_expression(qapp, tmp_path: Path):
    form = Form(SweepPlanCfg, {"sweep": Range.between(10, 4, 4)})
    registration = WorkflowAttachment(
        step=sample_step,
        plan_model=form.model,
        knobs_model=TunablesForm,
        plan_to_cfg=lambda value, env, snapshot: Plan(snapshot=snapshot),
        knobs_to_cfg=TunablesForm.to_cfg,
        seed=lambda plan, value: Context(1),
    )
    harness = StartHarness()
    window = WorkflowWindow(
        harness.state,
        harness.services,
        harness.owner,
        templates={},
        alias_presets=None,
        registrations=(registration,),
    )
    window.resize(1100, 900)
    window.show()
    qapp.processEvents()
    driver = ComponentDriver(window)
    try:
        driver.screenshot(tmp_path / "01-workflow-range.png")
        driver.invoke_context_action(
            object_name="form.input.sweep.stop", action="Switch to Center / Span"
        )
        driver.screenshot(tmp_path / "02-workflow-center.png")
        assert driver.observe(object_name="form.input.sweep.span").text == "-6.0"
        driver.invoke_context_action(
            object_name="form.input.sweep.center", action="Expression"
        )
        driver.enter_text(object_name="form.input.sweep.center", value="3 + 4")
        driver.screenshot(tmp_path / "03-workflow-expression.png")
        saved = window.capture_persisted_state()
        window.prepare_forms()
        driver.screenshot(tmp_path / "04-workflow-reset.png")
        assert driver.observe(object_name="form.input.sweep.start").text == "10.0"
        window.restore_persisted_state(saved)
        driver.screenshot(tmp_path / "05-workflow-restored.png")
        assert driver.observe(object_name="form.input.sweep.center").text == "3 + 4"
        assert driver.observe(object_name="form.input.sweep.n").text == "4"
        assert window.capture_persisted_state() == saved
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
