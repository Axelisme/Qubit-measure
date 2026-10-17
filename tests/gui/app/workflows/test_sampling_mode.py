"""Registered synthetic workflow Step expressions, startup restore and Prepare."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConfigBase, Range
from zcu_tools.gui.app.workflows.persistence import WorkflowInputSnapshot
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.sweep import SweepCfg

from tests.gui._user_input import ComponentDriver


class SamplingPlanCfg(ConfigBase):
    sweep: SweepCfg


@pytest.fixture
def sampling_window(sweep_window):
    form = Form(SamplingPlanCfg, {"sweep": Range.between(0, 1, 5)})
    return ComponentDriver(sweep_window(form.model, None))


@pytest.fixture
def restored_sampling_window(sweep_window):
    form = Form(SamplingPlanCfg, {"sweep": Range.between(0, 1, 5)})
    saved = WorkflowInputSnapshot(
        plan={
            "sweep": {
                "kind": "center_step",
                "center": 0.5,
                "span": 1,
                "step": {"expr": "0.1 * 3"},
            }
        }
    )
    return ComponentDriver(sweep_window(form.model, saved))


def test_workflow_step_expression_and_prepare(
    sampling_window: ComponentDriver, tmp_path: Path
):
    driver = sampling_window
    driver.screenshot(tmp_path / "00-workflow-range-expts.png")
    driver.invoke_context_action(
        object_name="form.input.sweep.n", action="Switch to Step"
    )
    driver.screenshot(tmp_path / "01-workflow-range-step.png")
    driver.invoke_context_action(
        object_name="form.input.sweep.start", action="Switch to Center / Span"
    )
    driver.screenshot(tmp_path / "02-workflow-center-step.png")
    driver.invoke_context_action(
        object_name="form.input.sweep.step", action="Expression"
    )
    driver.enter_text(object_name="form.input.sweep.step", value="0.1 * 3")
    driver.screenshot(tmp_path / "03-workflow-step-expression.png")
    assert float(
        driver.observe(object_name="form.value.sweep.center").text.split(": ")[-1]
    ) == pytest.approx(0.45)
    driver.click("Prepare forms")
    driver.screenshot(tmp_path / "04-workflow-prepare.png")
    count = next(
        item
        for item in driver.visible().widgets
        if item.object_name == "form.input.sweep.n"
    )
    assert count.text == "5"


def test_workflow_restores_step_expression(
    restored_sampling_window: ComponentDriver, tmp_path: Path
):
    driver = restored_sampling_window
    driver.screenshot(tmp_path / "01-workflow-restored-step.png")
    assert driver.observe(object_name="form.input.sweep.step").text == "0.1 * 3"
    assert float(
        driver.observe(object_name="form.value.sweep.center").text.split(": ")[-1]
    ) == pytest.approx(0.45)
