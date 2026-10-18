"""Registered handwritten workflow forms share sweep commands and restored input."""

from pathlib import Path
from typing import Annotated

import pytest
from pydantic import Field
from zcu_tools.gui.app.workflows.persistence import WorkflowInputSnapshot
from zcu_tools.gui.form import (
    Display,
    FormModel,
)
from zcu_tools.gui.form_adapter import LinRange, SweepInput

from tests.gui._user_input import ComponentDriver


class SweepPlanForm(FormModel):
    sweep: Annotated[SweepInput, Display(selector=False)] = Field(
        default_factory=lambda: LinRange(start=10, stop=4, n=4)
    )


@pytest.fixture
def coordinate_window(sweep_window):
    return ComponentDriver(sweep_window(SweepPlanForm, None))


@pytest.fixture
def restored_coordinate_window(sweep_window):
    saved = WorkflowInputSnapshot(
        plan={
            "sweep": {"kind": "center", "center": {"expr": "3 + 4"}, "span": -6, "n": 4}
        }
    )
    return ComponentDriver(sweep_window(SweepPlanForm, saved))


def test_registered_workflow_sweep_commands_and_prepare(
    coordinate_window: ComponentDriver, tmp_path: Path
):
    driver = coordinate_window
    driver.screenshot(tmp_path / "01-workflow-range.png")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Center / Span"
    )
    driver.screenshot(tmp_path / "02-workflow-center.png")
    assert driver.observe(object_name="form.input.sweep.span").text == "-6.0"
    driver.invoke_context_action(
        object_name="form.input.sweep.center", action="Expression"
    )
    driver.enter_text(object_name="form.input.sweep.center", value="3 + 4")
    driver.screenshot(tmp_path / "03-workflow-expression.png")
    assert driver.observe(object_name="form.input.sweep.center").text == "3 + 4"
    driver.click("Prepare forms")
    driver.screenshot(tmp_path / "04-workflow-reset.png")
    start = next(
        item
        for item in driver.visible().widgets
        if item.object_name == "form.input.sweep.start"
    )
    assert float(start.text) == 10.0


def test_registered_workflow_restores_center_expression(
    restored_coordinate_window: ComponentDriver, tmp_path: Path
):
    driver = restored_coordinate_window
    driver.screenshot(tmp_path / "01-workflow-restored.png")
    assert driver.observe(object_name="form.input.sweep.center").text == "3 + 4"
    assert float(driver.observe(object_name="form.input.sweep.span").text) == -6.0
    assert driver.observe(object_name="form.input.sweep.n").text == "4"
