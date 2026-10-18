"""Real context menus expose all sampling/coordinate combinations and endpoints."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConfigBase, Range
from zcu_tools.gui.form import FormEnv, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.program.v2.sweep import SweepCfg

from tests.gui._user_input import ComponentDriver


class SamplingPlan(ConfigBase):
    sweep: SweepCfg
    other: float = 1


@pytest.fixture
def sampling_editor(qapp):
    env = FormEnv()
    form = Form(SamplingPlan, {"sweep": Range.between(0, 1, 5)})
    editor = FormEditor(FormState(form.default_input(env)), env)
    editor.resize(800, 600)
    editor.show()
    qapp.processEvents()
    try:
        yield editor, ComponentDriver(editor)
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()


def test_sampling_context_menu_and_visible_effective_coordinates(
    sampling_editor, tmp_path: Path
):
    _, driver = sampling_editor
    driver.screenshot(tmp_path / "01-range-expts.png")
    driver.invoke_context_action(object_name="form.fold.sweep", action="Switch to Step")
    driver.screenshot(tmp_path / "02-range-step.png")
    assert float(driver.observe(object_name="form.input.sweep.step").text) == 0.25
    driver.enter_text(object_name="form.input.sweep.step", value="0.3")
    driver.screenshot(tmp_path / "03-effective-stop.png")
    assert float(driver.observe(object_name="form.input.sweep.stop").text) == 1
    assert float(
        driver.observe(object_name="form.value.sweep.stop").text.split(": ")[-1]
    ) == pytest.approx(0.9)
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Center / Span"
    )
    driver.screenshot(tmp_path / "04-center-step.png")
    assert float(
        driver.observe(object_name="form.value.sweep.center").text.split(": ")[-1]
    ) == pytest.approx(0.45)
    assert float(
        driver.observe(object_name="form.value.sweep.span").text.split(": ")[-1]
    ) == pytest.approx(0.9)
    driver.invoke_context_action(
        object_name="form.input.sweep.step", action="Expression"
    )
    driver.enter_text(object_name="form.input.sweep.step", value="0.1 * 3")
    driver.enter_text(object_name="form.input.other", value="unfinished")
    driver.screenshot(tmp_path / "05-expression-unrelated-error.png")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Expts"
    )
    driver.screenshot(tmp_path / "06-center-expts.png")
    assert float(
        driver.observe(object_name="form.input.sweep.center").text
    ) == pytest.approx(0.45)
    assert driver.observe(object_name="form.input.sweep.n").text == "4"
    assert driver.observe(object_name="form.input.other").text == "unfinished"
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Range"
    )
    driver.enter_text(object_name="form.input.sweep.start", value="1")
    driver.enter_text(object_name="form.input.sweep.stop", value="0")
    driver.invoke_context_action(object_name="form.fold.sweep", action="Switch to Step")
    driver.enter_text(object_name="form.input.sweep.step", value="-0.3")
    driver.screenshot(tmp_path / "07-descending.png")
    assert float(
        driver.observe(object_name="form.value.sweep.stop").text.split(": ")[-1]
    ) == pytest.approx(0.1)
    driver.invoke_context_action(
        object_name="form.input.sweep.step", action="Expression"
    )
    driver.enter_text(object_name="form.input.sweep.step", value="1 / 0")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Expts"
    )
    driver.screenshot(tmp_path / "08-rejected-expression.png")
    assert driver.observe(object_name="form.input.sweep.step").text == "1 / 0"
    driver.enter_text(object_name="form.input.sweep.step", value="unfinished")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Expts"
    )
    driver.screenshot(tmp_path / "09-rejected-local-text.png")
    assert driver.observe(object_name="form.input.sweep.step").text == "unfinished"


@pytest.fixture
def readonly_valid_sampling_editor(sampling_editor):
    editor, driver = sampling_editor
    driver.invoke_context_action(object_name="form.fold.sweep", action="Switch to Step")
    editor.set_editing_enabled(enabled=False)
    return driver


def test_readonly_sampling_command_cannot_convert_valid_step(
    readonly_valid_sampling_editor: ComponentDriver, tmp_path: Path
):
    driver = readonly_valid_sampling_editor
    driver.screenshot(tmp_path / "01-readonly-valid-step.png")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Expts"
    )
    driver.screenshot(tmp_path / "02-readonly-command.png")
    assert float(driver.observe(object_name="form.input.sweep.step").text) == 0.25


@pytest.fixture
def readonly_sampling_editor(sampling_editor):
    editor, driver = sampling_editor
    driver.invoke_context_action(object_name="form.fold.sweep", action="Switch to Step")
    driver.enter_text(object_name="form.input.sweep.step", value="unfinished")
    editor.set_editing_enabled(enabled=False)
    return driver


def test_readonly_sampling_command_cannot_replace_local_text(
    readonly_sampling_editor: ComponentDriver, tmp_path: Path
):
    driver = readonly_sampling_editor
    driver.screenshot(tmp_path / "01-readonly-step.png")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Expts"
    )
    driver.screenshot(tmp_path / "02-readonly-command.png")
    assert driver.observe(object_name="form.input.sweep.step").text == "unfinished"
