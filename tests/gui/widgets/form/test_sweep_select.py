"""The ordinary selector shows converted sweep inputs and fallback blanks."""

from pathlib import Path
from typing import Annotated

import pytest
from zcu_tools.gui.form import Display, FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.form_adapter import SweepInput
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


class SweepSelector(FormModel):
    sweep: Annotated[SweepInput, Display(selector=True)]


@pytest.fixture
def sweep_selector(qapp):
    env = FormEnv()
    state = FormState(
        FormInput(
            SweepSelector,
            {"sweep": {"kind": "linrange", "start": 10, "stop": 14, "n": 3}},
        )
    )
    editor = FormEditor(state, env)
    editor.resize(720, 520)
    editor.show()
    qapp.processEvents()
    try:
        yield ComponentDriver(editor)
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()


def test_select_displays_converted_values_and_expression_fallback(
    sweep_selector, tmp_path: Path
):
    driver = sweep_selector
    driver.click(object_name="form.fold.sweep")
    driver.screenshot(tmp_path / "01-range.png")
    driver.choose_option(object_name="form.input.sweep", value="center_step")
    driver.screenshot(tmp_path / "02-center-step.png")
    assert driver.observe(object_name="form.input.sweep.center").text == "12.0"
    assert driver.observe(object_name="form.input.sweep.span").text == "4.0"
    assert driver.observe(object_name="form.input.sweep.step").text == "2.0"
    driver.choose_option(object_name="form.input.sweep", value="linrange")
    driver.screenshot(tmp_path / "03-range-restored.png")
    assert driver.observe(object_name="form.input.sweep.start").text == "10.0"
    assert driver.observe(object_name="form.input.sweep.stop").text == "14.0"
    assert driver.observe(object_name="form.input.sweep.n").text == "3"
    driver.invoke_context_action(
        object_name="form.input.sweep.start", action="Expression"
    )
    driver.screenshot(tmp_path / "04-expression-mode.png")
    driver.enter_text(object_name="form.input.sweep.start", value="5 + 5")
    driver.screenshot(tmp_path / "05-expression.png")
    driver.choose_option(object_name="form.input.sweep", value="center")
    driver.screenshot(tmp_path / "06-inherited-sampling.png")
    assert driver.observe(object_name="form.input.sweep.center").text == ""
    assert driver.observe(object_name="form.input.sweep.span").text == ""
    assert driver.observe(object_name="form.input.sweep.n").text == "3"
