"""Real field context-menu events exercise the shared sweep command projection."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConfigBase, Range
from zcu_tools.gui.form import FormEnv, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.program.v2.sweep import SweepCfg

from tests.gui._user_input import ComponentDriver


class SweepPlan(ConfigBase):
    sweep: SweepCfg
    other: float = 1


@pytest.fixture
def coordinate_editor(qapp):
    env = FormEnv()
    form = Form(SweepPlan, {"sweep": Range.between(10, 4, 4)})
    state = FormState(form.default_input(env))
    editor = FormEditor(state, env)
    editor.resize(700, 600)
    editor.show()
    qapp.processEvents()
    try:
        yield editor, ComponentDriver(editor)
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()


@pytest.fixture
def readonly_coordinate_editor(coordinate_editor):
    editor, driver = coordinate_editor
    driver.enter_text(object_name="form.input.sweep.n", value="unfinished")
    editor.set_editing_enabled(enabled=False)
    return driver


def test_coordinate_context_menu_preserves_local_drafts(
    coordinate_editor, tmp_path: Path
):
    _, driver = coordinate_editor
    driver.screenshot(tmp_path / "01-range.png")
    driver.enter_text(object_name="form.input.sweep.n", value="unfinished")
    driver.screenshot(tmp_path / "02-sampling-draft.png")
    driver.invoke_context_action(
        object_name="form.fold.sweep",
        action="Switch to Center / Span",
        evidence=tmp_path / "03-row-menu.png",
    )
    driver.screenshot(tmp_path / "03-center.png")
    assert driver.observe(object_name="form.input.sweep.center").text == "7.0"
    assert driver.observe(object_name="form.input.sweep.span").text == "-6.0"
    assert driver.observe(object_name="form.input.sweep.n").text == "unfinished"
    driver.invoke_context_action(
        object_name="form.input.sweep.center", action="Expression"
    )
    driver.enter_text(object_name="form.input.sweep.center", value="3 + 4")
    driver.screenshot(tmp_path / "04-expression.png")
    driver.enter_text(object_name="form.input.sweep.span", value="broken")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Range"
    )
    driver.screenshot(tmp_path / "05-rejected.png")
    assert driver.observe(object_name="form.input.sweep.span").text == "broken"
    assert driver.observe(object_name="form.input.sweep.center").text == "3 + 4"
    driver.enter_text(object_name="form.input.sweep.span", value="-6")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Range"
    )
    driver.screenshot(tmp_path / "06-range.png")
    assert driver.observe(object_name="form.input.sweep.start").text == "10.0"
    assert driver.observe(object_name="form.input.sweep.n").text == "unfinished"


def test_readonly_coordinate_command_cannot_replace_draft(
    readonly_coordinate_editor: ComponentDriver, tmp_path: Path
):
    driver = readonly_coordinate_editor
    driver.screenshot(tmp_path / "01-readonly-range.png")
    driver.invoke_context_action(
        object_name="form.fold.sweep", action="Switch to Center / Span"
    )
    driver.screenshot(tmp_path / "02-readonly-command.png")
    assert float(driver.observe(object_name="form.input.sweep.start").text) == 10.0
    assert driver.observe(object_name="form.input.sweep.n").text == "unfinished"
