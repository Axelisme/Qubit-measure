"""Real field context-menu events exercise the shared sweep command projection."""

from pathlib import Path

from zcu_tools.cfg_model import ConfigBase, Range
from zcu_tools.gui.form import FormEnv, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.program.v2.sweep import SweepCfg

from tests.gui._user_input import ComponentDriver


class SweepPlan(ConfigBase):
    sweep: SweepCfg
    other: float = 1


def test_coordinate_context_menu_preserves_local_drafts(qapp, tmp_path: Path):
    env = FormEnv()
    form = Form(SweepPlan, {"sweep": Range.between(10, 4, 4)})
    state = FormState(form.default_input(env))
    editor = FormEditor(state, env)
    editor.resize(700, 600)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)
    try:
        driver.screenshot(tmp_path / "01-range.png")
        driver.enter_text(object_name="form.input.sweep.n", value="unfinished")
        driver.screenshot(tmp_path / "02-sampling-draft.png")
        driver.invoke_context_action(
            object_name="form.input.sweep.start", action="Switch to Center / Span"
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
            object_name="form.input.sweep.center", action="Switch to Range"
        )
        driver.screenshot(tmp_path / "05-rejected.png")
        assert driver.observe(object_name="form.input.sweep.span").text == "broken"
        assert driver.observe(object_name="form.input.sweep.center").text == "3 + 4"
        driver.enter_text(object_name="form.input.sweep.span", value="-6")
        driver.invoke_context_action(
            object_name="form.input.sweep.span", action="Switch to Range"
        )
        driver.screenshot(tmp_path / "06-range.png")
        assert driver.observe(object_name="form.input.sweep.start").text == "10.0"
        assert driver.observe(object_name="form.input.sweep.n").text == "unfinished"
        editor.set_editing_enabled(enabled=False)
        driver.invoke_context_action(
            object_name="form.input.sweep.start", action="Switch to Center / Span"
        )
        driver.screenshot(tmp_path / "07-readonly.png")
        assert driver.observe(object_name="form.input.sweep.start").text == "10.0"
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()
