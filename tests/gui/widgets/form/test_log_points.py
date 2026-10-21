"""Real input events switch logarithmic integer-list representations."""

from pathlib import Path
from typing import Annotated

from zcu_tools import cfg_model
from zcu_tools.gui.form import Display, FormEnv, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


class IntegerPlan(cfg_model.ConfigBase):
    depths: Annotated[list[int], Display(collapsed=False)]


def test_log_list_real_input_and_actual_point_count(qapp, tmp_path: Path):
    env = FormEnv()
    form = Form(IntegerPlan, {"depths": cfg_model.LogPoints(max=128, points=9)})
    editor = FormEditor(FormState(form.default_input(env)), env)
    editor.resize(800, 600)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)
    try:
        driver.screenshot(tmp_path / "01-log.png")
        driver.enter_text(object_name="form.input.depths.max", value="3")
        driver.screenshot(tmp_path / "02-deduplicated.png")
        assert (
            driver.observe(object_name="form.value.depths.points").text
            == "Effective: 4"
        )
        driver.choose_option(object_name="form.input.depths", value="List")
        driver.screenshot(tmp_path / "03-list.png")
        assert (
            driver.observe(object_name="form.input.depths.values").text
            == "[0, 1, 2, 3]"
        )
        driver.enter_text(object_name="form.input.depths.values", value="[0, 2, 7, 16]")
        driver.choose_option(object_name="form.input.depths", value="Log")
        driver.screenshot(tmp_path / "04-log-from-list.png")
        assert driver.observe(object_name="form.input.depths.max").text == "16"
        assert driver.observe(object_name="form.input.depths.points").text == "4"
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()
