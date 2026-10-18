"""A selector without a converter keeps the visible authored child draft."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import Field
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


class First(FormModel):
    kind: Literal["first"] = "first"
    amount: float = 2


class Second(FormModel):
    kind: Literal["second"] = "second"
    required: float


class SelectionForm(FormModel):
    branch: First | Second = Field(discriminator="kind")


def test_selection_only_keeps_visible_child_and_local_parse_error(
    qapp: QApplication, tmp_path: Path
):
    editor = FormEditor(
        FormState(FormInput(SelectionForm, {"branch": {"kind": "first", "amount": 2}})),
        FormEnv(),
    )
    editor.resize(620, 300)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)

    def snapshot(name: str) -> None:
        driver.screenshot(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.json").write_text(
            json.dumps(asdict(driver.visible()), indent=2), encoding="utf-8"
        )

    try:
        snapshot("01-first")
        driver.enter_text(object_name="form.input.branch.amount", value="broken")
        snapshot("02-local-error")
        driver.choose_option(object_name="form.input.branch", value="second")
        snapshot("03-new-selection-old-branch")
        assert driver.observe(object_name="form.input.branch").text == "second"
        assert driver.observe(object_name="form.input.branch.amount").text == "broken"
    finally:
        editor.close()
