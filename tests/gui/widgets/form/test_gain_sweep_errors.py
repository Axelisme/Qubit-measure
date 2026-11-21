"""A generated gain sweep shows bounds errors below the editable coordinate."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import pytest
from PyQt6.QtWidgets import QApplication
from zcu_tools.cfg_model import ConfigBase, FieldMeta, Range
from zcu_tools.gui.form import FormEnv, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.program.v2.sweep import GainSweepCfg

from tests.gui._user_input import ComponentDriver


class GainPlan(ConfigBase):
    gain: Annotated[GainSweepCfg, FieldMeta(label="Gain", unit="a.u.")]


@pytest.fixture
def gain_editor(qapp: QApplication) -> Iterator[ComponentDriver]:
    env = FormEnv()
    form = Form(GainPlan, {"gain": Range(0.0, 0.75, 4)})
    editor = FormEditor(FormState(form.default_input(env)), env)
    editor.resize(800, 420)
    editor.show()
    qapp.processEvents()
    try:
        driver = ComponentDriver(editor)
        driver.click(object_name="form.fold.gain")
        yield driver
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    ("actions", "edited", "invalid", "row", "valid"),
    [
        ((), "stop", "1.25", "stop", "0.75"),
        (("Switch to Center / Span",), "center", "1.1", "center", "0.375"),
        (("Switch to Step",), "stop", "1.3", "step", "0.75"),
        (
            ("Switch to Step", "Switch to Center / Span"),
            "center",
            "-0.9",
            "span",
            "0.375",
        ),
    ],
    ids=["range", "center", "range-step", "center-step"],
)
def test_gain_sweep_marks_only_the_visible_row_until_corrected(
    gain_editor: ComponentDriver,
    tmp_path: Path,
    actions: tuple[str, ...],
    edited: str,
    invalid: str,
    row: str,
    valid: str,
) -> None:
    driver = gain_editor
    driver.screenshot(tmp_path / "01-range-expts.png")
    for index, action in enumerate(actions, start=2):
        driver.invoke_context_action(object_name="form.fold.gain", action=action)
        driver.screenshot(tmp_path / f"0{index}-switched.png")

    driver.enter_text(object_name=f"form.input.gain.{edited}", value=invalid)
    driver.screenshot(tmp_path / "04-invalid.png")
    errors = [
        widget
        for widget in driver.visible().widgets
        if widget.object_name.startswith("form.error.gain")
    ]
    assert [error.object_name for error in errors] == [f"form.error.gain.{row}"]
    assert "[-1, 1]" in errors[0].text
    assert errors[0].fully_visible and errors[0].caption_fits
    assert not driver.observe(object_name="form.error.summary").visible

    driver.enter_text(object_name=f"form.input.gain.{edited}", value=valid)
    driver.screenshot(tmp_path / "05-corrected.png")
    assert not driver.observe(object_name=f"form.error.gain.{row}").visible
