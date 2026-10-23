"""Driver observations of disabled drafts and membership-driven recovery."""

from collections.abc import Mapping
from pathlib import Path

from pydantic import Field, JsonValue
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.form import FieldApplicability, FormEnv, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


class Conditional(FormModel):
    """Keep an invalid amplitude draft while another selection excludes it."""

    selection: list[int] = [0]
    amplitude: float | None = Field(default=-1, gt=0)
    readout: float = 6000

    @classmethod
    def applicability(cls, data: Mapping[str, JsonValue]) -> tuple[FieldApplicability, ...]:
        """Only selection membership controls amplitude applicability."""
        return (FieldApplicability(("selection",), 1, (("amplitude",),)),)


def test_membership_greys_draft_and_restores_its_value_and_error(qapp: QApplication, tmp_path: Path) -> None:
    editor = FormEditor(FormState(Conditional.default_input(FormEnv())), FormEnv())
    editor.resize(650, 360)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)
    try:
        driver.screenshot(tmp_path / "01-inapplicable.png")
        assert not driver.observe(object_name="form.input.amplitude").enabled
        assert driver.observe(object_name="form.input.readout").enabled
        driver.enter_text(object_name="form.input.selection", value="[1]")
        driver.screenshot(tmp_path / "02-applicable-invalid.png")
        assert driver.observe(object_name="form.input.amplitude").enabled
        assert driver.observe(object_name="form.input.amplitude").text == "-1"
        assert "greater than 0" in driver.observe(object_name="form.row.amplitude").tooltip
        driver.enter_text(object_name="form.input.amplitude", value="3")
        driver.enter_text(object_name="form.input.selection", value="[0]")
        driver.screenshot(tmp_path / "03-disabled-valid-draft.png")
        assert not driver.observe(object_name="form.input.amplitude").enabled
        assert driver.observe(object_name="form.input.amplitude").text == "3"
        driver.enter_text(object_name="form.input.selection", value="[1]")
        driver.screenshot(tmp_path / "04-restored-valid.png")
        assert driver.observe(object_name="form.input.amplitude").enabled
        assert driver.observe(object_name="form.input.amplitude").text == "3"
        assert "greater than 0" not in driver.observe(object_name="form.row.amplitude").tooltip
    finally:
        editor.close()
