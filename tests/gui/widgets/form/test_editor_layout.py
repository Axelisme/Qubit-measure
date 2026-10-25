"""Form layout stays usable with long labels, selectors and mixed units."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.form import Display, FormEnv, FormState
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._form_converters import RebuildingForm
from tests.gui._observations import WidgetObservation
from tests.gui._user_input import ComponentDriver


class Endpoints(BaseModel):
    start: Annotated[float, Display(unit="MHz")] = 4000.0
    stop: Annotated[float, Display(unit="MHz")] = 4100.0
    n: Annotated[int, Display(label="Expts")] = 11


class Inline(BaseModel):
    kind: Literal["Inline"] = "Inline"
    frequency: Annotated[float, Display(unit="MHz")] = 4050.0
    gain: Annotated[float, Display(unit="normalized")] = 0.2
    waveform: Literal["Constant", "Gaussian"] = "Constant"


class Reference(BaseModel):
    kind: Literal["Library"] = "Library"


class LayoutForm(RebuildingForm):
    branch_models = {("readout",): (Inline, Reference)}

    repetitions: Annotated[
        int, Display(label="Acquisition repetitions (per software round)")
    ] = 1000
    early_stop: Annotated[
        float | None,
        Display(
            label="Early-stop SNR", description="Leave blank to disable early stop."
        ),
    ] = None
    readout: Annotated[Inline | Reference, Field(discriminator="kind")] = Field(
        default_factory=Inline
    )
    frequency: Annotated[
        Endpoints, Display(group="Sweep", unit="MHz", label="Frequency")
    ] = Field(default_factory=Endpoints)
    bias: Annotated[Endpoints, Display(group="Sweep", unit="MHz", label="Bias")] = (
        Field(default_factory=Endpoints)
    )


@pytest.fixture
def layout_driver(qapp: QApplication) -> Iterator[ComponentDriver]:
    env = FormEnv()
    state = FormState(LayoutForm.default_input(env))
    editor = FormEditor(state, env)
    editor.resize(800, 850)
    editor.show()
    qapp.processEvents()
    yield ComponentDriver(editor)
    editor.close()
    editor.deleteLater()
    qapp.processEvents()


def bounds(observed: WidgetObservation) -> tuple[int, int, int, int]:
    """Require a fully visible control and return its screen rectangle."""
    assert observed.fully_visible, observed
    assert observed.bounds is not None, observed
    return observed.bounds


def test_group_selector_uses_space_beside_its_own_title(
    layout_driver: ComponentDriver, tmp_path: Path
) -> None:
    driver = layout_driver
    driver.screenshot(tmp_path / "01-long-label.png")
    driver.click(object_name="form.fold.readout")
    driver.screenshot(tmp_path / "02-readout-expanded.png")
    title = bounds(driver.observe(object_name="form.fold.readout"))
    selector = bounds(driver.observe(object_name="form.input.readout"))
    # A section selector starts beside its short title, not the widest leaf label.
    assert selector[0] - (title[0] + title[2]) <= 12
    long_label_input = bounds(driver.observe(object_name="form.input.repetitions"))
    assert selector[0] < long_label_input[0]
    assert selector[2] > driver.window_size().width() / 2
    driver.choose_option(object_name="form.input.readout", value="Library")
    driver.screenshot(tmp_path / "03-reference.png")
    assert driver.observe(object_name="form.input.readout").text == "Library"
    driver.choose_option(object_name="form.input.readout", value="Inline")
    driver.screenshot(tmp_path / "04-inline.png")
    assert driver.observe(object_name="form.input.readout.waveform").text == "Constant"
    controls = [
        bounds(driver.observe(object_name=f"form.input.readout.{name}"))
        for name in ("frequency", "gain", "waveform")
    ]
    assert len({x + width for x, _, width, _ in controls}) == 1
    assert "Leave blank" in driver.observe(object_name="form.row.early_stop").tooltip


def test_value_rows_share_a_unit_column_but_section_titles_have_no_units(
    layout_driver: ComponentDriver, tmp_path: Path
) -> None:
    driver = layout_driver
    driver.screenshot(tmp_path / "01-sweep-units.png")
    # Units belong beside each value, never in the section header band.
    title = bounds(driver.observe(object_name="form.fold.frequency"))
    start = bounds(driver.observe(object_name="form.input.frequency.start"))
    # Initial rendering must not squash a nested input below a normal text row.
    normal_height = bounds(driver.observe(object_name="form.input.repetitions"))[3]
    assert start[3] >= normal_height
    units = [item for item in driver.visible().widgets if item.text == "MHz"]
    assert all(item.bounds is not None and item.bounds[1] > title[1] for item in units)
    assert len(units) == 4
    stop = bounds(driver.observe(object_name="form.input.frequency.stop"))
    count = bounds(driver.observe(object_name="form.input.frequency.n"))
    assert start[0] + start[2] == stop[0] + stop[2] == count[0] + count[2]
    driver.enter_text(object_name="form.input.frequency.n", value="21")
    driver.screenshot(tmp_path / "02-expts-edited.png")
    assert driver.observe(object_name="form.input.frequency.n").text == "21"
    driver.resize_window(1100, 850)
    driver.screenshot(tmp_path / "03-wide.png")
    rows = [
        bounds(driver.observe(object_name=f"form.input.frequency.{name}"))
        for name in ("start", "stop", "n")
    ]
    assert len({x + width for x, _, width, _ in rows}) == 1
