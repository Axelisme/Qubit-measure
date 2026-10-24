"""Form-wide errors fit their text and leave spare height below the controls."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import field_validator
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget
from zcu_tools.gui.form import Display, FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


@pytest.fixture
def window(qapp: QApplication) -> Iterator[QWidget]:
    view = QWidget()
    yield view
    view.close()
    view.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize(
    "message",
    (
        "Calibration must be positive.",
        "Calibration must be positive. Select a valid calibration reference before "
        "starting acquisition. Check the reference value and its units, then refresh "
        "the form to confirm that the corrected calibration can be used safely.",
    ),
    ids=("two-lines", "wrapped-lines"),
)
def test_root_errors_fit_text_without_pushing_controls_down_on_taller_windows(
    window: QWidget, qapp: QApplication, tmp_path: Path, message: str
) -> None:
    class CalibrationForm(FormModel):
        sensor: Annotated[int, Display(group="Sensors")] = 1
        plain: int = 2
        calibration_a: Annotated[float, Display(hidden=True)] = 1.0
        calibration_b: Annotated[float, Display(hidden=True)] = 1.0

        @field_validator("calibration_a", "calibration_b")
        @classmethod
        def positive_calibration(cls, value: float) -> float:
            if value <= 0:
                raise ValueError(message)
            return value

    state = FormState(
        FormInput(CalibrationForm, {"calibration_a": -1.0, "calibration_b": -1.0})
    )
    layout = QVBoxLayout(window)
    layout.addWidget(FormEditor(state, FormEnv()))
    window.resize(800, 600)
    window.show()
    qapp.processEvents()
    driver = ComponentDriver(window)

    snapshots = []
    for step, width, height in (
        ("01-initial", 800, 600),
        ("02-taller", 800, 760),
        ("03-narrower", 400, 760),
    ):
        driver.resize_window(width, height)
        driver.screenshot(tmp_path / f"{step}.png")
        error = driver.observe(object_name="form.error")
        fold = driver.observe(object_name="form.fold.group.Sensors")
        assert error.visible and error.fully_visible and error.caption_fits
        assert error.text.count(message) == 2
        assert error.bounds is not None and fold.bounds is not None
        snapshots.append((error.bounds, fold.bounds))

    (initial, initial_fold), (taller, taller_fold), (narrower, _) = snapshots
    assert taller[3] == initial[3], snapshots
    assert taller_fold[1] - taller[1] == initial_fold[1] - initial[1], snapshots
    if message != "Calibration must be positive.":
        assert narrower[3] > taller[3], snapshots
