"""Registered synthetic adapter driven only by visible keyboard/mouse actions.

GUI edge verification follows tests/README.md UI validation, not logic-seam TDD.
Set RANGE_FIND_SCREENSHOTS to retain per-step PNGs outside pytest tmp_path.
"""

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.analysis.dips import DipFinderOptions

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._range_find_adapter import range_application


@pytest.fixture
def range_ui(qapp: QApplication, tmp_path: Path) -> Iterator[ComponentDriver]:
    with range_application(tmp_path, qapp) as app:
        yield ComponentDriver(app.window)


def _start(ui: ComponentDriver, shots: Path) -> None:
    ui.resize_window(1600, 950)
    ui.select_tab("Analysis")
    ui.click("Analyze")
    assert "1: west @ 5020 MHz" in ui.legends().texts
    ui.screenshot(shots / "01-initial.png")


def _edit(ui: ComponentDriver, shots: Path, name: str, value: str, count: int) -> None:
    ui.enter_text(value=value, object_name=f"rangeFind.{name}")
    assert ui.observe(object_name="rangeFind.status").text.startswith(
        f"Selected: {count} /"
    )
    assert float(ui.observe(object_name=f"rangeFind.{name}").text) == float(value)
    ui.screenshot(shots)


def _shots(tmp_path: Path, scenario: str) -> Path:
    return Path(os.environ.get("RANGE_FIND_SCREENSHOTS", str(tmp_path))) / scenario


def test_operator_recomputes_all_five_controls(
    range_ui: ComponentDriver, tmp_path: Path
):
    ui = range_ui
    shots = _shots(tmp_path, "parameters")
    _start(ui, shots)
    _edit(ui, shots / "02-smoothing.png", "smoothing_sigma", "1", 2)
    _edit(ui, shots / "03-prominence.png", "prominence_factor", "100000000", 0)
    _edit(ui, shots / "04-prominence-reset.png", "prominence_factor", "5", 2)
    _edit(ui, shots / "05-spacing.png", "min_separation", "60", 1)
    _edit(ui, shots / "06-spacing-reset.png", "min_separation", "1", 2)
    _edit(ui, shots / "07-count.png", "max_dips", "1", 1)
    _edit(ui, shots / "08-count-reset.png", "max_dips", "2", 2)
    _edit(ui, shots / "09-width.png", "max_fwhm", "0.5", 0)
    assert ui.legends().texts == ("Measured magnitude",)
    _edit(ui, shots / "10-width-reset.png", "max_fwhm", "25", 2)
    ui.click("Done")
    assert "1: unassigned @ 5020 MHz" in ui.legends().texts
    ui.screenshot(shots / "11-finished.png")


def test_operator_clicks_to_add_remove_and_assigns_out_of_order(
    range_ui: ComponentDriver, tmp_path: Path
):
    ui = range_ui
    shots = _shots(tmp_path, "selection")
    _start(ui, shots)
    # Fixed 1600x950 viewport: click the visible 5020 MHz star, then 5040 MHz.
    ui.click_at(788, 775)
    assert "west" not in " ".join(ui.legends().texts)
    ui.screenshot(shots / "02-remove.png")
    ui.click_at(910, 400)
    assert "1: unassigned @ 5040 MHz" in ui.legends().texts
    ui.screenshot(shots / "03-add.png")
    ui.select_combo("east", object_name="rangeFind.component")
    assert "1: east @ 5040 MHz" in ui.legends().texts
    assert "2: unassigned @ 5070 MHz" in ui.legends().texts
    ui.screenshot(shots / "04-assign-east.png")
    ui.select_combo("5070 MHz", object_name="rangeFind.selection")
    ui.select_combo("west", object_name="rangeFind.component")
    assert "2: west @ 5070 MHz" in ui.legends().texts
    ui.screenshot(shots / "05-assign-west.png")
    ui.click("Done")
    assert "1: east @ 5040 MHz" in ui.legends().texts
    assert "2: west @ 5070 MHz" in ui.legends().texts
    ui.screenshot(shots / "06-finished.png")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("smoothing_sigma", 0.12345678901234566),
        ("prominence_factor", 5.123456789012345),
        ("max_fwhm", 25.123456789012344),
        ("min_separation", 1.2345678901234567),
    ],
)
def test_done_preserves_precise_controls_and_manual_selection(
    qapp: QApplication, tmp_path: Path, name: str, value: float
):
    options = replace(DipFinderOptions(max_dips=2), **{name: value})
    with range_application(tmp_path, qapp, options=options) as app:
        ui = ComponentDriver(app.window)
        shots = _shots(tmp_path, f"precision-{name}")
        _start(ui, shots)
        displayed_value = ui.observe(object_name=f"rangeFind.{name}").text
        assert ui.observe(object_name="rangeFind.max_dips").text == "2"
        ui.click("Remove selected dip")
        assert "west" not in " ".join(ui.legends().texts)
        ui.screenshot(shots / "02-remove.png")
        ui.click_at(910, 400)
        assert "1: unassigned @ 5040 MHz" in ui.legends().texts
        ui.screenshot(shots / "03-add.png")
        ui.select_combo("east", object_name="rangeFind.component")
        assert "1: east @ 5040 MHz" in ui.legends().texts
        ui.screenshot(shots / "04-assign-east.png")
        ui.select_combo("5070 MHz", object_name="rangeFind.selection")
        ui.select_combo("west", object_name="rangeFind.component")
        assert "2: west @ 5070 MHz" in ui.legends().texts
        ui.screenshot(shots / "05-assign-west.png")
        ui.click("Done")
        ui.screenshot(shots / "06-finished.png")
        assert "1: east @ 5040 MHz" in ui.legends().texts
        assert "2: west @ 5070 MHz" in ui.legends().texts
        assert float(displayed_value) == value


def test_invalid_controls_and_empty_selection_remain_editable_until_cancel(
    range_ui: ComponentDriver, tmp_path: Path
):
    ui = range_ui
    shots = _shots(tmp_path, "rejection")
    _start(ui, shots)
    ui.enter_text(value="-1", object_name="rangeFind.max_fwhm")
    ui.click("Done")
    assert ui.observe("Done").enabled
    assert "max_fwhm" in ui.observe(object_name="rangeFind.status").text
    ui.screenshot(shots / "02-invalid.png")
    ui.enter_text(value="25", object_name="rangeFind.max_fwhm")
    ui.click("Remove selected dip")
    ui.screenshot(shots / "03-remove-first.png")
    ui.click("Remove selected dip")
    ui.click("Done")
    assert "Select at least one" in ui.observe(object_name="rangeFind.status").text
    assert ui.observe("Done").enabled
    ui.screenshot(shots / "04-empty.png")
    ui.click("Cancel")
    assert ui.legends().texts == ()
    ui.screenshot(shots / "05-cancelled.png")
