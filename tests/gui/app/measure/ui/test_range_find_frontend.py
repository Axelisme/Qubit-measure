"""Registered synthetic adapter driven only by visible keyboard/mouse actions.

GUI edge verification follows tests/README.md UI validation, not logic-seam TDD.
Set RANGE_FIND_SCREENSHOTS to retain per-step PNGs outside pytest tmp_path.
"""

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.analysis.dips import DipFinderOptions

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._range_find_adapter import SpectrumData, range_application


@pytest.fixture
def range_ui(qapp: QApplication, tmp_path: Path) -> Iterator[ComponentDriver]:
    with range_application(tmp_path, qapp) as app:
        yield ComponentDriver(app.window)


def _start(ui: ComponentDriver, shots: Path) -> None:
    ui.resize_window(1600, 950)
    ui.select_tab("Analysis")
    ui.click("Analyze")
    assert "west @ 5020 MHz" in ui.legends().texts
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


def _plot_point(ui: ComponentDriver, x: float, y: float) -> tuple[int, int]:
    axes = ui.plots().axes[0]
    left, top, width, height = axes.bounds
    return (
        round(left + width * (x - axes.xlim[0]) / (axes.xlim[1] - axes.xlim[0])),
        round(top + height * (axes.ylim[1] - y) / (axes.ylim[1] - axes.ylim[0])),
    )


def _wide_spectrum() -> SpectrumData:
    freqs = np.arange(3000, 8001, dtype=np.float64)
    magnitude = 2 + 0.00005 * (freqs - 3000)
    for freq in np.linspace(3500, 7800, 18):
        magnitude -= 0.1 * np.exp(-(((freqs - freq) / 2) ** 2))
    for freq in (7080, 7190):
        magnitude -= 1.5 * np.exp(-(((freqs - freq) / 2) ** 2))
    return SpectrumData(freqs, magnitude)


@pytest.mark.parametrize(("width", "height"), [(1275, 1539), (1100, 700)])
def test_operator_can_read_picker_and_zoom_then_home(
    qapp: QApplication, tmp_path: Path, width: int, height: int
):
    with range_application(
        tmp_path, qapp, data=_wide_spectrum(), components=("R1", "Resonator right")
    ) as app:
        ui = ComponentDriver(app.window)
        shots = _shots(tmp_path, f"layout-{width}x{height}")
        ui.resize_window(width, height)
        ui.select_tab("Analysis")
        ui.click("Analyze")
        ui.screenshot(shots / "01-initial.png")
        assert (ui.window_size().width(), ui.window_size().height()) == (width, height)
        canvas = ui.observe(object_name="rangeFind.spectrum")
        component = ui.observe(object_name="rangeFind.component")
        assert canvas.bounds is not None and component.bounds is not None
        assert (
            canvas.bounds[2] >= 500
            or component.bounds[1] >= canvas.bounds[1] + canvas.bounds[3]
        )
        assert all(
            b.canvas_left <= b.ylabel_left < b.ylabel_right <= b.canvas_right
            for b in ui.plot_label_bounds()
        )
        ui.select_combo("Unassigned", object_name="rangeFind.component")
        ui.screenshot(shots / "02-unassigned.png")
        assert ui.observe(object_name="rangeFind.component").caption_fits
        ui.select_combo("R1", object_name="rangeFind.component")
        ui.select_combo("7190 MHz", object_name="rangeFind.selection")
        ui.select_combo("Resonator right", object_name="rangeFind.component")
        assert ui.observe(object_name="rangeFind.component").caption_fits
        original = ui.plots().axes[0]
        selected = ui.observe(object_name="rangeFind.status").text
        ui.click_tooltip("Zoom to rectangle\nx/y fixes axis")
        ui.drag_at(_plot_point(ui, 7000, 2.3), _plot_point(ui, 7300, 0.64))
        ui.screenshot(shots / "03-zoom.png")
        assert ui.plots().axes[0].xlim == pytest.approx((7000, 7300), abs=15)
        assert ui.observe(object_name="rangeFind.status").text == selected
        ui.click_tooltip("Reset original view")
        ui.screenshot(shots / "04-home.png")
        assert ui.plots().axes[0].xlim == pytest.approx(original.xlim)
        ui.click_tooltip(
            "Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect"
        )
        bounds = ui.plots().axes[0].bounds
        start = (round(bounds[0] + bounds[2] / 2), round(bounds[1] + bounds[3] / 2))
        ui.drag_at(start, (start[0] + 30, start[1]))
        ui.screenshot(shots / "05-pan.png")
        assert ui.plots().axes[0].xlim != pytest.approx(original.xlim)
        assert ui.observe(object_name="rangeFind.status").text == selected
        ui.click_tooltip("Reset original view")
        ui.screenshot(shots / "06-home.png")
        assert ui.plots().axes[0].xlim == pytest.approx(original.xlim)


@pytest.mark.parametrize(("width", "height"), [(1275, 1539), (1100, 700)])
def test_operator_keeps_all_candidates_readable(
    qapp: QApplication, tmp_path: Path, width: int, height: int
):
    with range_application(
        tmp_path, qapp, data=_wide_spectrum(), components=("R1", "Resonator right")
    ) as app:
        ui = ComponentDriver(app.window)
        shots = _shots(tmp_path, f"legend-{width}x{height}")
        ui.resize_window(width, height)
        ui.select_tab("Analysis")
        ui.click("Analyze")
        ui.enter_text(value="0", object_name="rangeFind.max_dips")
        ui.screenshot(shots / "01-all-candidates.png")
        assert (
            ui.observe(object_name="rangeFind.status").text
            == "Selected: 20 / candidates: 20"
        )
        assert ui.legends().texts == (
            "Measured magnitude",
            "Candidates",
            "R1 @ 7080 MHz",
            "Resonator right @ 7190 MHz",
        )
        assert ui.plots().axes[0].annotations == ("R1", "Resonator right")
        ui.select_combo("7080 MHz", object_name="rangeFind.selection")
        ui.select_combo("Unassigned", object_name="rangeFind.component")
        ui.screenshot(shots / "02-unassigned.png")
        assert ui.legends().texts == (
            "Measured magnitude",
            "Candidates",
            "Resonator right @ 7190 MHz",
        )
        assert ui.plots().axes[0].annotations == ("Resonator right",)
        assert ui.observe(object_name="rangeFind.component").caption_fits


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
    assert ui.legends().texts == ("Measured magnitude", "Candidates")
    ui.screenshot(shots / "11-finished.png")


def test_operator_clicks_to_add_remove_and_assigns_out_of_order(
    range_ui: ComponentDriver, tmp_path: Path
):
    ui = range_ui
    shots = _shots(tmp_path, "selection")
    _start(ui, shots)
    ui.click_at(*_plot_point(ui, 5020, 0.3))
    assert "west" not in " ".join(ui.legends().texts)
    ui.screenshot(shots / "02-remove.png")
    ui.click_at(*_plot_point(ui, 5040, 0.8))
    assert ui.observe(object_name="rangeFind.selection").text == "5040 MHz"
    ui.screenshot(shots / "03-add.png")
    ui.select_combo("east", object_name="rangeFind.component")
    assert "east @ 5040 MHz" in ui.legends().texts
    assert ui.observe(object_name="rangeFind.component").text == "east"
    ui.screenshot(shots / "04-assign-east.png")
    ui.select_combo("5070 MHz", object_name="rangeFind.selection")
    ui.select_combo("west", object_name="rangeFind.component")
    assert "west @ 5070 MHz" in ui.legends().texts
    ui.screenshot(shots / "05-assign-west.png")
    ui.click("Done")
    assert "east @ 5040 MHz" in ui.legends().texts
    assert "west @ 5070 MHz" in ui.legends().texts
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
        ui.click_at(*_plot_point(ui, 5040, 0.8))
        assert ui.observe(object_name="rangeFind.selection").text == "5040 MHz"
        ui.screenshot(shots / "03-add.png")
        ui.select_combo("east", object_name="rangeFind.component")
        assert "east @ 5040 MHz" in ui.legends().texts
        ui.screenshot(shots / "04-assign-east.png")
        ui.select_combo("5070 MHz", object_name="rangeFind.selection")
        ui.select_combo("west", object_name="rangeFind.component")
        assert "west @ 5070 MHz" in ui.legends().texts
        ui.screenshot(shots / "05-assign-west.png")
        ui.click("Done")
        ui.screenshot(shots / "06-finished.png")
        assert "east @ 5040 MHz" in ui.legends().texts
        assert "west @ 5070 MHz" in ui.legends().texts
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
