"""Visible predictor editing and curve interactions with synthetic query results."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pytest
from qtpy.QtCore import QTimer
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog
from zcu_tools.resources.predictor_file import PredictorFlux
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.session.ui._predictor_support import SyntheticCurves


@pytest.fixture
def predictor_ui(qapp, monkeypatch: pytest.MonkeyPatch, request):
    # Qt scheduling is the external clock seam; completion is still visible state.
    original = QTimer.setInterval

    def immediate_debounce(timer: QTimer, milliseconds: int) -> None:
        original(timer, 0 if milliseconds == 150 else milliseconds)

    monkeypatch.setattr(QTimer, "setInterval", immediate_debounce)
    dialog = PredictorDialog(
        PredictorControlFacet(SyntheticCurves()),
        params=FluxoniumParams(4.0, 1.0, 1.0),
    )
    dialog.set_flux(PredictorFlux(0.0, {"coil": 1.0}, {"coil": "A"}), {"coil": 0.0})
    if getattr(request, "param", False):
        dialog.set_live_mode(enabled=True)
    dialog.show()
    driver = ComponentDriver(dialog)
    qt_until(lambda: driver.observe("Refresh files").visible, label="predictor shown")
    yield driver
    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def _capture(driver: ComponentDriver, tmp_path: Path, step: str) -> None:
    root = Path(os.environ.get("PREDICTOR_UI_EVIDENCE", str(tmp_path)))
    driver.screenshot(root / f"{step}.png")


def _has_curve(driver: ComponentDriver, ylabel: str) -> bool:
    return any(
        axes.ylabel == ylabel and axes.visible_curves >= 5
        for axes in driver.plots().axes
    )


def test_predictor_model_tabs_and_nullable_fields(predictor_ui, tmp_path: Path) -> None:
    driver = predictor_ui
    for index, (tab, ylabel) in enumerate(
        (
            ("Frequency", "Frequency (MHz)"),
            ("|n|", "|<i|n|j>|"),
            ("|phi|", "|<i|phi|j>|"),
        )
    ):
        driver.select_tab(tab)
        qt_until(partial(_has_curve, driver, ylabel), label=f"{tab} curves")
        assert driver.tabs(tab).current == tab
        assert driver.legends().texts == ("0→1", "0→2", "0→3", "1→2", "1→3")
        _capture(driver, tmp_path, f"0{index + 1}-fluxonium-{index}")

    driver.choose_option("Model:", "SQUID transmon")
    qt_until(lambda: _has_curve(driver, "Frequency (MHz)"), label="transmon curves")
    assert driver.tabs("Frequency").current == "Frequency"
    driver.select_tab("|n|")
    assert driver.tabs("Frequency").current == "Frequency"
    driver.select_tab("|phi|")
    assert driver.tabs("Frequency").current == "Frequency"
    _capture(driver, tmp_path, "04-transmon-matrix-disabled")

    driver.choose_option("Model:", "Fluxonium")
    driver.select_tab("|n|")
    qt_until(lambda: _has_curve(driver, "|<i|n|j>|"), label="matrix restored")
    assert driver.tabs("|n|").current == "|n|"
    _capture(driver, tmp_path, "05-fluxonium-matrix-restored")

    driver.enter_text(object_name="predictor_input_EJ", value="", commit=False)
    qt_until(
        lambda: "EJ" in driver.observe(object_name="predictor_status").text,
        label="missing energy",
    )
    assert driver.observe(object_name="predictor_input_EJ").text == ""
    assert not _has_curve(driver, "|<i|n|j>|")
    _capture(driver, tmp_path, "06-missing-energy")
    driver.enter_text(object_name="predictor_input_EJ", value="4.0", commit=False)
    qt_until(lambda: _has_curve(driver, "|<i|n|j>|"), label="energy recovered")
    assert driver.observe(object_name="predictor_input_EJ").text == "4.0"
    _capture(driver, tmp_path, "07-energy-recovered")


@pytest.mark.parametrize("predictor_ui", [True], indirect=True)
def test_predictor_live_controls_stay_locked(predictor_ui, tmp_path: Path) -> None:
    driver = predictor_ui
    assert not driver.observe(object_name="predictor_input_EJ").enabled
    assert not driver.observe("Refresh files").enabled
    assert not driver.observe("Load").enabled
    assert not driver.observe("Add").enabled
    driver.enter_text(object_name="predictor_input_EJ", value="", commit=False)
    driver.choose_option("Model:", "SQUID transmon")
    assert driver.observe(object_name="predictor_input_EJ").text == "4.0"
    driver.select_tab("|phi|")
    qt_until(lambda: _has_curve(driver, "|<i|phi|j>|"), label="live matrix viewing")
    assert driver.tabs("|phi|").current == "|phi|"
    _capture(driver, tmp_path, "08-live-locked")
