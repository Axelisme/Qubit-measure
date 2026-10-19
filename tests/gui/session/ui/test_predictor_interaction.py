"""Visible predictor editing and curve interactions with synthetic query results."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtCore import QTimer
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.predictor import (
    PredictCurveRequest,
    PredictCurveResult,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    PredictMatrixCurveResult,
    PredictorService,
)
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.predictor import PredictorParams

from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver


class SyntheticCurves(PredictorService):
    """Supply deterministic visible curves without numerical solvers or devices."""

    def predict_freq_curve(
        self, params: PredictorParams, req: PredictCurveRequest
    ) -> PredictCurveResult:
        labels = tuple(f"{frm}→{to}" for frm, to in req.transitions)
        series = np.array(
            [1000 * (to - frm) + 100 * req.values for frm, to in req.transitions]
        )
        return PredictCurveResult(labels, req.values, req.values, series)

    def predict_matrix_element_curve(
        self, params: PredictorParams, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        labels = tuple(f"{frm}→{to}" for frm, to in req.transitions)
        scale = 0.1 if req.operator == "n" else 0.2
        series = np.array(
            [scale * (to - frm) + 0.01 * req.values for frm, to in req.transitions]
        )
        return PredictMatrixCurveResult(labels, req.values, req.values, series)

    def predict_freq(self, params: PredictorParams, req: PredictFreqRequest) -> float:
        return 1000 * (req.transition[1] - req.transition[0]) + 100 * req.value


@pytest.fixture
def predictor_ui(qapp, monkeypatch: pytest.MonkeyPatch, request):
    # Qt scheduling is the external clock seam; completion is still visible state.
    original = QTimer.setInterval

    def immediate_debounce(timer: QTimer, milliseconds: int) -> None:
        original(timer, 0 if milliseconds == 150 else milliseconds)

    monkeypatch.setattr(QTimer, "setInterval", immediate_debounce)
    dialog = PredictorDialog(
        PredictorControlFacet(SyntheticCurves()),
        params=FluxoniumParams(4.0, 1.0, 1.0, 0.0, 1.0, 0.0),
    )
    if getattr(request, "param", False):
        dialog.set_live_mode(enabled=True)
    dialog.show()
    driver = ComponentDriver(dialog)
    qt_until(lambda: driver.observe("Browse").visible, label="predictor shown")
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


def test_predictor_model_tabs_and_known_fields(predictor_ui, tmp_path: Path) -> None:
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

    driver.click("EJ")
    qt_until(
        lambda: "EJ" in driver.observe(object_name="predictor_status").text,
        label="missing energy",
    )
    assert driver.observe("EJ").checked is False
    assert not _has_curve(driver, "|<i|n|j>|")
    _capture(driver, tmp_path, "06-missing-energy")
    driver.click("EJ")
    qt_until(lambda: _has_curve(driver, "|<i|n|j>|"), label="energy recovered")
    assert driver.observe("EJ").checked is True
    _capture(driver, tmp_path, "07-energy-recovered")


@pytest.mark.parametrize("predictor_ui", [True], indirect=True)
def test_predictor_live_controls_stay_locked(predictor_ui, tmp_path: Path) -> None:
    driver = predictor_ui
    assert not driver.observe("EJ").enabled
    assert not driver.observe("Browse").enabled
    assert not driver.observe("Load component").enabled
    assert not driver.observe("Add").enabled
    driver.click("EJ")
    driver.choose_option("Model:", "SQUID transmon")
    assert driver.observe("EJ").checked is True
    driver.select_tab("|phi|")
    qt_until(lambda: _has_curve(driver, "|<i|phi|j>|"), label="live matrix viewing")
    assert driver.tabs("|phi|").current == "|phi|"
    _capture(driver, tmp_path, "08-live-locked")
