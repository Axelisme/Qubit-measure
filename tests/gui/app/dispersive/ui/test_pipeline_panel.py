"""Tests for public controls in the dispersive pipeline panel."""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtWidgets import QLabel, QPushButton, QSlider
from zcu_tools.analysis.dispersive.models import PreprocessResult
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.state import (
    DispersiveState,
    FluxoniumInputs,
    OnetoneEntry,
)
from zcu_tools.gui.app.dispersive.ui.pipeline_panel import PipelinePanelWidget
from zcu_tools.gui.project import ProjectInfo


def _inputs() -> FluxoniumInputs:
    return FluxoniumInputs(
        params=(4.0, 1.0, 0.5),
        flux_half=0.5,
        flux_int=1.0,
        flux_period=2.0,
        bare_rf_seed=5.3,
    )


def _onetone(n_flux=12, n_freq=30) -> OnetoneEntry:
    fluxs = np.linspace(0.0, 1.0, n_flux).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, n_freq).astype(np.float64)
    signals = np.ones((n_flux, n_freq), dtype=np.complex128)
    return OnetoneEntry(
        name="r1",
        raw=SpectrumData(
            dev_values=fluxs.copy(), fluxs=fluxs.copy(), freqs=freqs, signals=signals
        ),
    )


def _preprocess(n_flux=12, n_freq=30, median_rf=5.4) -> PreprocessResult:
    fluxs = np.linspace(0.0, 1.0, n_flux).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, n_freq).astype(np.float64)
    return PreprocessResult(
        sp_fluxs=fluxs,
        sp_freqs=freqs,
        norm_phases=np.random.RandomState(0).rand(n_flux, n_freq),
        edelays=np.zeros(n_flux),
        edelay=0.0,
        median_rf=median_rf,
        signature=("x",),
    )


@pytest.mark.parametrize("stage", range(5))
def test_actions_gated_on_initial_pipeline_progress(qapp, stage):
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    if stage >= 1:
        state.set_fit_inputs(_inputs())
    if stage >= 2:
        state.set_onetone(_onetone())
    if stage >= 3:
        state.set_preprocess(_preprocess())
    if stage >= 4:
        state.set_disp_result(g=0.06, bare_rf=5.3, res_dim=4)
    panel = PipelinePanelWidget(Controller(state))
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}
    for threshold, text in enumerate(
        (
            "Load one-tone…",
            "Run preprocessing",
            "Use these g / r_f",
            "Export to params.json",
        ),
        start=1,
    ):
        assert buttons[text].isEnabled() == (stage >= threshold)


def test_g_slider_range_and_default(qapp):
    panel = PipelinePanelWidget(Controller(DispersiveState()))
    # The g control is the first slider in the displayed g / r_f rows.
    slider = panel.findChildren(QSlider)[0]
    assert (slider.minimum(), slider.maximum(), slider.value()) == (0, 200, 50)
    assert "50.0 MHz" in [label.text() for label in panel.findChildren(QLabel)]
    slider.setValue(120)
    assert "120.0 MHz" in [label.text() for label in panel.findChildren(QLabel)]
