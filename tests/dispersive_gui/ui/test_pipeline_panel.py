"""Tests for the dispersive PipelinePanelWidget — gating + worker done→record→redraw.

The QThreadPool workers' ``done`` signals need a running event loop to deliver, so
these tests call the panel's main-thread ``done`` slots directly (the slot is what
records State + redraws) and exercise ``_sync_enabled`` gating — the State-invariant
contract is that only those slots write State. scqubits is monkeypatched.
"""

from __future__ import annotations

import numpy as np
import pytest
import zcu_tools.gui.app.dispersive.services.predict as predict_mod
from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.services.fit import AutoFitResult
from zcu_tools.gui.app.dispersive.state import (
    DispersiveState,
    FluxoniumInputs,
    OnetoneEntry,
    PreprocessResult,
    ProjectInfo,
)
from zcu_tools.notebook.persistance import SpectrumData


def _stub(params, fluxs, bare_rf, g, *, progress=False, res_dim=4, **kw):
    return tuple(
        np.full(len(fluxs), 5.4 + 0.1 * k) for k in range(kw.get("return_dim", 2))
    )


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


def _preprocess(n_flux=12, n_freq=30) -> PreprocessResult:
    fluxs = np.linspace(0.0, 1.0, n_flux).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, n_freq).astype(np.float64)
    return PreprocessResult(
        sp_fluxs=fluxs,
        sp_freqs=freqs,
        norm_phases=np.random.RandomState(0).rand(n_flux, n_freq),
        edelays=np.zeros(n_flux),
        edelay=0.0,
        signature=("x",),
    )


def _panel(qapp, state):
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import PipelinePanelWidget

    return PipelinePanelWidget(Controller(state))


def test_sections_gated_on_pipeline_progress(qapp):
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    panel = _panel(qapp, state)
    # nothing loaded → only section 1 active
    assert not panel._load_box.isEnabled()
    assert not panel._preprocess_box.isEnabled()

    state.set_fit_inputs(_inputs())
    panel._sync_enabled()
    assert panel._load_box.isEnabled()
    assert not panel._preprocess_box.isEnabled()

    state.set_onetone(_onetone())
    panel._sync_enabled()
    assert panel._preprocess_box.isEnabled()
    assert not panel._tune_box.isEnabled()

    state.set_preprocess(_preprocess())
    panel._sync_enabled()
    assert panel._tune_box.isEnabled()
    assert panel._autofit_box.isEnabled()
    assert not panel._result_box.isEnabled()

    state.set_disp_result(g=0.06, bare_rf=5.3, auto=True)
    panel._sync_enabled()
    assert panel._result_box.isEnabled()
    assert panel._export_box.isEnabled()


def test_preprocess_done_slot_records_and_enables(qapp, monkeypatch):
    monkeypatch.setattr(predict_mod, "calculate_dispersive_vs_flux", _stub)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    panel = _panel(qapp, state)

    result = _preprocess()
    panel._on_preprocess_done(result)  # the main-thread slot

    assert state.preprocess is result  # recorded
    assert panel._tune_box.isEnabled()


def test_autofit_done_slot_records_and_updates_sliders(qapp, monkeypatch):
    monkeypatch.setattr(predict_mod, "calculate_dispersive_vs_flux", _stub)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    state.set_preprocess(_preprocess())
    panel = _panel(qapp, state)

    panel._on_autofit_done(AutoFitResult(g=0.07, bare_rf=5.35))

    assert state.disp_fit.g == 0.07
    assert state.disp_fit.bare_rf == 5.35
    # sliders reflect the fitted values (MHz)
    assert panel._g_spin.value() == pytest.approx(70.0)
    assert panel._rf_spin.value() == pytest.approx(5350.0)
    assert panel._result_box.isEnabled()


def test_tune_slider_change_predicts_without_crash(qapp, monkeypatch):
    monkeypatch.setattr(predict_mod, "calculate_dispersive_vs_flux", _stub)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    state.set_preprocess(_preprocess())
    panel = _panel(qapp, state)

    # a slider move triggers _on_tune_changed → predict → redraw (no raise)
    panel._g_spin.setValue(80.0)
    panel._on_tune_changed()
    assert panel._tune_artists is not None


def test_finish_tune_records_manual_result(qapp):
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    state.set_preprocess(_preprocess())
    panel = _panel(qapp, state)

    panel._g_spin.setValue(65.0)
    panel._rf_spin.setValue(5320.0)
    panel._on_finish_tune()

    assert state.disp_fit.g == pytest.approx(0.065)
    assert state.disp_fit.bare_rf == pytest.approx(5.32)
    assert state.disp_fit.auto_fit_done is False
