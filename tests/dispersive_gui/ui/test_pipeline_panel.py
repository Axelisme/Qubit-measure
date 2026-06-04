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
    assert not panel._export_box.isEnabled()  # no result yet

    state.set_disp_result(g=0.06, bare_rf=5.3, res_dim=4, step=1)
    panel._sync_enabled()
    assert panel._export_box.isEnabled()


def test_preprocess_done_slot_records_and_enables(qapp, monkeypatch):
    monkeypatch.setattr(predict_mod, "calculate_dispersive_vs_flux_fast", _stub)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    panel = _panel(qapp, state)

    result = _preprocess(median_rf=5.4)  # freqs 5.0..6.0 GHz
    panel._on_preprocess_done(result)  # the main-thread slot

    assert state.preprocess is result  # recorded
    assert panel._tune_box.isEnabled()
    # the r_f slider has a fixed 0..300-tick range (precision = span/300); its GHz
    # value maps from the tick, default nearest median_rf=5.4 → tick 120 → 5.4 GHz
    assert panel._rf_slider.maximum() == 300
    assert panel._rf_slider.value() == 120
    assert panel._rf_ghz() == pytest.approx(5.4)
    assert panel._tune_artists is not None
    assert panel._tune_artists.line_ground is None  # no prediction yet


def test_rf_slider_moves_bare_line_live(qapp):
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    pp = _preprocess(median_rf=5.4)  # freqs 5.0..6.0 GHz, span 1.0
    state.set_preprocess(pp)
    panel = _panel(qapp, state)
    panel._init_tune_view(pp)

    # tick 165 → 5.0 + 165/300 * 1.0 = 5.55 GHz = 5550 MHz; moves the line live
    panel._rf_slider.setValue(165)
    assert panel._rf_ghz() == pytest.approx(5.55)
    assert "5550.0" in panel._rf_label.text()
    assert panel._tune_artists is not None
    ydata = np.asarray(panel._tune_artists.line_bare.get_ydata())
    assert float(ydata[0]) == pytest.approx(5550.0)  # MHz
    assert panel._tune_artists.line_ground is None  # still no dispersion lines


def test_rf_slider_precision_is_span_over_300(qapp):
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    pp = _preprocess(median_rf=5.4)
    state.set_preprocess(pp)
    panel = _panel(qapp, state)
    panel._init_tune_view(pp)
    # one tick = span / 300 = 1.0 GHz / 300 ≈ 3.333 MHz
    base = panel._rf_ghz()
    panel._rf_slider.setValue(panel._rf_slider.value() + 1)
    assert 1e3 * (panel._rf_ghz() - base) == pytest.approx(1000.0 / 300.0, abs=1e-6)


def test_busy_progress_bars_show_and_hide(qapp):
    # use isHidden() (the explicit shown/hidden flag) rather than isVisible(), which
    # is False for any widget whose top-level window has not been shown (offscreen).
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    panel = _panel(qapp, state)
    # preprocess (step 3) and tune (step 4) each have their own busy bar
    assert panel._progress is not panel._tune_progress
    assert panel._progress.isHidden()
    assert panel._tune_progress.isHidden()

    # begin → that bar shows as indeterminate (range 0,0 = busy spinner)
    panel._begin_progress(panel._tune_progress)
    assert panel._active_progress is panel._tune_progress
    assert not panel._tune_progress.isHidden()
    assert panel._tune_progress.maximum() == 0  # indeterminate
    panel._end_progress()
    assert panel._tune_progress.isHidden()
    assert panel._active_progress is None


def test_tune_done_slot_records_manual_fit_and_draws(qapp):
    # the predict worker's main-thread done slot: records the chosen g/bare_rf (the
    # manual tuning IS the final fit) and draws the tune figure (no scqubits —
    # _TuneData carries precomputed arrays).
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import _TuneData

    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    pp = _preprocess()
    state.set_preprocess(pp)
    panel = _panel(qapp, state)
    panel._init_tune_view(pp)  # the tune background must exist before predict

    t = np.linspace(0.0, 1.0, 12)
    data = _TuneData(
        rf_0=np.full(12, 5.4),
        rf_1=np.full(12, 5.6),
        t_fluxs=t,
        g=0.065,
        bare_rf=5.32,
        step=2,
    )
    panel._on_tune_done(data)

    assert state.disp_fit.g == pytest.approx(0.065)
    assert state.disp_fit.bare_rf == pytest.approx(5.32)
    assert state.disp_fit.step == 2
    assert panel._export_box.isEnabled()  # the tuning is the result
    # dispersion lines now drawn on the tune figure
    assert panel._tune_artists is not None
    assert panel._tune_artists.line_ground is not None


def test_tune_button_disabled_during_compute(qapp, monkeypatch):
    # pressing "Use these g/r_f" disables the button until the worker finishes,
    # so a new compute cannot start while one is in flight.
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import _TuneData

    monkeypatch.setattr(predict_mod, "calculate_dispersive_vs_flux_fast", _stub)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    pp = _preprocess()
    state.set_preprocess(pp)
    panel = _panel(qapp, state)
    panel._init_tune_view(pp)

    panel._on_tune()  # spawns the worker
    assert panel._tune_btn.isEnabled() is False
    panel._pool.waitForDone(5000)  # let the spawned worker finish before teardown
    # the done slot re-enables it
    t = np.linspace(0.0, 1.0, 12)
    panel._on_tune_done(
        _TuneData(np.full(12, 5.4), np.full(12, 5.6), t, 0.06, 5.3, step=1)
    )
    assert panel._tune_btn.isEnabled() is True
