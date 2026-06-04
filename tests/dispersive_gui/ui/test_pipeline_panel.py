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
    assert not panel._export_box.isEnabled()  # no result yet

    state.set_disp_result(g=0.06, bare_rf=5.3, res_dim=4, step=1)
    panel._sync_enabled()
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


def test_progress_routes_to_the_active_bar(qapp):
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    panel = _panel(qapp, state)
    # preprocess (step 3) and tune (step 4) each have their own bar
    assert panel._progress is not panel._tune_progress
    assert not panel._progress.isVisible()
    assert not panel._tune_progress.isVisible()

    # routing the shared progress signal targets whichever bar is active
    panel._begin_progress(panel._tune_progress)
    assert panel._active_progress is panel._tune_progress
    panel._on_progress(3.0, 10.0, "Predicting")
    assert panel._tune_progress.value() == 3
    panel._end_progress()
    assert not panel._tune_progress.isVisible()
    assert panel._active_progress is None
    # with no active worker, a stray progress signal is ignored (no bar updated)
    panel._on_progress(5.0, 10.0, "stray")  # must not raise / touch any bar

    # preprocess routes to its own bar
    panel._begin_progress(panel._progress)
    panel._on_progress(4.0, 8.0, "edelay")
    assert panel._progress.value() == 4
    panel._end_progress()


def test_tune_done_slot_records_manual_fit_and_draws(qapp):
    # the predict worker's main-thread done slot: records the chosen g/bare_rf (the
    # manual tuning IS the final fit) and draws the tune figure (no scqubits —
    # _TuneData carries precomputed arrays).
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import _TuneData

    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    state.set_preprocess(_preprocess())
    panel = _panel(qapp, state)

    t = np.linspace(0.0, 1.0, 12)
    data = _TuneData(
        rf_0=np.full(12, 5.4),
        rf_1=np.full(12, 5.6),
        t_fluxs=t,
        g=0.065,
        bare_rf=5.32,
        res_dim=5,
        step=2,
    )
    panel._on_tune_done(data)

    assert state.disp_fit.g == pytest.approx(0.065)
    assert state.disp_fit.bare_rf == pytest.approx(5.32)
    assert state.disp_fit.res_dim == 5 and state.disp_fit.step == 2
    assert panel._export_box.isEnabled()  # the tuning is the result
    assert panel._tune_artists is not None  # figure drawn


def test_tune_button_disabled_during_compute(qapp, monkeypatch):
    # pressing "Use these g/r_f" disables the button until the worker finishes,
    # so a new compute cannot start while one is in flight.
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import _TuneData

    monkeypatch.setattr(predict_mod, "calculate_dispersive_vs_flux", _stub)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    state.set_preprocess(_preprocess())
    panel = _panel(qapp, state)

    panel._on_tune()  # spawns the worker
    assert panel._tune_btn.isEnabled() is False
    # the done slot re-enables it
    t = np.linspace(0.0, 1.0, 12)
    panel._on_tune_done(
        _TuneData(np.full(12, 5.4), np.full(12, 5.6), t, 0.06, 5.3, res_dim=4, step=1)
    )
    assert panel._tune_btn.isEnabled() is True
