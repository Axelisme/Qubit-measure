"""Tests for the dispersive PipelinePanelWidget — gating + worker done→record→redraw.

The QThreadPool workers' ``done`` signals need a running event loop to deliver, so
these tests call the panel's main-thread ``done`` slots directly (the slot is what
records State + redraws) and exercise ``_sync_enabled`` gating — the State-invariant
contract is that only those slots write State. scqubits is monkeypatched.
"""

from __future__ import annotations

import numpy as np
import pytest
import zcu_tools.simulate.fluxonium.prediction as prediction_mod
from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.state import (
    DispersiveState,
    FluxoniumInputs,
    OnetoneEntry,
    PreprocessResult,
)
from zcu_tools.gui.project import ProjectInfo
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
    assert not panel._export_btn.isEnabled()  # export (in step 4) — no result yet

    state.set_disp_result(g=0.06, bare_rf=5.3, res_dim=4)
    panel._sync_enabled()
    assert panel._export_btn.isEnabled()


def test_preprocess_done_slot_records_and_enables(qapp, monkeypatch):
    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", _stub)
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


def test_g_slider_range_and_default(qapp):
    # g is a slider over a fixed 0..200 MHz (1 MHz ticks), default 50 MHz.
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    panel = _panel(qapp, state)
    assert panel._g_slider.minimum() == 0
    assert panel._g_slider.maximum() == 200
    assert panel._g_slider.value() == 50
    assert panel._g_mhz() == pytest.approx(50.0)
    assert "50.0 MHz" in panel._g_label.text()
    panel._g_slider.setValue(120)
    assert panel._g_mhz() == pytest.approx(120.0)
    assert "120.0 MHz" in panel._g_label.text()


def test_export_button_lives_in_tune_section(qapp):
    # Step 5 is gone: the export button is part of step 4 and gated on a result.
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    panel = _panel(qapp, state)
    assert not hasattr(panel, "_export_box")
    assert panel._export_btn.parent() is panel._tune_box
    assert not panel._export_btn.isEnabled()  # no result yet


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
    )
    panel._on_tune_done(data)

    assert state.disp_fit.g == pytest.approx(0.065)
    assert state.disp_fit.bare_rf == pytest.approx(5.32)
    assert panel._export_btn.isEnabled()  # the tuning is the result
    # dispersion lines now drawn on the tune figure
    assert panel._tune_artists is not None
    assert panel._tune_artists.line_ground is not None


def test_tune_button_disabled_during_compute(qapp, monkeypatch):
    # pressing "Use these g/r_f" disables the button until the worker finishes,
    # so a new compute cannot start while one is in flight.
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import _TuneData

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", _stub)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    pp = _preprocess()
    state.set_preprocess(pp)
    panel = _panel(qapp, state)
    panel._init_tune_view(pp)

    panel._on_tune()  # spawns the worker
    assert panel._tune_btn.isEnabled() is False
    # Drain the worker AND flush its queued main-thread delivery before teardown, so
    # the carrier is not destroyed with a pending QMetaCallEvent (would segfault).
    panel._runner.quiesce()
    # the done slot re-enables it
    t = np.linspace(0.0, 1.0, 12)
    panel._on_tune_done(_TuneData(np.full(12, 5.4), np.full(12, 5.6), t, 0.06, 5.3))
    assert panel._tune_btn.isEnabled() is True


# --- sample-flux lines ------------------------------------------------------


def _tune_panel(qapp, monkeypatch, median_rf=5.4):
    """A panel with inputs/onetone/preprocess set and the tune view initialised."""
    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", _stub)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    state.set_fit_inputs(_inputs())
    state.set_onetone(_onetone())
    pp = _preprocess(median_rf=median_rf)
    state.set_preprocess(pp)
    panel = _panel(qapp, state)
    panel._init_tune_view(pp)
    return panel


def test_add_sample_drops_line_with_dots(qapp, monkeypatch):
    panel = _tune_panel(qapp, monkeypatch)
    panel._on_add_sample()
    assert panel._tune_artists is not None
    assert len(panel._tune_artists.samples) == 1
    s = panel._tune_artists.samples[0]
    # dropped at the flux-axis centre (0.0..1.0 → 0.5)
    assert s.flux == pytest.approx(0.5)
    # the stub returns 5.4 / 5.5 GHz → dots exist (red/blue)
    assert s.dot_ground is not None and s.dot_excited is not None
    assert float(np.asarray(s.dot_ground.get_ydata())[0]) == pytest.approx(5400.0)


def test_rf_slider_refreshes_sample_dots_after_debounce(qapp, monkeypatch):
    # The slider move debounces the dot recompute (a QTimer): moving the slider does
    # NOT recompute synchronously; firing the debounce slot does. Stub echoes bare_rf.
    panel = _tune_panel(qapp, monkeypatch)

    def echo_rf(params, fluxs, bare_rf, g, *, progress=False, res_dim=4, **kw):
        return tuple(
            np.full(len(fluxs), bare_rf + k) for k in range(kw.get("return_dim", 2))
        )

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", echo_rf)
    panel._on_add_sample()
    s = panel._tune_artists.samples[0]  # type: ignore[union-attr]

    panel._rf_slider.setValue(165)  # → 5.55 GHz; the debounce timer is (re)started
    assert panel._dot_debounce.isActive()  # recompute is pending, not done yet
    panel._on_dot_debounce_fired()  # simulate the timer firing
    # the dot's ground y now reflects the new r_f (echoed by the stub), in MHz
    assert s.dot_ground is not None
    assert float(np.asarray(s.dot_ground.get_ydata())[0]) == pytest.approx(5550.0)


def test_g_change_refreshes_sample_dots_after_debounce(qapp, monkeypatch):
    panel = _tune_panel(qapp, monkeypatch)

    def echo_g(params, fluxs, bare_rf, g, *, progress=False, res_dim=4, **kw):
        return tuple(np.full(len(fluxs), g + k) for k in range(kw.get("return_dim", 2)))

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", echo_g)
    panel._on_add_sample()
    s = panel._tune_artists.samples[0]  # type: ignore[union-attr]

    panel._g_slider.setValue(90)  # 90 MHz → g = 0.09 GHz; debounce started
    assert panel._dot_debounce.isActive()
    panel._on_dot_debounce_fired()
    # ground dot y = g (echoed) in MHz = 0.09 GHz → 90 MHz
    assert s.dot_ground is not None
    assert float(np.asarray(s.dot_ground.get_ydata())[0]) == pytest.approx(90.0)


def test_drag_moves_line_without_recompute_until_drop(qapp, monkeypatch):
    # While dragging, the line moves but the dot is NOT recomputed (no compute spam);
    # the recompute happens only on _on_sample_drop (mouse release).
    panel = _tune_panel(qapp, monkeypatch)

    def echo_flux(params, fluxs, bare_rf, g, *, progress=False, res_dim=4, **kw):
        # echo the flux into the ground value so we can see the recompute
        arr = np.asarray(fluxs, dtype=float)
        return tuple(arr + k for k in range(kw.get("return_dim", 2)))

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", echo_flux)
    panel._on_add_sample()
    s = panel._tune_artists.samples[0]  # type: ignore[union-attr]
    before = float(np.asarray(s.dot_ground.get_ydata())[0])  # type: ignore[union-attr]

    panel._on_sample_drag(s, 0.3)  # motion: line moves, dot unchanged
    assert s.flux == pytest.approx(0.3)
    assert float(np.asarray(s.line.get_xdata())[0]) == pytest.approx(0.3)
    assert s.dot_ground is not None
    assert float(np.asarray(s.dot_ground.get_ydata())[0]) == pytest.approx(before)

    panel._on_sample_drop(s)  # release: now recompute
    # the dot ground y = flux (echoed) in MHz = 0.3 GHz → 300 MHz
    assert float(np.asarray(s.dot_ground.get_ydata())[0]) == pytest.approx(300.0)


def test_drag_clamps_flux_to_axis_range(qapp, monkeypatch):
    panel = _tune_panel(qapp, monkeypatch)
    panel._on_add_sample()
    s = panel._tune_artists.samples[0]  # type: ignore[union-attr]
    panel._on_sample_drag(s, 5.0)  # way past the 0..1 flux axis
    assert s.flux == pytest.approx(1.0)  # clamped to the axis max


def test_clear_samples_removes_all(qapp, monkeypatch):
    panel = _tune_panel(qapp, monkeypatch)
    panel._on_add_sample()
    panel._on_add_sample()
    assert len(panel._tune_artists.samples) == 2  # type: ignore[union-attr]
    panel._on_clear_samples()
    assert panel._tune_artists.samples == []  # type: ignore[union-attr]


def test_fresh_preprocess_drops_sample_lines(qapp, monkeypatch):
    panel = _tune_panel(qapp, monkeypatch)
    panel._on_add_sample()
    assert len(panel._tune_artists.samples) == 1  # type: ignore[union-attr]
    # re-initialising the tune view (a new preprocess) starts with no sample lines
    panel._init_tune_view(_preprocess())
    assert panel._tune_artists.samples == []  # type: ignore[union-attr]


# --- auto tune --------------------------------------------------------------


def test_auto_tune_button_enabled_only_with_samples(qapp, monkeypatch):
    panel = _tune_panel(qapp, monkeypatch)
    # no sample lines yet → Auto tune disabled
    assert panel._auto_tune_btn.isEnabled() is False
    panel._on_add_sample()
    assert panel._auto_tune_btn.isEnabled() is True
    panel._on_clear_samples()
    assert panel._auto_tune_btn.isEnabled() is False


def test_auto_tune_done_writes_back_sliders(qapp, monkeypatch):
    # the done slot sets the g / r_f sliders to the optimised values (it does NOT
    # accept the fit — no result is recorded).
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import _AutoTuneResult

    panel = _tune_panel(qapp, monkeypatch, median_rf=5.4)  # freqs 5.0..6.0
    panel._on_add_sample()
    # optimum g = 0.085 GHz → 85 MHz tick; bare_rf = 5.55 GHz → tick 165 (span 1.0)
    panel._on_auto_tune_done(_AutoTuneResult(g=0.085, bare_rf=5.55))
    assert panel._g_slider.value() == 85
    assert panel._rf_ghz() == pytest.approx(5.55)
    # auto-tune does not accept → no result recorded, export still disabled
    assert not panel._ctrl.state.disp_fit.has_result
    assert not panel._export_btn.isEnabled()


def test_auto_tune_disables_button_during_compute(qapp, monkeypatch):
    panel = _tune_panel(qapp, monkeypatch)
    panel._on_add_sample()
    panel._on_auto_tune()  # spawns the worker
    assert panel._auto_tune_btn.isEnabled() is False
    # Drain the worker AND flush its queued main-thread delivery before teardown, so
    # the carrier is not destroyed with a pending QMetaCallEvent (would segfault).
    panel._runner.quiesce()
