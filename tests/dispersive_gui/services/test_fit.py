"""Tests for dispersive FitService — auto_fit_dispersive port + compute/record split.

``calculate_dispersive_vs_flux`` (scqubits) is monkeypatched to a cheap analytic
stub so the optimizer runs fast and deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest
import zcu_tools.gui.app.dispersive.services.fit as fit_mod
from zcu_tools.gui.app.dispersive.services.fit import (
    AutoFitResult,
    FitService,
    auto_fit_dispersive,
)
from zcu_tools.gui.app.dispersive.state import (
    DispersiveState,
    FluxoniumInputs,
    PreprocessResult,
)


def _stub_calc(g_true=0.06):
    """A stub where the predicted rf overlaps the signal best at g=g_true.

    rf_0 moves *gently* with g (small slope, stays inside the freq axis) so the
    predicted line sweeps across the signal's 5.5 GHz peak as g varies, giving a
    smooth overlap landscape the optimizer can descend; overlap peaks at g_true.
    """

    def fake(params, fluxs, bare_rf, g, *, progress=False, res_dim=4):
        # rf_0 hits the 5.5 GHz peak exactly at g == g_true; slope kept small so it
        # stays within [5.0, 6.0] across the g_bound, keeping the gradient live.
        rf0 = np.full(len(fluxs), 5.5 + 2.0 * (g - g_true))
        rf1 = np.full(len(fluxs), 5.9)
        return rf0, rf1

    return fake


def _preprocess(n_flux=20, n_freq=40):
    fluxs = np.linspace(0.0, 1.0, n_flux).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, n_freq).astype(np.float64)
    # signal peaks at 5.5 GHz (a broad Lorentzian), so overlap is best when
    # rf_0 ≈ 5.5; the width is wide enough that the optimizer sees a gradient.
    norm = 1.0 / (1.0 + ((freqs - 5.5) / 0.2) ** 2)
    norm_phases = np.tile(norm, (n_flux, 1))
    return PreprocessResult(
        sp_fluxs=fluxs,
        sp_freqs=freqs,
        norm_phases=norm_phases,
        edelays=np.zeros(n_flux),
        edelay=0.0,
        signature=("x",),
    )


def test_auto_fit_finds_g(monkeypatch):
    monkeypatch.setattr(fit_mod, "calculate_dispersive_vs_flux", _stub_calc(0.06))
    pp = _preprocess()
    result = auto_fit_dispersive(
        params=(4.0, 1.0, 0.5),
        bare_rf=5.3,
        sp_fluxs=pp.sp_fluxs,
        sp_freqs=pp.sp_freqs,
        norm_phases=pp.norm_phases,
        g_bound=(0.0, 0.2),
        g_init=0.1,
        fit_bare_rf=False,
    )
    assert isinstance(result, AutoFitResult)
    assert result.bare_rf is None  # not fit
    assert result.g == pytest.approx(0.06, abs=0.01)


def test_compute_autofit_requires_preprocess():
    st = DispersiveState()
    with pytest.raises(RuntimeError, match="no preprocessing"):
        FitService(st).compute_autofit()


def _wired_state(monkeypatch):
    monkeypatch.setattr(fit_mod, "calculate_dispersive_vs_flux", _stub_calc(0.06))
    st = DispersiveState()
    st.set_fit_inputs(
        FluxoniumInputs(
            params=(4.0, 1.0, 0.5),
            flux_half=0.5,
            flux_int=1.0,
            flux_period=2.0,
            bare_rf_seed=5.3,
        )
    )
    st.set_preprocess(_preprocess())
    return st


def test_compute_then_record_split(monkeypatch):
    st = _wired_state(monkeypatch)
    svc = FitService(st)
    result = svc.compute_autofit()  # pure, no State write
    assert st.disp_fit.g is None
    svc.record_result(result)  # main-thread write
    assert st.disp_fit.g == pytest.approx(0.06, abs=0.01)
    assert st.disp_fit.bare_rf == 5.3  # kept seed (bare_rf not fit)
    assert st.disp_fit.auto_fit_done is True


def test_set_manual_fit_records_slider_values():
    st = DispersiveState()
    FitService(st).set_manual_fit(0.07, 5.35)
    assert st.disp_fit.g == 0.07
    assert st.disp_fit.bare_rf == 5.35
    assert st.disp_fit.auto_fit_done is False
