"""Tests for dispersive PreprocessService — the signal pipeline (numba edelay kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.dispersive.services.preprocess import (
    PreprocessService,
)
from zcu_tools.gui.app.dispersive.state import (
    DispersiveState,
    FluxoniumInputs,
    OnetoneEntry,
)


def _synthetic_onetone(n_flux=10, n_freq=60):
    """A synthetic one-tone: a resonator dip sweeping with flux, with edelay."""
    rng = np.random.RandomState(0)
    fluxs = np.linspace(0.0, 1.0, n_flux).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, n_freq).astype(np.float64)  # GHz
    edelay = 30.0  # large electronic delay (rad/GHz scale)
    signals = np.empty((n_flux, n_freq), dtype=np.complex128)
    for i, fl in enumerate(fluxs):
        f0 = 5.3 + 0.2 * np.cos(2 * np.pi * fl)  # resonance moves with flux
        lorentz = 1.0 / (1.0 + ((freqs - f0) / 0.02) ** 2)
        base = 1.0 - 0.8 * lorentz  # dip
        phase = np.exp(1j * 2 * np.pi * freqs * edelay)
        signals[i] = base * phase + 0.01 * (rng.randn(n_freq) + 1j * rng.randn(n_freq))
    return fluxs, freqs, signals


def test_service_compute_requires_onetone():
    st = DispersiveState()
    with pytest.raises(RuntimeError, match="no one-tone"):
        PreprocessService(st).compute()


def test_service_compute_then_record_writes_state():
    fluxs, freqs, signals = _synthetic_onetone()
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
    st.set_onetone(
        OnetoneEntry(
            name="r1",
            raw=SpectrumData(
                dev_values=fluxs.copy(),
                fluxs=fluxs.copy(),
                freqs=freqs.copy(),
                signals=signals,
            ),
        )
    )
    svc = PreprocessService(st)
    result = svc.compute()  # pure, no State write yet
    assert st.preprocess is None
    svc.record(result)  # main-thread write
    assert st.preprocess is result
