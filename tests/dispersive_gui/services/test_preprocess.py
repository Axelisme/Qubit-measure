"""Tests for dispersive PreprocessService — the signal pipeline (numba edelay kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.dispersive.services._fast_edelay import fast_edelays
from zcu_tools.gui.app.dispersive.services.preprocess import (
    PreprocessService,
    compute_preprocess,
)
from zcu_tools.gui.app.dispersive.state import (
    DispersiveState,
    FluxoniumInputs,
    OnetoneEntry,
)
from zcu_tools.notebook.persistance import SpectrumData
from zcu_tools.utils.fitting.resonance import HangerModel, fit_edelay


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


def _homophasal_like_grid(
    freq: float, Ql: float, half_span: float, count: int
) -> np.ndarray:
    def theta(value: float) -> float:
        return 2.0 * np.arctan(2.0 * Ql * (1.0 - value / freq))

    phases = np.linspace(theta(freq - half_span), theta(freq + half_span), count)
    return np.asarray(
        freq * (1.0 - np.tan(phases / 2.0) / (2.0 * Ql)),
        dtype=np.float64,
    )


def test_fast_edelays_shares_large_nonuniform_branch_with_scalar_fit() -> None:
    freq, Ql, edelay = 5549.0, 740.0, 11.299
    freqs = _homophasal_like_grid(freq, Ql, 15.0, 301)
    signals = np.asarray(
        [
            HangerModel.calc_signals(
                freqs,
                freq + offset,
                Ql,
                1100.0,
                0.12,
                1.25 * np.exp(0.31j),
                edelay,
                0.004,
            )
            for offset in (-0.3, 0.0, 0.25)
        ],
        dtype=np.complex128,
    )

    fast = fast_edelays(freqs, signals)
    scalar = np.asarray([fit_edelay(freqs, row) for row in signals])

    np.testing.assert_allclose(fast, scalar, atol=3e-3)
    np.testing.assert_allclose(np.median(fast), edelay, atol=3e-3)


def test_compute_preprocess_shapes_and_norm():
    fluxs, freqs, signals = _synthetic_onetone()
    result = compute_preprocess(fluxs, freqs, signals)

    assert result.norm_phases.shape == signals.shape
    np.testing.assert_allclose(result.sp_fluxs, fluxs)
    np.testing.assert_allclose(result.sp_freqs, freqs)
    # row-normalized: each row's max is 1.0
    np.testing.assert_allclose(result.norm_phases.max(axis=1), 1.0)
    assert result.edelays.shape == (len(fluxs),)
    assert np.isfinite(result.edelay)
    # median_rf is the median over flux of each row's peak frequency, in range
    assert float(freqs.min()) <= result.median_rf <= float(freqs.max())


def test_median_rf_lands_on_the_resonance_band():
    # the synthetic onetone's resonance sweeps around 5.3 GHz (±0.2 cos); the median
    # of per-flux peak frequencies should land in that band, not at an axis edge.
    fluxs, freqs, signals = _synthetic_onetone(n_flux=12, n_freq=80)
    result = compute_preprocess(fluxs, freqs, signals)
    assert 5.1 <= result.median_rf <= 5.5


def test_compute_preprocess_small_grid_does_not_crash():
    # fewer freqs than a smoothing divisor → σ would be 0; the floor must save it.
    fluxs, freqs, signals = _synthetic_onetone(n_flux=6, n_freq=8)
    result = compute_preprocess(fluxs, freqs, signals)
    assert result.norm_phases.shape == signals.shape


def test_compute_preprocess_signature_is_deterministic():
    fluxs, freqs, signals = _synthetic_onetone()
    r1 = compute_preprocess(fluxs, freqs, signals)
    r2 = compute_preprocess(fluxs, freqs, signals)
    assert r1.signature == r2.signature
    # signature encodes smoothing method, divisors, and grid shape
    assert r1.signature == ("wavelet", 30, 10, len(fluxs), len(freqs))


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
