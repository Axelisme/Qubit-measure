from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.utils.fitting.resonance import (
    HangerModel,
    TransmissionModel,
    calc_phase,
    fit_circle_params,
    get_proper_model,
    normalize_signal,
    phase_func,
)


def _hanger_truth(
    freqs: NDArray[np.float64],
    *,
    freq: float,
    Ql: float,
    Qc_abs: float,
    phi: float,
    a0: complex,
    edelay: float,
    bg_amp_slope: float,
) -> NDArray[np.complex128]:
    """Independent analytic oracle; intentionally does not call production calc."""
    detuning = Ql * (freqs / freq - 1.0)
    ideal = a0 * (1.0 - (Ql / Qc_abs) * np.exp(1j * phi) / (1.0 + 2j * detuning))
    background = np.exp(bg_amp_slope * (freqs - freq))
    delay = np.exp(-1j * 2.0 * np.pi * freqs * edelay)
    return np.asarray(background * ideal * delay, dtype=np.complex128)


def _transmission_truth(
    freqs: NDArray[np.float64],
    *,
    freq: float,
    Ql: float,
    a0: complex,
    edelay: float,
    bg_amp_slope: float,
) -> NDArray[np.complex128]:
    """Independent analytic oracle; intentionally does not call production calc."""
    detuning = Ql * (freqs / freq - 1.0)
    ideal = a0 / (1.0 + 2j * detuning)
    background = np.exp(bg_amp_slope * (freqs - freq))
    delay = np.exp(-1j * 2.0 * np.pi * freqs * edelay)
    return np.asarray(background * ideal * delay, dtype=np.complex128)


def _homophasal_like_grid(
    freq: float, Ql: float, half_span: float, count: int
) -> NDArray[np.float64]:
    def theta(value: float) -> float:
        return 2.0 * np.arctan(2.0 * Ql * (1.0 - value / freq))

    phases = np.linspace(theta(freq - half_span), theta(freq + half_span), count)
    return np.asarray(
        freq * (1.0 - np.tan(phases / 2.0) / (2.0 * Ql)),
        dtype=np.float64,
    )


def test_transmission_fit_recovers_freq_and_Ql():
    f0, Ql = 5000.0, 500.0
    a0 = 1.0 + 0j
    edelay = 0.0
    freqs = np.linspace(f0 - 50, f0 + 50, 801)
    signals = TransmissionModel.calc_signals(freqs, f0, Ql, a0, edelay)
    params = TransmissionModel.fit(freqs, signals, edelay=edelay)
    assert abs(params["freq"] - f0) / f0 < 1e-3
    assert abs(params["Ql"] - Ql) / Ql < 1e-1


def test_hanger_fit_recovers_freq_and_Ql():
    f0, Ql, Qc = 5000.0, 4000.0, 6000.0
    phi = 0.0
    a0 = 1.0 + 0j
    edelay = 0.0
    freqs = np.linspace(f0 - 20, f0 + 20, 801)
    signals = HangerModel.calc_signals(freqs, f0, Ql, Qc, phi, a0, edelay)
    params = HangerModel.fit(freqs, signals, edelay=edelay)
    assert abs(params["freq"] - f0) / f0 < 1e-3
    assert abs(params["Ql"] - Ql) / Ql < 2e-1


def test_circle_fit_recovers_nonuniform_exact_circle():
    center = 2.3 - 0.7j
    radius = 1.9
    angles = np.asarray(np.linspace(-2.8, 2.4, 301) ** 3 / 8.0, dtype=np.float64)
    signals = center + radius * np.exp(1j * angles)

    xc, yc, fitted_radius = fit_circle_params(signals.real, signals.imag)

    np.testing.assert_allclose(xc, center.real, atol=1e-9)
    np.testing.assert_allclose(yc, center.imag, atol=1e-9)
    np.testing.assert_allclose(fitted_radius, radius, atol=1e-9)


def test_hanger_fit_recovers_nonuniform_grid_with_edelay():
    f0, Ql, Qc = 6050.0, 4000.0, 6000.0
    phi = 0.1
    a0 = 35.0 * np.exp(0.2j)
    edelay = -1.117
    theta0 = -0.32

    def theta(freq: float) -> float:
        return theta0 + 2.0 * np.arctan(2.0 * Ql * (1.0 - freq / f0))

    thetas = np.linspace(theta(6000.0), theta(6100.0), 301)
    freqs = f0 * (1.0 - np.tan((thetas - theta0) / 2.0) / (2.0 * Ql))
    signals = HangerModel.calc_signals(freqs, f0, Ql, Qc, phi, a0, edelay)

    params = HangerModel.fit(freqs, signals)

    np.testing.assert_allclose(params["freq"], f0, atol=0.1)
    np.testing.assert_allclose(params["Ql"], Ql, rtol=0.1)
    np.testing.assert_allclose(params["edelay"], edelay, atol=1e-3)


def test_get_proper_model_hanger_vs_transmission():
    freqs = np.linspace(4980, 5020, 801)
    hanger = HangerModel.calc_signals(freqs, 5000.0, 4000.0, 6000.0, 0.0, 1.0 + 0j, 0.0)
    tm = TransmissionModel.calc_signals(freqs, 5000.0, 4000.0, 1.0 + 0j, 0.0)
    assert isinstance(get_proper_model(freqs, hanger), HangerModel)
    assert isinstance(get_proper_model(freqs, tm), TransmissionModel)


@pytest.mark.parametrize(
    ("model", "truth", "grid", "bg_amp_slope", "explicit_edelay"),
    [
        (
            HangerModel,
            _hanger_truth,
            lambda f, q: np.linspace(f - 35.0, f + 35.0, 501),
            0.012,
            True,
        ),
        (
            HangerModel,
            _hanger_truth,
            lambda f, q: _homophasal_like_grid(f, q, 35.0, 501),
            -0.009,
            False,
        ),
        (
            TransmissionModel,
            _transmission_truth,
            lambda f, q: np.linspace(f - 35.0, f + 35.0, 501),
            -0.011,
            False,
        ),
        (
            TransmissionModel,
            _transmission_truth,
            lambda f, q: _homophasal_like_grid(f, q, 35.0, 501),
            0.010,
            True,
        ),
    ],
)
def test_bg_amp_fit_recovers_analytic_truth_across_models_and_grids(
    model: type[HangerModel] | type[TransmissionModel],
    truth: Callable[..., NDArray[np.complex128]],
    grid: Callable[[float, float], NDArray[np.float64]],
    bg_amp_slope: float,
    explicit_edelay: bool,
) -> None:
    freq, Ql = 6053.0, 820.0
    a0 = 1.4 * np.exp(0.37j)
    edelay = 0.031
    freqs = grid(freq, Ql)
    common = dict(
        freq=freq,
        Ql=Ql,
        a0=a0,
        edelay=edelay,
        bg_amp_slope=bg_amp_slope,
    )
    if model is HangerModel:
        signals = truth(freqs, Qc_abs=1150.0, phi=0.13, **common)
    else:
        signals = truth(freqs, **common)

    params = model.fit(
        freqs,
        signals,
        edelay=edelay if explicit_edelay else None,
        fit_bg_amp_slope=True,
    )

    np.testing.assert_allclose(params["freq"], freq, atol=0.08)
    np.testing.assert_allclose(params["Ql"], Ql, rtol=0.03)
    np.testing.assert_allclose(params["bg_amp_slope"], bg_amp_slope, atol=3e-4)
    np.testing.assert_allclose(params["edelay"], edelay, atol=5e-4)
    if model is HangerModel:
        np.testing.assert_allclose(abs(params["Qc"]), 1150.0, rtol=0.04)
        np.testing.assert_allclose(params["phi"], 0.13, atol=0.02)


def test_bg_amp_signal_is_global_positive_real_multiplier() -> None:
    freqs = np.asarray([5980.0, 6000.0, 6030.0], dtype=np.float64)
    g = 0.017
    zero = HangerModel.calc_signals(
        freqs, 6000.0, 700.0, 950.0, 0.21, 1.2 - 0.3j, 0.04, 0.0
    )
    tilted = HangerModel.calc_signals(
        freqs, 6000.0, 700.0, 950.0, 0.21, 1.2 - 0.3j, 0.04, g
    )

    np.testing.assert_array_equal(
        zero,
        HangerModel.calc_signals(freqs, 6000.0, 700.0, 950.0, 0.21, 1.2 - 0.3j, 0.04),
    )
    ratio = tilted / zero
    np.testing.assert_allclose(ratio.imag, 0.0, atol=1e-14)
    np.testing.assert_allclose(ratio.real, np.exp(g * (freqs - 6000.0)))


def test_disabled_fit_returns_zero_amplitude_slope() -> None:
    freqs = np.linspace(5970.0, 6030.0, 401)
    signals = _transmission_truth(
        freqs,
        freq=6000.0,
        Ql=650.0,
        a0=0.8 + 0.2j,
        edelay=0.0,
        bg_amp_slope=0.0,
    )

    params = TransmissionModel.fit(freqs, signals, edelay=0.0)

    assert params["bg_amp_slope"] == 0.0
    assert set(params) == {
        "freq",
        "fwhm",
        "Ql",
        "a0",
        "edelay",
        "theta0",
        "bg_amp_slope",
        "circle_params",
    }


def test_fixed_noise_refinement_reduces_complex_residual() -> None:
    freq, Ql, Qc_abs = 6053.0, 710.0, 1040.0
    freqs = _homophasal_like_grid(freq, Ql, 45.0, 601)
    clean = _hanger_truth(
        freqs,
        freq=freq,
        Ql=Ql,
        Qc_abs=Qc_abs,
        phi=-0.17,
        a0=1.1 + 0.28j,
        edelay=0.024,
        bg_amp_slope=0.013,
    )
    rng = np.random.default_rng(20260716)
    signals = clean + 0.0015 * (
        rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    )

    sequential = HangerModel.fit(freqs, signals, edelay=0.024)
    refined = HangerModel.fit(freqs, signals, edelay=0.024, fit_bg_amp_slope=True)
    sequential_fit = HangerModel.calc_signals(freqs, **sequential)
    refined_fit = HangerModel.calc_signals(freqs, **refined)

    sequential_rmse = np.sqrt(np.mean(np.abs(sequential_fit - signals) ** 2))
    refined_rmse = np.sqrt(np.mean(np.abs(refined_fit - signals) ** 2))
    assert refined_rmse < 0.35 * sequential_rmse


def test_bound_limited_refinement_warns_and_returns_sequential_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freqs = np.linspace(5970.0, 6030.0, 401)
    signals = _transmission_truth(
        freqs,
        freq=6000.0,
        Ql=650.0,
        a0=0.8 + 0.2j,
        edelay=0.0,
        bg_amp_slope=0.006,
    )
    sequential = TransmissionModel.fit(freqs, signals, edelay=0.0)

    def bound_result(_residual, x0, **_kwargs):
        return SimpleNamespace(
            success=True,
            x=x0,
            fun=np.zeros(2 * len(freqs)),
            active_mask=np.asarray([1, 0, 0, 0, 0]),
        )

    monkeypatch.setattr(
        "zcu_tools.utils.fitting.resonance.base.sp.optimize.least_squares",
        bound_result,
    )

    with pytest.warns(RuntimeWarning, match="parameter bound"):
        result = TransmissionModel.fit(
            freqs, signals, edelay=0.0, fit_bg_amp_slope=True
        )

    assert result == sequential


def test_refined_corrected_geometry_and_quality_factors_are_consistent() -> None:
    freq, Ql, Qc_abs, phi = 6053.0, 760.0, 1080.0, 0.18
    freqs = _homophasal_like_grid(freq, Ql, 40.0, 551)[::-1]
    signals = _hanger_truth(
        freqs,
        freq=freq,
        Ql=Ql,
        Qc_abs=Qc_abs,
        phi=phi,
        a0=1.3 * np.exp(-0.22j),
        edelay=-0.027,
        bg_amp_slope=-0.010,
    )

    params = HangerModel.fit(freqs, signals, fit_bg_amp_slope=True)
    corrected = signals * np.exp(1j * 2.0 * np.pi * freqs * params["edelay"])
    corrected *= np.exp(-params["bg_amp_slope"] * (freqs - params["freq"]))
    expected_circle = fit_circle_params(corrected.real, corrected.imag)

    np.testing.assert_allclose(params["circle_params"], expected_circle, atol=1e-10)
    np.testing.assert_allclose(abs(params["Qc"]), Qc_abs, rtol=0.04)
    expected_qi = 1.0 / (1.0 / params["Ql"] - np.real(1.0 / params["Qc"]))
    np.testing.assert_allclose(params["Qi"], expected_qi)

    phase_data = calc_phase(corrected, expected_circle[0], expected_circle[1])
    order = np.argsort(freqs)
    phase_at_resonance = np.interp(params["freq"], freqs[order], phase_data[order])
    assert abs(params["theta0"] - phase_at_resonance) < 0.03


@pytest.mark.parametrize(
    ("model", "truth_kwargs"),
    [
        (
            HangerModel,
            {"Qc_abs": 980.0, "phi": 0.12},
        ),
        (TransmissionModel, {}),
    ],
)
def test_visualization_uses_corrected_domain_and_background_envelope(
    model: type[HangerModel] | type[TransmissionModel],
    truth_kwargs: dict[str, float],
) -> None:
    freq, Ql, g = 6000.0, 700.0, 0.008
    freqs = np.linspace(5965.0, 6035.0, 401)
    common = dict(
        freq=freq,
        Ql=Ql,
        a0=1.2 * np.exp(0.3j),
        edelay=0.021,
        bg_amp_slope=g,
    )
    truth_fn = _hanger_truth if model is HangerModel else _transmission_truth
    signals = truth_fn(freqs, **truth_kwargs, **common)
    params = model.fit(freqs, signals, edelay=common["edelay"], fit_bg_amp_slope=True)

    fig = model.visualize_fit(freqs, signals, params)
    try:
        iq_ax, phase_ax, magnitude_ax = fig.axes
        iq_data = next(
            line for line in iq_ax.lines if line.get_label() == "corrected data"
        )
        phase_fit = next(
            line for line in phase_ax.lines if line.get_label() == "ideal phase fit"
        )
        raw_data = next(
            line for line in magnitude_ax.lines if line.get_label() == "raw data"
        )
        total_fit = next(
            line for line in magnitude_ax.lines if line.get_label() == "total fit"
        )
        envelope = next(
            line
            for line in magnitude_ax.lines
            if line.get_label() == "multiplicative background envelope"
        )

        corrected = signals * np.exp(1j * 2.0 * np.pi * freqs * params["edelay"])
        corrected *= np.exp(-params["bg_amp_slope"] * (freqs - params["freq"]))
        norm, _ = normalize_signal(corrected, params["circle_params"], params["a0"])
        np.testing.assert_allclose(iq_data.get_xdata(), norm.real)
        np.testing.assert_allclose(iq_data.get_ydata(), norm.imag)
        np.testing.assert_allclose(
            phase_fit.get_ydata(),
            phase_func(freqs, params["freq"], params["Ql"], params["theta0"]),
        )
        np.testing.assert_allclose(raw_data.get_ydata(), np.abs(signals))
        np.testing.assert_allclose(
            total_fit.get_ydata(), np.abs(model.calc_signals(freqs, **params))
        )
        np.testing.assert_allclose(
            envelope.get_ydata(),
            abs(params["a0"])
            * np.exp(params["bg_amp_slope"] * (freqs - params["freq"])),
        )
        assert all(
            "phase background" not in line.get_label() for line in phase_ax.lines
        )
        fit_info = next(
            text for text in magnitude_ax.texts if "MHz$^{-1}$" in text.get_text()
        )
        assert fit_info.get_position()[0] == pytest.approx(0.98)
        assert fit_info.get_horizontalalignment() == "right"
        legend = magnitude_ax.get_legend()
        assert legend is not None
        assert legend._loc == 3  # lower left
    finally:
        plt.close(fig)
