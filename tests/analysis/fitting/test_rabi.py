from collections.abc import Callable

import numpy as np
import pytest
from zcu_tools.analysis.fitting import compute_fit_quality
from zcu_tools.analysis.fitting.base import cosfunc, decaycos
from zcu_tools.analysis.fitting.rabi import fit_rabi

from tests.analysis.fitting._fit_contract import assert_frozen_fields


@pytest.mark.parametrize("invalid_call", [fit_rabi], ids=["fit_rabi"])
@pytest.mark.parametrize("decay", [True, False], ids=["true", "false"])
def test_fit_rabi_rejects_short_positional_boolean(
    invalid_call: Callable[..., object], decay: bool
) -> None:
    xs = np.linspace(0, 4, 400)
    signals = cosfunc(xs, 0.0, 1.0, 0.5, 0.0)
    with pytest.raises(TypeError, match="positional"):
        invalid_call(xs, signals, decay)


def test_fit_rabi_default_matches_explicit_no_decay() -> None:
    xs = np.linspace(0, 4, 400)
    signals = cosfunc(xs, 0.0, 1.0, 0.5, 0.0)
    default = fit_rabi(xs, signals)
    explicit = fit_rabi(xs, signals, decay=False, init_phase=None)
    np.testing.assert_allclose(default[:6], explicit[:6])
    np.testing.assert_allclose(default[6], signals, atol=1e-7)
    np.testing.assert_allclose(default[6], explicit[6])


@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("fixed_phase", [None, 0.0])
def test_rabi_named_result_preserves_pulse_coordinates_and_quality(
    decay: bool,
    fixed_phase: float | None,
) -> None:
    xs = np.linspace(0.0, 6.0, 160)
    ys = 0.2 + 0.8 * np.cos(2 * np.pi * 0.5 * xs) * (np.exp(-xs / 10.0) if decay else 1)
    ys += np.random.default_rng(42).normal(0.0, 0.001, xs.size)
    result = fit_rabi(xs, ys, decay=decay, init_phase=fixed_phase)
    pi_x, pi_err, pi2_x, pi2_err, freq, freq_err, curve, (params, cov) = result

    assert isinstance(result, tuple)
    assert result.pi_x == pi_x == pytest.approx(1.0, rel=0.01)
    assert result.pi_x_err == pi_err
    assert result.pi2_x == pi2_x == pytest.approx(0.5, rel=0.01)
    assert result.pi2_x_err == pi2_err
    assert result.freq == freq == pytest.approx(0.5, rel=0.01)
    assert result.freq_err == freq_err
    assert result.curve is curve
    assert result.params is params
    assert result.cov is cov
    names = (
        ("y0", "yscale", "freq", "phase", "decay_time")
        if decay
        else ("y0", "yscale", "freq", "phase")
    )
    expected = compute_fit_quality(
        ys,
        curve,
        dict(zip(names, params, strict=True)),
        cov,
        fixed_parameters=() if fixed_phase is None else ("phase",),
    )
    assert result.quality == expected
    assert result.quality.invalid == ()
    assert pi_err > 0.0 and pi2_err > 0.0 and freq_err > 0.0
    assert_frozen_fields(
        result, ("pi_x", "pi2_x", "freq", "curve", "params", "cov", "quality")
    )


def test_fit_rabi_no_decay():
    xs = np.linspace(0, 4, 400)
    y0, yscale, freq, phase = 0.0, 1.0, 0.5, 0.0
    ys = cosfunc(xs, y0, yscale, freq, phase)
    pi_x, pi_x_err, pi2_x, pi2_x_err, fit_freq, freq_err, _, _ = fit_rabi(
        xs, ys, decay=False
    )
    assert abs(fit_freq - freq) < 1e-3
    assert abs(pi_x - 1.0) < 1e-2
    assert abs(pi2_x - 0.5) < 1e-2
    assert freq_err >= 0
    assert pi_x_err >= 0
    assert pi2_x_err >= 0


def test_fit_rabi_with_decay():
    xs = np.linspace(0, 6, 400)
    y0, yscale, freq, phase, tau = 0.0, 1.0, 0.4, 0.0, 10.0
    ys = decaycos(xs, y0, yscale, freq, phase, tau)
    pi_x, pi_x_err, _, pi2_x_err, fit_freq, freq_err, _, _ = fit_rabi(
        xs, ys, decay=True
    )
    assert abs(fit_freq - freq) / freq < 5e-2
    assert abs(pi_x - 1.25) < 5e-2
    assert freq_err >= 0
    assert pi_x_err >= 0
    assert pi2_x_err >= 0


def test_fit_rabi_retains_parameter_tuple_shape_for_both_models():
    xs = np.linspace(0, 6, 400)
    for decay in (False, True):
        expected_params = (0.2, 0.8, 0.5, 0.0, 10.0) if decay else (0.2, 0.8, 0.5, 0.0)
        model = decaycos if decay else cosfunc
        signals = model(xs, *expected_params)

        pi_x, pi_x_err, pi2_x, pi2_x_err, freq, freq_err, fitted, fit_result = fit_rabi(
            xs, signals, decay=decay, init_phase=0.0
        )
        params, covariance = fit_result

        assert isinstance(params, tuple)
        assert len(params) == (5 if decay else 4)
        np.testing.assert_allclose(params, expected_params, rtol=1e-5, atol=1e-7)
        assert covariance.shape == (len(params), len(params))
        assert np.all(np.isfinite(covariance))
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        np.testing.assert_allclose(fitted, signals, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose((pi_x, pi2_x, freq), (1.0, 0.5, 0.5), atol=1e-6)
        assert 0 <= pi_x_err < 1e-6
        assert 0 <= pi2_x_err < 1e-6
        assert 0 <= freq_err < 1e-6
