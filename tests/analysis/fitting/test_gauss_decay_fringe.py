"""Gaussian-envelope fringes through the public numerical fitting seam."""

import numpy as np
import pytest
from zcu_tools.analysis import fitting

from tests.analysis.fitting._fit_contract import assert_frozen_fields


@pytest.mark.parametrize("fixed_phase", (None, 0.0))
def test_gaussian_fringe_named_result_preserves_frequency_and_quality(
    fixed_phase: float | None,
) -> None:
    xs = np.linspace(0.0, 24.0, 240)
    phase = 37.0 if fixed_phase is None else fixed_phase
    ys = 0.23 + 0.8 * np.exp(-((xs / 6.0) ** 2)) * np.cos(
        2 * np.pi * (0.7 * xs + phase / 360)
    )
    ys += np.random.default_rng(42).normal(0.0, 0.001, xs.size)
    result = fitting.fit_gauss_decay_fringe(
        xs,
        ys,
        fit_params=(0.23, 0.8, 0.7, phase, 6.0),
        fixedparams=(None, None, None, fixed_phase, None),
    )
    time, error, freq, freq_err, curve, (params, cov) = result

    assert result.decay_time == time == pytest.approx(6.0, rel=0.01)
    assert isinstance(result, fitting.GaussDecayFringeFit)
    assert result.decay_time_err == error == pytest.approx(np.sqrt(cov[4, 4]))
    assert result.freq == freq == pytest.approx(0.7, rel=0.001)
    assert result.freq_err == freq_err == pytest.approx(np.sqrt(cov[2, 2]))
    assert result.curve is curve
    assert result.params is params
    assert result.cov is cov
    expected = fitting.compute_fit_quality(
        ys,
        curve,
        dict(zip(("y0", "yscale", "freq", "phase", "decay_time"), params, strict=True)),
        cov,
        decay_bounds=fitting.decay_time_bounds(xs),
        fixed_parameters=() if fixed_phase is None else ("phase",),
    )
    assert result.quality == expected
    assert result.quality.invalid == ()
    assert result.quality.r2 is not None and result.quality.r2 > 0.999
    assert error > 0.0 and freq_err > 0.0
    assert_frozen_fields(
        result,
        (
            "decay_time",
            "decay_time_err",
            "freq",
            "freq_err",
            "params",
            "cov",
            "curve",
            "quality",
        ),
    )


def test_gaussian_fringe_recovers_zero_centered_envelope_and_frequency() -> None:
    xs = np.linspace(0.3, 24.0, 240)
    expected = 0.23 + 0.8 * np.exp(-((xs / 6.0) ** 2)) * np.cos(
        2 * np.pi * (0.7 * xs + 37.0 / 360)
    )
    fit = getattr(fitting, "fit_gauss_decay_fringe", None)
    assert fit is not None, "Public Gaussian-envelope fringe fitting is missing"
    lifetime, error, frequency, frequency_error, curve, _ = fit(xs, expected)
    assert lifetime == pytest.approx(6.0, rel=1e-3)
    assert frequency == pytest.approx(0.7, rel=1e-3)
    assert np.isfinite(error) and np.isfinite(frequency_error)
    np.testing.assert_allclose(curve, expected, atol=1e-6)
