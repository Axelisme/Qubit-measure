import numpy as np
import pytest
from zcu_tools.analysis import fitting
from zcu_tools.analysis.failure import AnalysisFailed


def test_weighted_fringes_share_frequency_but_resolve_distinct_envelopes() -> None:
    delays = np.arange(121, dtype=np.float64) * 0.05
    truth = ((0.08, 0.58, 21.0, 2.0), (-0.14, 0.38, 67.0, 8.0))
    rng = np.random.default_rng(206)
    sigmas = (
        np.where(delays < 3.0, 0.0007, 0.03),
        np.where(delays < 2.0, 0.02, 0.0012),
    )
    observations = tuple(
        offset
        + amp
        * np.exp(-delays / decay)
        * np.cos(2 * np.pi * 1.17 * delays + np.deg2rad(phase))
        + rng.normal(0.0, sigma)
        for (offset, amp, phase, decay), sigma in zip(truth, sigmas, strict=True)
    )

    result = fitting.fit_weighted_decay_fringe(delays, observations, sigmas)

    assert result.shared.diagnostics.valid
    assert result.frequency == pytest.approx(1.17, abs=0.005)
    assert np.isfinite(result.frequency_se) and result.frequency_se > 0
    assert result.frequency_se == pytest.approx(
        np.sqrt(result.shared.variance("frequency"))
    )
    assert result.shared.covariance.shape == (9, 9)
    advance = np.array([0.01, 1.06])
    squared_error = 0.0
    for trace, (offset, amp, phase, decay), obs, sigma in zip(
        result.traces, truth, observations, sigmas, strict=True
    ):
        assert trace.frequency == result.frequency
        assert trace.decay_time == pytest.approx(decay, rel=0.05)
        assert trace.offset == pytest.approx(offset, abs=0.01)
        # Compare the modeled curves, since amplitude/phase can trade a sign and 180°.
        expected = offset + amp * np.exp(-advance / decay) * np.cos(
            2 * np.pi * 1.17 * advance + np.deg2rad(phase)
        )
        np.testing.assert_allclose(trace.evaluate(advance), expected, atol=0.01, rtol=0)
        assert trace.quality.relative_parameter_errors["frequency"] == pytest.approx(
            result.frequency_se / result.frequency
        )
        squared_error += float(np.sum(((obs - trace.curve) / sigma) ** 2))
    assert result.shared.diagnostics.reduced_chi_square == pytest.approx(
        squared_error / (2 * len(delays) - 9)
    )


def test_weighted_fringe_accepts_a_single_trace() -> None:
    delays = np.arange(121, dtype=np.float64) * 0.05
    expected = 0.2 - 0.5 * np.exp(-delays / 3.0) * np.cos(2 * np.pi * 1.51 * delays)
    sigma = np.full_like(delays, 0.001)
    obs = expected + np.random.default_rng(222).normal(0.0, sigma)

    result = fitting.fit_weighted_decay_fringe(delays, (obs,), (sigma,))

    assert result.shared.diagnostics.valid
    assert result.frequency == pytest.approx(1.51, abs=0.005)
    (trace,) = result.traces
    np.testing.assert_allclose(trace.curve, expected, rtol=0, atol=0.002)
    assert trace.decay_time == pytest.approx(3.0, rel=0.05)


def test_weighted_fringe_rejects_invalid_weights_and_unidentifiable_traces() -> None:
    delays = np.arange(121, dtype=np.float64) * 0.05
    obs = np.cos(2 * np.pi * 1.17 * delays)
    sigma = np.full_like(delays, 0.01)
    invalid_sigma = sigma.copy()
    invalid_sigma[12] = 0.0

    with pytest.raises(ValueError, match="sigmas.*positive finite"):
        fitting.fit_weighted_decay_fringe(delays, (obs,), (invalid_sigma,))
    with pytest.raises(AnalysisFailed, match="contrast"):
        fitting.fit_weighted_decay_fringe(
            delays, (obs, np.ones_like(obs)), (sigma, sigma)
        )
