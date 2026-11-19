"""Tests for readout SNR from IQ moments and curve-based early stop."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.acquisition_facts import StopDecision
from zcu_tools.experiment.v2.runtime import SignalBuffer
from zcu_tools.experiment.v2.utils.snr import calc_snr, snr_as_signal, snr_checker


def _moments(
    *,
    separation: float,
    sigma_g: float = 1.0,
    sigma_e: float = 1.0,
    skew_g: float = 0.0,
    skew_e: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    mean = np.array([[0.0, 0.0], [separation, 0.0]], dtype=np.float64)
    cov = np.zeros((2, 2, 2), dtype=np.float64)
    cov[0] = np.diag([sigma_g**2, 1.0])
    cov[1] = np.diag([sigma_e**2, 1.0])
    m3 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    m3[0, 0, 0, 0] = skew_g * sigma_g**3
    m3[1, 0, 0, 0] = skew_e * sigma_e**3
    return mean, cov, m3


def _score(
    *,
    separation: float,
    sigma_g: float = 1.0,
    sigma_e: float = 1.0,
    skew_g: float = 0.0,
    skew_e: float = 0.0,
    skew_penalty: float = 0.0,
) -> float:
    mean, cov, m3 = _moments(
        separation=separation,
        sigma_g=sigma_g,
        sigma_e=sigma_e,
        skew_g=skew_g,
        skew_e=skew_e,
    )
    return float(calc_snr(mean, cov, m3, ge_axis=0, skew_penalty=skew_penalty))


def test_curve_stop_decision_retains_the_threshold_crossing_snr() -> None:
    signals = SignalBuffer((5,), dtype=np.float64)
    signals.set(np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
    check = snr_checker(signals[()], 1.0, lambda values: values, check_interval=None)
    decision = check()
    assert isinstance(decision, StopDecision)
    assert decision.met
    # Five-point quadratic smoothing gives range 5/7 and mean residual 16/35.
    assert decision.value == pytest.approx(25 / 16)
    signals.set(np.zeros(5))
    assert check() == decision


@pytest.mark.parametrize(
    ("curve", "met"),
    [(np.full(5, np.nan), False), (np.array([0.0, 1.0]), True)],
    ids=["nan-projection-keeps-acquiring", "zero-noise-variation-stops"],
)
def test_curve_stop_handles_nan_and_infinite_estimates(
    curve: NDArray[np.float64], met: bool
) -> None:
    signals = SignalBuffer(curve.shape, dtype=np.float64)
    signals.set(np.ones_like(curve))
    estimates: list[float] = []
    check = snr_checker(
        signals.at(),
        1.0,
        lambda values: curve,
        after_check=estimates.append,
        check_interval=None,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        decision = check()
    assert decision.met is met
    assert decision.value is None
    if met:
        assert estimates == [np.inf]
    else:
        assert estimates == []


def test_default_score_is_unbounded_pooled_sigma_snr() -> None:
    assert _score(separation=4.0) == pytest.approx(4.0)
    assert _score(separation=20.0) == pytest.approx(20.0)


def test_negative_skew_penalty_is_rejected() -> None:
    mean, cov, m3 = _moments(separation=1.0)

    with pytest.raises(ValueError, match="skew_penalty"):
        calc_snr(mean, cov, m3, skew_penalty=-0.1)


@pytest.mark.parametrize("penalty", [0.0, 0.75])
def test_snr_from_cubic_sums_projects_covariance_and_skew(penalty: float) -> None:
    # (I, Q) = (2, -3) + t*(3, 4), with g t=(-1, 0, 1) and e t=(2, 2, 5).
    sums = np.array(
        [
            [3, 6, -9, 30, 6, 59, 132, 6, -26, -369],
            [3, 33, 27, 417, 369, 339, 5937, 5553, 5313, 5163],
        ],
        dtype=np.float64,
    )
    # The common scale 5 cancels. Along t, the means differ by 3, sample
    # variances are 1 and 3, and population third central moments are 0 and 2.
    expected = (3 / np.sqrt(2)) / (
        1 + penalty * (np.log(np.sqrt(3)) ** 2 + (2 / (3 * np.sqrt(3))) ** 2)
    )
    assert snr_as_signal(sums, ge_axis=0, skew_penalty=penalty).item() == pytest.approx(
        expected
    )


def _symmetric_readout_sums(
    separation: NDArray[np.float64], *, ge_axis: int = 0
) -> NDArray[np.float64]:
    # g has four shots at (±1, 0), (0, ±1); e is scaled by 2 and shifted in I.
    result = np.zeros((2, *separation.shape, 10), dtype=np.float64)
    result[..., 0] = 4
    result[0, ..., 3] = result[0, ..., 5] = 2
    result[1, ..., 1] = 4 * separation
    result[1, ..., 3] = 8 + 4 * separation**2
    result[1, ..., 5] = 8
    result[1, ..., 6] = 24 * separation + 4 * separation**3
    result[1, ..., 8] = 8 * separation
    return np.moveaxis(result, 0, ge_axis)


@pytest.mark.parametrize("sweep_shape", [(), (1,), (3,), (1, 3), (2, 3)])
@pytest.mark.parametrize("skew_penalty", [0.0, 0.5])
@pytest.mark.parametrize("use_default_axis", [False, True])
def test_snr_as_signal_returns_only_sweep_axes(
    sweep_shape: tuple[int, ...], skew_penalty: float, use_default_axis: bool
) -> None:
    separations = np.arange(1.0, np.prod(sweep_shape) + 1.0).reshape(sweep_shape)
    sums = _symmetric_readout_sums(separations)

    if use_default_axis:
        out = snr_as_signal(sums, skew_penalty=skew_penalty)
    else:
        out = snr_as_signal(sums, ge_axis=0, skew_penalty=skew_penalty)

    assert out.shape == sweep_shape
    assert out.dtype == np.dtype(np.float64)
    expected = separations / np.sqrt(5.0 / 3.0)
    expected /= 1.0 + skew_penalty * np.log(0.5) ** 2
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    ("shape", "ge_axis", "message"),
    [
        ((2, 6), 0, "length 10"),
        ((10,), 0, "length 10"),
        ((2, 10), 1, "ge_axis"),
        ((2, 10), -1, "ge_axis"),
        ((2, 10), -3, "ge_axis"),
        ((3, 10), 0, "state"),
    ],
)
def test_snr_as_signal_rejects_invalid_sums_or_state_axis(
    shape: tuple[int, ...], ge_axis: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        snr_as_signal(np.zeros(shape), ge_axis=ge_axis)


@pytest.mark.parametrize("ge_axis", [1, -2])
def test_snr_as_signal_accepts_an_explicit_nondefault_state_axis(ge_axis: int) -> None:
    sums = _symmetric_readout_sums(np.array([1.0, 2.0, 4.0]))

    np.testing.assert_array_equal(
        snr_as_signal(sums.swapaxes(0, 1), ge_axis=ge_axis), snr_as_signal(sums)
    )


def test_shape_mismatch_penalty_is_gradual_and_monotone() -> None:
    target_snr = 4.0
    ratios = np.array([1.0, 1.5, 2.0, 3.0], dtype=np.float64)
    pooled = np.sqrt(0.5 * (1.0 + ratios**2))
    separations = target_snr * pooled
    pure_scores = np.array(
        [
            _score(separation=float(sep), sigma_e=float(ratio))
            for sep, ratio in zip(separations, ratios, strict=True)
        ],
        dtype=np.float64,
    )
    penalized = np.array(
        [
            _score(separation=float(sep), sigma_e=float(ratio), skew_penalty=0.5)
            for sep, ratio in zip(separations, ratios, strict=True)
        ],
        dtype=np.float64,
    )
    expected = target_snr / (1.0 + 0.5 * np.log(ratios) ** 2)

    assert pure_scores == pytest.approx(np.full_like(ratios, target_snr))
    assert penalized == pytest.approx(expected)
    assert np.all(np.diff(penalized) < 0.0)


def test_skew_penalty_downweights_one_sided_skew_gradually() -> None:
    skews = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    pure_scores = np.array(
        [_score(separation=4.0, skew_e=float(skew)) for skew in skews],
        dtype=np.float64,
    )
    mild_penalty = np.array(
        [
            _score(separation=4.0, skew_e=float(skew), skew_penalty=0.25)
            for skew in skews
        ],
        dtype=np.float64,
    )
    strong_penalty = np.array(
        [
            _score(separation=4.0, skew_e=float(skew), skew_penalty=1.0)
            for skew in skews
        ],
        dtype=np.float64,
    )

    assert pure_scores == pytest.approx(np.full_like(skews, 4.0))
    assert mild_penalty == pytest.approx(4.0 / (1.0 + 0.25 * skews**2))
    assert np.all(np.diff(mild_penalty) < 0.0)
    assert np.all(strong_penalty[1:] < mild_penalty[1:])


def test_shape_clean_candidate_beats_same_snr_skewed_candidate_when_penalized() -> None:
    clean = _score(separation=4.0, skew_penalty=1.0)
    skewed_without_penalty = _score(separation=4.0, skew_e=2.0)
    skewed_with_penalty = _score(separation=4.0, skew_e=2.0, skew_penalty=1.0)

    assert skewed_without_penalty == pytest.approx(clean)
    assert skewed_with_penalty == pytest.approx(0.8)
    assert skewed_with_penalty < clean


def test_score_is_invariant_under_ge_label_swap() -> None:
    mean, cov, m3 = _moments(
        separation=4.0,
        sigma_g=1.2,
        sigma_e=2.0,
        skew_g=0.25,
        skew_e=-0.75,
    )
    swapped_mean = np.stack([mean[1], mean[0]], axis=0)
    swapped_cov = np.stack([cov[1], cov[0]], axis=0)
    swapped_m3 = np.stack([m3[1], m3[0]], axis=0)

    score = calc_snr(mean, cov, m3, skew_penalty=0.75)
    swapped = calc_snr(swapped_mean, swapped_cov, swapped_m3, skew_penalty=0.75)

    assert float(score) == pytest.approx(float(swapped))
