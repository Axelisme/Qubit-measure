"""Tests for MomentTracker-based readout SNR helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.v2.utils.snr import calc_snr, snr_as_signal


@dataclass
class _FakeTracker:
    """Minimal stand-in exposing the MomentTracker fields used by snr_as_signal."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    third_moment: NDArray[np.float64]


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


def test_default_score_is_unbounded_pooled_sigma_snr() -> None:
    assert _score(separation=4.0) == pytest.approx(4.0)
    assert _score(separation=20.0) == pytest.approx(20.0)


def test_negative_skew_penalty_is_rejected() -> None:
    mean, cov, m3 = _moments(separation=1.0)

    with pytest.raises(ValueError, match="skew_penalty"):
        calc_snr(mean, cov, m3, skew_penalty=-0.1)


def test_snr_as_signal_broadcasts_sweep_axis() -> None:
    separations = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    moments = [_moments(separation=float(sep)) for sep in separations]
    mean = np.stack([item[0] for item in moments], axis=0)
    cov = np.stack([item[1] for item in moments], axis=0)
    m3 = np.stack([item[2] for item in moments], axis=0)
    raw = [_FakeTracker(mean=mean, covariance=cov, third_moment=m3)]

    out = snr_as_signal(raw, ge_axis=1)  # type: ignore[arg-type]

    assert out.shape == (3,)
    assert out == pytest.approx(separations)


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
