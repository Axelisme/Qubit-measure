"""Tests for snr_as_signal with the disc*sym SNR metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from zcu_tools.experiment.v2.tracker import MomentTracker
from zcu_tools.experiment.v2.utils.snr import calc_snr, snr_as_signal


# ---------- helpers ----------


@dataclass
class _FakeTracker:
    """Minimal stand-in for MomentTracker exposing the attributes snr_as_signal reads."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    third_moment: NDArray[np.float64]


def _third_moment(samples: np.ndarray) -> np.ndarray:
    """Population third central moment for one set of IQ samples.

    samples: (N, 2) → out: (2, 2, 2).
    """
    centered = samples - samples.mean(axis=0, keepdims=True)
    return np.einsum("mi,mj,mk->ijk", centered, centered, centered) / centered.shape[0]


def _stats_from_samples(samples_ge: np.ndarray):
    """samples_ge: shape (2, N, 2) — (ge, shots, IQ).

    Returns ``[fake_tracker]`` matching snr_as_signal's contract.
    """
    mean = samples_ge.mean(axis=1)  # (2, 2)
    cov = np.stack([np.cov(s, rowvar=False) for s in samples_ge], axis=0)  # (2, 2, 2)
    m3 = np.stack([_third_moment(s) for s in samples_ge], axis=0)  # (2, 2, 2, 2)
    return [_FakeTracker(mean=mean, covariance=cov, third_moment=m3)]


def _isotropic(center, sigma, n, rng):
    return rng.normal(loc=center, scale=sigma, size=(n, 2))


def _rotated_elliptical(center, sx, sy, angle, n, rng):
    pts = rng.normal(size=(n, 2)) * np.array([sx, sy])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.asarray(center)


def _score(samples_ge):
    return float(
        snr_as_signal(cast(list[MomentTracker], _stats_from_samples(samples_ge)))
    )


def _third_gaussian_score(
    separation: float,
    third_strength: float,
    bisector_offset: float,
    rng: np.random.Generator,
    n: int = 8000,
    sigma: float = 1.0,
    mode: Literal["e_only", "ge_both"] = "e_only",
) -> float:
    if mode == "e_only":
        n_third_g = 0
        n_third_e = int(round(n * third_strength))
    elif mode == "ge_both":
        n_third_each = int(round(n * third_strength / 2.0))
        n_third_g = n_third_each
        n_third_e = n_third_each
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    g_main = _isotropic([0.0, 0.0], sigma, n - n_third_g, rng)
    e_main = _isotropic([separation, 0.0], sigma, n - n_third_e, rng)
    if n_third_g > 0:
        g_third = _isotropic([separation / 2.0, bisector_offset], sigma, n_third_g, rng)
        g = np.concatenate([g_main, g_third], axis=0)
    else:
        g = g_main
    if n_third_e > 0:
        e_third = _isotropic([separation / 2.0, bisector_offset], sigma, n_third_e, rng)
        e = np.concatenate([e_main, e_third], axis=0)
    else:
        e = e_main
    return _score(np.stack([g, e]))


# ---------- tests ----------


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_ideal_two_gaussians_well_separated_high_score(rng):
    n = 5000
    g = _isotropic([0.0, 0.0], 1.0, n, rng)
    e = _isotropic([10.0, 0.0], 1.0, n, rng)
    score = _score(np.stack([g, e]))
    assert score > 0.9


def test_ideal_two_gaussians_overlapping_low_score(rng):
    n = 5000
    g = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_overlap = _isotropic([0.3, 0.0], 1.0, n, rng)
    e_sep = _isotropic([10.0, 0.0], 1.0, n, rng)
    overlap_score = _score(np.stack([g, e_overlap]))
    sep_score = _score(np.stack([g, e_sep]))
    assert overlap_score < sep_score


def test_identical_centers_near_zero(rng):
    n = 5000
    g = _isotropic([0.0, 0.0], 1.0, n, rng)
    e = _isotropic([0.0, 0.0], 1.0, n, rng)
    score = _score(np.stack([g, e]))
    assert score == pytest.approx(0.0, abs=0.1)


def test_separation_monotonicity(rng):
    n = 5000
    scores = []
    for d in [2.0, 5.0, 10.0, 20.0]:
        g = _isotropic([0.0, 0.0], 1.0, n, rng)
        e = _isotropic([d, 0.0], 1.0, n, rng)
        scores.append(_score(np.stack([g, e])))
    assert all(scores[i] <= scores[i + 1] + 0.03 for i in range(len(scores) - 1))
    assert scores[0] < scores[-1]


D = 1.2  # near-threshold center separation in units of σ


def test_horizontal_ellipse_penalized_vs_isotropic(rng):
    """σ along separation axis grows → disc shrinks."""
    n = 8000
    g_iso = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_iso = _isotropic([D, 0.0], 1.0, n, rng)
    iso_score = _score(np.stack([g_iso, e_iso]))

    g_el = _rotated_elliptical([0.0, 0.0], 2.0, 0.3, 0.0, n, rng)
    e_el = _rotated_elliptical([D, 0.0], 2.0, 0.3, 0.0, n, rng)
    el_score = _score(np.stack([g_el, e_el]))

    assert el_score < iso_score
    assert el_score < iso_score * 0.95


def test_shape_mismatch_between_g_and_e_penalized(rng):
    """σ_g ≠ σ_e → sym factor drops; total score lower."""
    n = 6000
    g_same = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_same = _isotropic([D, 0.0], 1.0, n, rng)
    matched = _score(np.stack([g_same, e_same]))

    g_diff = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_diff = _rotated_elliptical([D, 0.0], 3.0, 0.3, 0.0, n, rng)
    mismatched = _score(np.stack([g_diff, e_diff]))

    assert mismatched < matched
    assert mismatched < matched * 0.9


def test_ge_distance_increases_snr_without_third_interference(rng):
    sigma = 1.0
    separations = [0.5 * sigma, 1.0 * sigma, 2.0 * sigma, 3.0 * sigma]
    scores = [
        _third_gaussian_score(sep, 0.0, 0.0, rng, sigma=sigma) for sep in separations
    ]
    assert all(scores[i] <= scores[i + 1] + 0.02 for i in range(len(scores) - 1))
    assert scores[-1] > scores[0]


def test_bisector_strength_sweep_at_2sigma(rng):
    """Stronger one-sided 3rd Gaussian should reduce the score."""
    sigma = 1.0
    separation = 2.0 * sigma
    strengths = [0.0, 0.1, 0.2, 0.35]

    # Use offset=0 (projection-axis only — perpendicular offset is invisible
    # to a 1D-projected skew metric).
    scores = [
        _third_gaussian_score(separation, s, 0.0, rng, sigma=sigma) for s in strengths
    ]

    assert all(scores[i + 1] <= scores[i] + 0.03 for i in range(len(strengths) - 1))
    assert scores[-1] < scores[0] - 0.05


def test_e_only_third_contamination_lower_than_symmetric_ge(rng):
    sigma = 1.0
    separation = 2.0 * sigma
    strength = 0.2
    snr_e_only = _third_gaussian_score(
        separation, strength, 0.0, rng, sigma=sigma, mode="e_only"
    )
    snr_ge_both = _third_gaussian_score(
        separation, strength, 0.0, rng, sigma=sigma, mode="ge_both"
    )
    assert snr_e_only < snr_ge_both


def test_symmetry_one_for_bisector_symmetric_shapes(rng):
    n = 6000
    g = _isotropic([0.0, 0.0], 1.0, n, rng)
    e = _isotropic([10.0, 0.0], 1.0, n, rng)
    iso_score = _score(np.stack([g, e]))

    g_a = _rotated_elliptical([0.0, 0.0], 1.5, 0.7, 0.0, n, rng)
    e_a = _rotated_elliptical([10.0, 0.0], 1.5, 0.7, 0.0, n, rng)
    aligned_score = _score(np.stack([g_a, e_a]))

    assert iso_score > 0.9
    assert aligned_score > 0.8


def test_sweep_axis_broadcasting(rng):
    n = 3000
    n_sweep = 4

    means, covs, m3s = [], [], []
    for k in range(n_sweep):
        d = 2.0 + 3.0 * k
        g = _isotropic([0.0, 0.0], 1.0, n, rng)
        e = _isotropic([d, 0.0], 1.0, n, rng)
        samples = np.stack([g, e])
        means.append(samples.mean(axis=1))
        covs.append(np.stack([np.cov(s, rowvar=False) for s in samples]))
        m3s.append(np.stack([_third_moment(s) for s in samples]))

    mean = np.stack(means, axis=0)  # (sweep, ge, IQ)
    cov = np.stack(covs, axis=0)  # (sweep, ge, IQ, IQ)
    m3 = np.stack(m3s, axis=0)  # (sweep, ge, 2, 2, 2)

    raw = cast(
        list[MomentTracker],
        [_FakeTracker(mean=mean, covariance=cov, third_moment=m3)],
    )
    out = snr_as_signal(raw, ge_axis=1)
    assert out.shape == (n_sweep,)
    assert np.all(np.diff(out) >= -0.04)


def test_symmetric_leakage_keeps_symmetry_factor_high(rng):
    """Both g and e leak symmetrically — sym stays near 1, disc drops
    because projected σ inflates."""
    n = 8000
    sep = 6.0

    g = np.concatenate(
        [
            _isotropic([0.0, 0.0], 1.0, int(n * 0.8), rng),
            _isotropic([sep, 0.0], 1.0, int(n * 0.2), rng),
        ]
    )
    e = np.concatenate(
        [
            _isotropic([sep, 0.0], 1.0, int(n * 0.8), rng),
            _isotropic([0.0, 0.0], 1.0, int(n * 0.2), rng),
        ]
    )
    stats = _stats_from_samples(np.stack([g, e]))[0]
    mean_d = stats.mean
    cov_d = stats.covariance
    m3_d = stats.third_moment

    # Decompose: sym must be ≈1 for mirror-symmetric leakage.
    axis_vec = mean_d[1] - mean_d[0]
    axis = axis_vec / np.linalg.norm(axis_vec)
    sigma_g = float(np.sqrt(axis @ cov_d[0] @ axis))
    sigma_e = float(np.sqrt(axis @ cov_d[1] @ axis))
    skew_g = np.einsum("i,j,k,ijk->", axis, axis, axis, m3_d[0]) / sigma_g**3
    skew_e = np.einsum("i,j,k,ijk->", axis, axis, axis, m3_d[1]) / sigma_e**3
    d_sym = 0.5 * (skew_g + skew_e) ** 2 + 0.5 * np.log(sigma_g / sigma_e) ** 2
    sym = float(np.exp(-d_sym))

    assert sym > 0.95  # mirror-symmetric → sym ≈ 1

    sym_score = float(calc_snr(mean_d, cov_d, m3_d, ge_axis=0))

    # Clean baseline with the same inflated σ (so same disc) but no leakage.
    inflated_var = float(np.cov(g, rowvar=False)[0, 0])
    g_baseline = _isotropic([0.0, 0.0], np.sqrt(inflated_var), n, rng)
    e_baseline = _isotropic([sep, 0.0], np.sqrt(inflated_var), n, rng)
    baseline = _score(np.stack([g_baseline, e_baseline]))

    # sym≈1 both sides; baseline's Δμ is the full sep, leakage case's Δμ
    # is smaller → disc drops → sym_score < baseline.
    assert sym_score < baseline


def test_score_bounded_in_zero_one(rng):
    """New metric is bounded by [0, 1] with tiny MC overshoot."""
    n = 2000
    cases = [
        (_isotropic([0, 0], 1.0, n, rng), _isotropic([0, 0], 1.0, n, rng)),
        (_isotropic([0, 0], 1.0, n, rng), _isotropic([20, 0], 1.0, n, rng)),
        (
            _rotated_elliptical([0, 0], 2.0, 0.2, 0.0, n, rng),
            _rotated_elliptical([D, 0], 2.0, 0.2, 0.0, n, rng),
        ),
    ]
    for g, e in cases:
        s = _score(np.stack([g, e]))
        assert -0.01 <= s <= 1.01


def test_skewness_antisymmetric_under_ge_swap(rng):
    """Swapping G↔E labels reflects axis, so skew_g,e swap and negate;
    their sum is invariant → sym invariant → total score invariant
    (disc already symmetric in G/E)."""
    n = 8000
    # deliberately asymmetric contamination
    g = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_main = _isotropic([4.0, 0.0], 1.0, int(n * 0.7), rng)
    e_third = _isotropic([2.0, 0.0], 1.0, int(n * 0.3), rng)
    e = np.concatenate([e_main, e_third], axis=0)

    score_ge = _score(np.stack([g, e]))
    score_eg = _score(np.stack([e, g]))
    assert score_ge == pytest.approx(score_eg, abs=1e-9)


def test_sym_one_for_perfectly_symmetric_bimodal(rng):
    """Each side is a symmetric bimodal (mirror-symmetric about its own
    mean) → skew=0 on both → sym ≈ 1."""
    n = 20000
    sep = 6.0
    # g = symmetric bimodal around 0: peaks at ±a
    a = 0.5
    g = np.concatenate(
        [
            _isotropic([-a, 0.0], 0.3, n // 2, rng),
            _isotropic([a, 0.0], 0.3, n // 2, rng),
        ]
    )
    e = np.concatenate(
        [
            _isotropic([sep - a, 0.0], 0.3, n // 2, rng),
            _isotropic([sep + a, 0.0], 0.3, n // 2, rng),
        ]
    )

    stats = _stats_from_samples(np.stack([g, e]))[0]
    mean_d, cov_d, m3_d = stats.mean, stats.covariance, stats.third_moment
    axis = (mean_d[1] - mean_d[0]) / np.linalg.norm(mean_d[1] - mean_d[0])
    sigma_g = float(np.sqrt(axis @ cov_d[0] @ axis))
    sigma_e = float(np.sqrt(axis @ cov_d[1] @ axis))
    skew_g = np.einsum("i,j,k,ijk->", axis, axis, axis, m3_d[0]) / sigma_g**3
    skew_e = np.einsum("i,j,k,ijk->", axis, axis, axis, m3_d[1]) / sigma_e**3
    sym = float(
        np.exp(-0.5 * (skew_g + skew_e) ** 2 - 0.5 * np.log(sigma_g / sigma_e) ** 2)
    )
    assert sym > 0.98
