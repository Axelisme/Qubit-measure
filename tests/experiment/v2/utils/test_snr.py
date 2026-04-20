"""Tests for snr_as_signal — verifies it rewards well-separated isotropic
two-Gaussian signals and penalizes asymmetry about the g-e bisector.

Current metric: erf(||Δcenter|| / (sqrt(32) σ)) × symmetry, where
symmetry = BC(reflect_g_across_bisector, e), the Bhattacharyya
coefficient between e's gaussian and g's gaussian reflected across the
perpendicular bisector of the g-e center segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from zcu_tools.experiment.v2.utils.snr import snr_as_signal


# ---------- helpers ----------


@dataclass
class _FakeTracker:
    """Minimal stand-in for PCATracker exposing the attributes snr_as_signal reads."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    leader_center: NDArray[np.float64]


def _stats_from_samples(samples_ge: np.ndarray):
    """samples_ge: shape (2, N, 2) — (ge, shots, IQ).

    Returns ``[fake_tracker]`` matching the runtime contract where the
    raw is ``Sequence[PCATracker]`` and snr_as_signal reads
    ``raw[0].leader_center`` / ``raw[0].covariance``.
    """
    avg = samples_ge.mean(axis=1)  # (2, 2)
    med = np.median(samples_ge, axis=1)  # (2, 2)
    cov = np.stack([np.cov(s, rowvar=False) for s in samples_ge], axis=0)  # (2, 2, 2)
    return [_FakeTracker(mean=avg, covariance=cov, leader_center=med)]


def _isotropic(center, sigma, n, rng):
    return rng.normal(loc=center, scale=sigma, size=(n, 2))


def _rotated_elliptical(center, sx, sy, angle, n, rng):
    pts = rng.normal(size=(n, 2)) * np.array([sx, sy])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.asarray(center)


def _score(samples_ge):
    return float(snr_as_signal(_stats_from_samples(samples_ge)))


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
    e = _isotropic([0.3, 0.0], 1.0, n, rng)
    score = _score(np.stack([g, e]))
    assert score < 0.2


def test_identical_centers_near_zero(rng):
    n = 5000
    g = _isotropic([0.0, 0.0], 1.0, n, rng)
    e = _isotropic([0.0, 0.0], 1.0, n, rng)
    score = _score(np.stack([g, e]))
    assert score < 0.05


def test_separation_monotonicity(rng):
    """Larger center distance → higher score (all else equal)."""
    n = 5000
    scores = []
    for d in [2.0, 5.0, 10.0, 20.0]:
        g = _isotropic([0.0, 0.0], 1.0, n, rng)
        e = _isotropic([d, 0.0], 1.0, n, rng)
        scores.append(_score(np.stack([g, e])))
    assert all(scores[i] <= scores[i + 1] + 1e-3 for i in range(len(scores) - 1))
    assert scores[0] < scores[-1]


D = 1.2  # near-threshold center separation in units of σ


def test_horizontal_ellipse_penalized_vs_isotropic(rng):
    """Ellipses elongated along the separation axis. Pooled-cov σ inflates
    along the long axis, shrinking the erf factor."""
    n = 8000
    g_iso = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_iso = _isotropic([D, 0.0], 1.0, n, rng)
    iso_score = _score(np.stack([g_iso, e_iso]))

    g_el = _rotated_elliptical([0.0, 0.0], 2.0, 0.3, 0.0, n, rng)
    e_el = _rotated_elliptical([D, 0.0], 2.0, 0.3, 0.0, n, rng)
    el_score = _score(np.stack([g_el, e_el]))

    # σ inflates from 1 to ~1.43 → erf argument shrinks ~30%. Same shape
    # so consistency≈1. Penalty is moderate (not the >10x of the legacy
    # metric — that was an explicit (λ_min/λ_max) factor we dropped).
    assert el_score < iso_score
    assert el_score < iso_score * 0.85


def test_shape_mismatch_between_g_and_e_penalized(rng):
    """g tight & round, e horizontally elongated, same 1.2σ center
    distance. Bhattacharyya consistency must pull the score down."""
    n = 6000
    g_same = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_same = _isotropic([D, 0.0], 1.0, n, rng)
    matched = _score(np.stack([g_same, e_same]))

    g_diff = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_diff = _rotated_elliptical([D, 0.0], 3.0, 0.3, 0.0, n, rng)
    mismatched = _score(np.stack([g_diff, e_diff]))

    assert mismatched < matched
    assert mismatched < matched * 0.6


def test_third_gaussian_at_midpoint_penalized(rng):
    """Contaminate e with a third cluster at the midpoint between g and e.
    Median is robust so peak_contrast barely moves, but inflated cov
    shrinks the erf term — score must drop."""
    n = 6000
    g = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_clean = _isotropic([D, 0.0], 1.0, n, rng)
    clean = _score(np.stack([g, e_clean]))

    e_main = _isotropic([D, 0.0], 1.0, int(n * 0.7), rng)
    e_third = _isotropic([D / 2, 0.0], 1.0, int(n * 0.3), rng)
    e_contam = np.concatenate([e_main, e_third], axis=0)
    contaminated = _score(np.stack([g, e_contam]))

    assert contaminated < clean
    assert contaminated < clean * 0.95


def test_symmetry_one_for_bisector_symmetric_shapes(rng):
    """Shapes that are themselves mirror-symmetric about the bisector
    (axis-aligned ellipses with the separation along an axis) should
    score close to 1 at large separation."""
    n = 6000
    g = _isotropic([0.0, 0.0], 1.0, n, rng)
    e = _isotropic([10.0, 0.0], 1.0, n, rng)
    iso_score = _score(np.stack([g, e]))

    # Identical axis-aligned ellipses — symmetric about the y-axis bisector.
    g_a = _rotated_elliptical([0.0, 0.0], 1.5, 0.7, 0.0, n, rng)
    e_a = _rotated_elliptical([10.0, 0.0], 1.5, 0.7, 0.0, n, rng)
    aligned_score = _score(np.stack([g_a, e_a]))

    assert iso_score > 0.9
    assert aligned_score > 0.9


def test_symmetry_drops_for_rotated_identical_ellipses(rng):
    """Identical-shape ellipses tilted at an angle — Σ_g == Σ_e but the
    pair is NOT mirror-symmetric about the perpendicular bisector, so
    the new metric should penalize them. This is a deliberate semantic
    change from the previous shape-consistency metric."""
    n = 6000
    g_a = _rotated_elliptical([0.0, 0.0], 1.5, 0.7, 0.0, n, rng)
    e_a = _rotated_elliptical([10.0, 0.0], 1.5, 0.7, 0.0, n, rng)
    aligned_score = _score(np.stack([g_a, e_a]))

    g_r = _rotated_elliptical([0.0, 0.0], 1.5, 0.7, 0.5, n, rng)
    e_r = _rotated_elliptical([10.0, 0.0], 1.5, 0.7, 0.5, n, rng)
    rotated_score = _score(np.stack([g_r, e_r]))

    assert rotated_score < aligned_score


def test_consistency_drops_for_shape_mismatch(rng):
    """g circular, e wide ellipse, same center → erf factor identical
    (same trace), but consistency factor distinguishes them."""
    n = 6000
    # Identical-shape baseline at large separation
    g_a = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_a = _isotropic([10.0, 0.0], 1.0, n, rng)
    same_shape = _score(np.stack([g_a, e_a]))

    # Asymmetric shapes at the same separation
    g_b = _isotropic([0.0, 0.0], 1.0, n, rng)
    e_b = _rotated_elliptical([10.0, 0.0], 2.5, 0.4, 0.0, n, rng)
    diff_shape = _score(np.stack([g_b, e_b]))

    assert diff_shape < same_shape
    # consistency for {I, diag(6.25, 0.16)} ≈ 0.55 → score should drop notably
    assert diff_shape < same_shape * 0.8


def test_sweep_axis_broadcasting(rng):
    """snr_as_signal should produce one score per sweep point when the
    tracker's leading dims include a sweep axis."""
    n = 3000
    n_sweep = 4

    avgs, meds, covs = [], [], []
    for k in range(n_sweep):
        d = 2.0 + 3.0 * k
        g = _isotropic([0.0, 0.0], 1.0, n, rng)
        e = _isotropic([d, 0.0], 1.0, n, rng)
        samples = np.stack([g, e])
        avgs.append(samples.mean(axis=1))
        meds.append(np.median(samples, axis=1))
        covs.append(np.stack([np.cov(s, rowvar=False) for s in samples]))

    avg = np.stack(avgs, axis=0)  # (sweep, ge, IQ)
    med = np.stack(meds, axis=0)
    cov = np.stack(covs, axis=0)  # (sweep, ge, IQ, IQ)

    raw = [_FakeTracker(mean=avg, covariance=cov, leader_center=med)]
    out = snr_as_signal(raw, ge_axis=1)
    assert out.shape == (n_sweep,)
    assert np.all(np.diff(out) > 0)


def test_symmetric_leakage_keeps_symmetry_factor_high(rng):
    """Both g and e leak 20% into the other state. The pair remains
    mirror-symmetric about the bisector, so the symmetry factor stays
    near 1 — any drop in total score comes from inflated pooled cov,
    not from asymmetry."""
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
    sym_score = _score(np.stack([g, e]))

    # match the inflated pooled covariance with a clean baseline that
    # has the same per-state cov shape — score should be similar (i.e.
    # the symmetric leakage didn't get hit hard by the symmetry factor).
    inflated_var = float(np.cov(g, rowvar=False)[0, 0])
    g_baseline = _isotropic([0.0, 0.0], np.sqrt(inflated_var), n, rng)
    e_baseline = _isotropic([sep, 0.0], np.sqrt(inflated_var), n, rng)
    baseline = _score(np.stack([g_baseline, e_baseline]))

    # symmetry factor ≈ 1 → leakage score should track the matched-cov
    # baseline closely (within ~15%).
    assert sym_score > baseline * 0.85


def test_rotated_pair_penalized_vs_axis_aligned(rng):
    """Identical-shape ellipses with the same eigenvalues but rotated
    by 45° are NOT mirror-symmetric about the y-axis bisector — the
    symmetry factor should make the rotated case score lower than the
    axis-aligned case (erf factor identical: same trace, same medians)."""
    n = 8000
    g_a = _rotated_elliptical([0.0, 0.0], 1.5, 0.5, 0.0, n, rng)
    e_a = _rotated_elliptical([6.0, 0.0], 1.5, 0.5, 0.0, n, rng)
    aligned = _score(np.stack([g_a, e_a]))

    g_r = _rotated_elliptical([0.0, 0.0], 1.5, 0.5, np.pi / 4, n, rng)
    e_r = _rotated_elliptical([6.0, 0.0], 1.5, 0.5, np.pi / 4, n, rng)
    rotated = _score(np.stack([g_r, e_r]))

    assert rotated < aligned
    assert rotated < aligned * 0.85





def test_score_bounded_in_zero_one(rng):
    """erf ∈ [0,1] and consistency ∈ (0,1] → product ∈ [0,1]."""
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
        assert 0.0 <= s <= 1.0
