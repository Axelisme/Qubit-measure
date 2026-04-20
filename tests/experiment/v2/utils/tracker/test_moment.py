from __future__ import annotations

import numpy as np

from zcu_tools.experiment.v2.utils.tracker import MomentTracker


def _direct_moments(points: np.ndarray):
    n = points.shape[-2]
    mean = points.mean(axis=-2)
    centered = points - mean[..., None, :]
    m2 = np.einsum("...mi,...mj->...ij", centered, centered)
    cov = m2 if n <= 1 else m2 / (n - 1)
    m3 = np.einsum("...mi,...mj,...mk->...ijk", centered, centered, centered) / n
    return mean, cov, m3


def test_moment_tracker_matches_direct_statistics():
    rng = np.random.default_rng(11)
    leading = (2, 3)
    n = 1500
    pts = rng.normal(size=leading + (n, 2))
    skew = rng.exponential(scale=0.8, size=leading + (n,)) - 0.8
    pts[..., 0] += skew
    pts[..., 1] += 0.5 * skew

    tracker = MomentTracker()
    tracker.update(pts)

    mean_ref, cov_ref, m3_ref = _direct_moments(pts)
    assert np.allclose(tracker.mean, mean_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(tracker.covariance, cov_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(tracker.third_moment, m3_ref, atol=1e-10, rtol=1e-10)


def test_moment_tracker_chunked_update_consistency():
    rng = np.random.default_rng(12)
    pts = rng.normal(size=(3, 1025, 2))
    skew = rng.exponential(scale=0.5, size=(3, 1025)) - 0.5
    pts[..., 0] += skew
    pts[..., 1] -= 0.3 * skew

    full = MomentTracker()
    full.update(pts)

    chunked = MomentTracker()
    chunked.update(pts[..., :257, :])
    chunked.update(pts[..., 257:800, :])
    chunked.update(pts[..., 800:, :])

    assert chunked.n == full.n
    assert np.allclose(chunked.mean, full.mean, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.covariance, full.covariance, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.third_moment, full.third_moment, atol=1e-10, rtol=1e-10)
