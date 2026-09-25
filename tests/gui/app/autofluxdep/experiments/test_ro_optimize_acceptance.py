from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.experiments.ro_optimize import _accepted_ro_optimum


def _axes() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([5999.0, 6000.0, 6001.0], dtype=np.float64),
        np.array([0.45, 0.50, 0.55], dtype=np.float64),
    )


def test_ro_optimize_acceptance_rejects_invalid_landscape():
    freqs, gains = _axes()
    landscape = np.full((3, 3), np.nan, dtype=np.float64)

    assert _accepted_ro_optimum(landscape, freqs, gains) is None


def test_ro_optimize_acceptance_rejects_flat_landscape():
    freqs, gains = _axes()
    landscape = np.ones((3, 3), dtype=np.float64)

    assert _accepted_ro_optimum(landscape, freqs, gains) is None


def test_ro_optimize_acceptance_rejects_boundary_peak():
    freqs, gains = _axes()
    landscape = np.zeros((3, 3), dtype=np.float64)
    landscape[0, 1] = 5.0

    assert _accepted_ro_optimum(landscape, freqs, gains) is None


def test_ro_optimize_acceptance_rejects_non_prominent_peak():
    freqs, gains = _axes()
    landscape = np.zeros((3, 3), dtype=np.float64)
    landscape[1, 1] = 1.0005
    landscape[1, 2] = 1.0

    assert _accepted_ro_optimum(landscape, freqs, gains) is None


def test_ro_optimize_acceptance_rejects_non_finite_freq_axis():
    freqs, gains = _axes()
    freqs[1] = np.nan
    landscape = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.3, 5.0, 0.4],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )

    assert _accepted_ro_optimum(landscape, freqs, gains) is None


def test_ro_optimize_acceptance_rejects_non_finite_gain_axis():
    freqs, gains = _axes()
    gains[1] = np.inf
    landscape = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.3, 5.0, 0.4],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )

    assert _accepted_ro_optimum(landscape, freqs, gains) is None


def test_ro_optimize_acceptance_accepts_clear_interior_peak():
    freqs, gains = _axes()
    landscape = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.3, 5.0, 0.4],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )

    optimum = _accepted_ro_optimum(landscape, freqs, gains)

    assert optimum is not None
    assert optimum.freq_index == 1
    assert optimum.gain_index == 1
    assert optimum.freq == 6000.0
    assert optimum.gain == 0.50
