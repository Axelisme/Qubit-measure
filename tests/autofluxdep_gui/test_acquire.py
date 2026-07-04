"""Shared real-acquire helper tests (``nodes/acquire.py``).

The fit-quality gate has no hardware dependency, so it is unit-tested here
directly (the Nodes' real acquire path is covered by the ``test_*_acquire.py``
integration tests). The gate is the angle the integration tests do not isolate:
it decides whether a noisy / dead flux point is discarded (empty Patch, no
calibrate).
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    fill_decay_fit_or_skip,
    is_good_fit,
    is_trusted_decay_scalar_fit,
)
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult

# --- is_good_fit: accepts a clean fit, rejects an all-noise (dead-point) fit ---


def test_is_good_fit_accepts_a_clean_fit():
    x = np.linspace(0, 10, 100)
    clean = np.exp(-x / 3.0)  # a real decay
    assert is_good_fit(clean, clean)  # residual 0 vs a real span


def test_is_good_fit_rejects_a_flat_fit():
    x = np.linspace(0, 10, 100)
    flat = np.zeros_like(x)  # a dead fit (no span)
    noisy = np.random.RandomState(0).randn(100) * 0.5
    assert not is_good_fit(noisy, flat)  # span 0 → rejected


def test_is_good_fit_rejects_a_large_residual():
    x = np.linspace(0, 10, 100)
    fit = np.exp(-x / 3.0)
    # the measured signal is pure noise unrelated to the fitted curve → the mean
    # residual swamps the fit's span, so the gate discards the point.
    noisy = np.random.RandomState(1).randn(100) * 5.0
    assert not is_good_fit(noisy, fit)


def test_t1_t2_scalar_gate_uses_legacy_residual_threshold():
    x = np.linspace(0.5, 10.0, 100)
    fit = np.exp(-x / 3.0)
    measured = fit + 0.15 * np.ptp(fit)

    assert not is_trusted_decay_scalar_fit(measured, fit, 3.0, x)


def test_t1_t2_scalar_gate_rejects_fit_beyond_sweep_window():
    x = np.linspace(0.5, 10.0, 100)
    fit = np.exp(-x / 3.0)

    assert not is_trusted_decay_scalar_fit(fit, fit, 20.1, x)


def test_t1_t2_scalar_gate_accepts_clean_fit_inside_sweep_window():
    x = np.linspace(0.5, 10.0, 100)
    fit = np.exp(-x / 3.0)

    assert is_trusted_decay_scalar_fit(fit, fit, 10.0, x)


def test_fill_decay_fit_or_skip_preserves_raw_evidence_on_reject():
    x = np.linspace(0.5, 10.0, 5)
    result = Sweep1DResult.allocate(np.array([0.0]), x, x_label="time")
    raw = np.linspace(1.0, 0.2, 5)
    fit = raw.copy()
    notified: list[int] = []

    accepted = fill_decay_fit_or_skip(
        result,
        0,
        raw,
        x,
        20.1,
        fit,
        notified.append,
        logger=type("_Logger", (), {"debug": lambda self, *args: None})(),
        node_name="t1",
    )

    assert not accepted
    assert np.isnan(result.fit_value[0])
    assert np.isnan(result.fit_curve[0]).all()
    assert notified == [0]
