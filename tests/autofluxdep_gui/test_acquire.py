"""Shared real-acquire helper tests (``nodes/acquire.py``).

The fit-quality gate (``is_good_fit``) and the free-text axis parser
(``parse_linear_axis``) are pure helpers shared by every measurement Node's
produce. They have no hardware dependency, so they are unit-tested here directly
(the Nodes' real acquire path is covered by the ``test_*_acquire.py`` integration
tests). The gate is the angle the integration tests do not isolate: it decides
whether a noisy / dead flux point is discarded (empty Patch, no calibrate).
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.nodes.acquire import is_good_fit, parse_linear_axis

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


# --- parse_linear_axis: free text → linspace, malformed → default ---


def test_parse_linear_axis_parses_text():
    axis = parse_linear_axis("0,10,11", (0.0, 1.0, 2))
    assert axis[0] == 0.0 and axis[-1] == 10.0 and axis.shape[0] == 11


def test_parse_linear_axis_parses_tuple():
    axis = parse_linear_axis((0.0, 5.0, 6), (0.0, 1.0, 2))
    assert axis[0] == 0.0 and axis[-1] == 5.0 and axis.shape[0] == 6


def test_parse_linear_axis_degrades_to_default_on_bad_value():
    axis = parse_linear_axis("not,a,number", (0.0, 1.0, 3))
    assert axis[0] == 0.0 and axis[-1] == 1.0 and axis.shape[0] == 3


def test_parse_linear_axis_clamps_to_two_points():
    axis = parse_linear_axis("0,1,1", (0.0, 1.0, 2))
    assert axis.shape[0] == 2  # at least 2 points (a linspace needs endpoints)
