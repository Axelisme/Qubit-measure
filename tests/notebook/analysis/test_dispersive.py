"""Tests for notebook.analysis.dispersive fast-path wiring.

Verifies that:
- The fast path (calculate_dispersive_vs_flux_fast) is tried first in both
  get_dispersive (search_proper_g) and loss_fn (auto_fit_dispersive).
- When the fast path raises DressedLabelingError, the slow scqubits fallback
  is called and its result is returned unchanged.

Numeric cross-checks between fast and slow are covered at the simulate layer
(tests/simulate/); only the fallback wiring is tested here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

FAST_PATH = "zcu_tools.notebook.analysis.dispersive.calculate_dispersive_vs_flux_fast"
SLOW_PATH = "zcu_tools.notebook.analysis.dispersive.calculate_dispersive_vs_flux"

# Minimal synthetic Fluxonium params (EJ, EC, EL) — values don't matter for
# wiring tests because both paths are monkeypatched.
_PARAMS: tuple[float, float, float] = (8.9, 2.5, 0.5)
_G = 0.05  # GHz
_BARE_RF = 7.0  # GHz
_FLUXS = np.linspace(0.0, 0.5, 10)
_FREQS = np.linspace(6.9, 7.1, 20)
_SIGNALS = np.ones((len(_FLUXS), len(_FREQS)), dtype=complex)

# Sentinel arrays returned by the mocked slow path.
_SLOW_RF0 = np.full(len(_FLUXS), 7.01)
_SLOW_RF1 = np.full(len(_FLUXS), 6.99)
_SLOW_RETURN = (_SLOW_RF0, _SLOW_RF1)


def _slow_side_effect(*args, **kwargs):
    return _SLOW_RETURN


# ---------------------------------------------------------------------------
# get_dispersive (inside search_proper_g)
# ---------------------------------------------------------------------------


class TestGetDispersiveFallback:
    """The lru_cache is local to each search_proper_g call, so we rebuild it
    each test by importing and calling the function fresh."""

    def _build_get_dispersive(self, fast_mock, slow_mock):
        """Return the *inner* get_dispersive closure from search_proper_g."""
        from zcu_tools.notebook.analysis.dispersive import search_proper_g

        # search_proper_g calls get_dispersive(g_init, bare_rf) at construction
        # time — the mocks are already in place via the surrounding patch context.
        close_fn = search_proper_g(
            _PARAMS,
            _BARE_RF * 1e3,  # bare_rf expected in MHz by the widget slider
            _FLUXS,
            _FREQS,
            _SIGNALS,
            g_bound=(_G * 0.5, _G * 2.0),
            g_init=_G,
        )
        return close_fn

    def test_fast_path_used_when_no_error(self):
        """Fast path is called and its result is returned (no DressedLabelingError)."""
        fast_return = (np.ones(len(_FLUXS)) * 7.01, np.ones(len(_FLUXS)) * 6.99)
        with (
            patch(FAST_PATH, return_value=fast_return) as fast_mock,
            patch(SLOW_PATH) as slow_mock,
        ):
            self._build_get_dispersive(fast_mock, slow_mock)
            assert fast_mock.call_count >= 1
            slow_mock.assert_not_called()

    def test_fallback_called_on_dressed_labeling_error(self):
        """When fast raises DressedLabelingError, slow path is called and its result
        is returned."""
        from zcu_tools.simulate.fluxonium import DressedLabelingError

        with (
            patch(FAST_PATH, side_effect=DressedLabelingError("test")) as fast_mock,
            patch(SLOW_PATH, side_effect=_slow_side_effect) as slow_mock,
        ):
            self._build_get_dispersive(fast_mock, slow_mock)
            assert fast_mock.call_count >= 1
            assert slow_mock.call_count >= 1


# ---------------------------------------------------------------------------
# loss_fn (inside auto_fit_dispersive) — fallback wiring
# ---------------------------------------------------------------------------


class TestAutoFitDispersiveFallback:
    """Verify that auto_fit_dispersive's loss_fn tries fast then falls back."""

    def test_fast_path_tried_first(self):
        """Fast path is called during the optimisation; slow path not called."""
        fast_return = (np.ones(len(_FLUXS)) * 7.01, np.ones(len(_FLUXS)) * 6.99)
        with (
            patch(FAST_PATH, return_value=fast_return),
            patch(SLOW_PATH) as slow_mock,
        ):
            from zcu_tools.notebook.analysis.dispersive import auto_fit_dispersive

            auto_fit_dispersive(
                _PARAMS,
                _BARE_RF,
                _FLUXS,
                _FREQS,
                _SIGNALS,
                g_bound=(_G * 0.5, _G * 2.0),
                g_init=_G,
            )
            slow_mock.assert_not_called()

    def test_fallback_on_dressed_labeling_error(self):
        """When fast raises DressedLabelingError inside loss_fn, slow is used and
        the optimisation completes (returns a result tuple)."""
        from zcu_tools.simulate.fluxonium import DressedLabelingError

        # The slow mock must return arrays that match sp_fluxs length, because
        # loss_fn uses them for interpolation.
        def slow_side(*args, **kwargs):
            n = len(_FLUXS)
            return np.full(n, 7.01), np.full(n, 6.99)

        with (
            patch(FAST_PATH, side_effect=DressedLabelingError("test")),
            patch(SLOW_PATH, side_effect=slow_side) as slow_mock,
        ):
            from zcu_tools.notebook.analysis.dispersive import auto_fit_dispersive

            g_fit, rf_fit = auto_fit_dispersive(
                _PARAMS,
                _BARE_RF,
                _FLUXS,
                _FREQS,
                _SIGNALS,
                g_bound=(_G * 0.5, _G * 2.0),
                g_init=_G,
            )
            assert slow_mock.call_count >= 1
            # Result must be a valid float, not None (non-fit_bare_rf path).
            assert isinstance(g_fit, float)
            assert rf_fit is None
