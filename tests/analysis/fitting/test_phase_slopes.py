"""Public multi-group quadrature slope recovery and alias contracts."""

from importlib import import_module

import numpy as np
import pytest


def quadratures(n, groups=1):
    angle = np.deg2rad(
        n[None, None, :, None] * np.array([31.0, -47.0])[None, :, None, None]
    )
    offsets = np.arange(groups)[:, None, None, None] * 43.0
    angle = angle + np.deg2rad(offsets)
    values = 0.7 * np.stack((np.cos(angle), np.sin(angle)), axis=2)
    noise = np.random.default_rng(41).normal(0, 0.003, values.shape)
    return values + noise


@pytest.mark.parametrize("groups", [1, 2])
def test_shared_slope_recovers_both_quadratures_with_covariance(groups):
    owner = import_module("zcu_tools.analysis.fitting")
    n = np.arange(9, dtype=np.int64)
    values = quadratures(n, groups)
    result = owner.fit_phase_slopes(
        n, values, np.full_like(values, 0.003**2), np.full_like(values, 1000)
    )
    np.testing.assert_allclose(result.slopes_deg[:, 0], [31, -47], atol=0.15)
    assert result.reasons == ((None,), (None,))
    for state in range(2):
        fitted = result.fits[state][0]
        assert fitted is not None
        assert fitted.covariance.shape == (1 + 2 * groups, 1 + 2 * groups)
        np.testing.assert_allclose(fitted.covariance, fitted.covariance.T)
        assert np.all(np.linalg.eigvalsh(fitted.covariance) > 0)
        assert result.errors_deg[state, 0] ** 2 == pytest.approx(
            fitted.variance("slope")
        )
        # Independent Fisher estimate: sigma/(A*sqrt(sum((n-mean(n))²))).
        assert result.errors_deg[state, 0] == pytest.approx(
            0.0317 / np.sqrt(groups), rel=0.35
        )


def test_accumulated_phase_uses_shared_slope_and_each_groups_offset():
    owner = import_module("zcu_tools.analysis.fitting")
    n = np.arange(9, dtype=np.int64)
    shots = 2000
    excited = np.rint(shots * (1 + quadratures(n, groups=2)) / 2)
    values = 2 * excited / shots - 1
    result = owner.fit_phase_slopes(
        n, values, (1 - values**2) / shots, np.full_like(values, shots)
    )

    # Both groups accumulate -47°/gate in channel e; only group 1 starts at 43°.
    grid = np.array([0.0, 2.5, 8.0])
    for group, expected in enumerate(([0, -117.5, -376], [43, -74.5, -333])):
        phases = result.accumulated_phase_deg(
            grid, channel=1, coordinate=0, group=group
        )
        np.testing.assert_allclose(phases, expected, atol=0.5)
    assert result.errors_deg[1, 0] == pytest.approx(0.0224, rel=0.35)


def test_each_group_must_identify_its_own_gate_count_alias():
    owner = import_module("zcu_tools.analysis.fitting")
    n = np.arange(5, dtype=np.int64) * 2
    values = quadratures(n)
    result = owner.fit_phase_slopes(
        n, values, np.ones_like(values), np.ones_like(values)
    )
    assert np.isnan(result.slopes_deg).all()
    assert "gcd=2" in result.reasons[0][0]
    # The union has gcd=1, but independent group offsets cannot break an alias.
    n = np.arange(6, dtype=np.int64)
    values = quadratures(n, 2)
    shots = np.ones_like(values)
    shots[0, :, :, 1::2, :] = 0
    shots[1, :, :, ::2, :] = 0
    result = owner.fit_phase_slopes(n, values, np.ones_like(values), shots)
    assert np.isnan(result.slopes_deg).all()
    assert "gcd=2" in result.reasons[0][0]


def test_incomplete_group_blocks_fit_but_zero_acceptance_removes_point():
    owner = import_module("zcu_tools.analysis.fitting")
    n = np.arange(6, dtype=np.int64)
    values = quadratures(n, 2)
    shots = np.full_like(values, 1000)
    shots[1, :, :, 0, :] = np.nan
    result = owner.fit_phase_slopes(n, values, np.ones_like(values), shots)
    assert np.isnan(result.slopes_deg).all()
    assert "Group 1: incomplete" in result.reasons[0][0]
    shots[1, :, :, 0, :] = 0
    result = owner.fit_phase_slopes(n, values, np.ones_like(values), shots)
    np.testing.assert_allclose(result.slopes_deg[:, 0], [31, -47], atol=0.2)
    shots[1, :, :, 1:, :] = 0
    result = owner.fit_phase_slopes(n, values, np.ones_like(values), shots)
    assert "At least two" in result.reasons[0][0]
