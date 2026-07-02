"""Anchor tests for the Floquet branch ge-SNR pipeline.

These lock the numeric output of ``calc_ge_snr`` so that the performance
refactors in ``floquet.py`` (removing the photon-layer joblib, relaxing the
ODE tolerance) are provably behaviour-preserving where they must be:

- the strict (qutip-default) solver path stays bit-exact vs the golden values;
- the relaxed-tolerance path stays within a tight relative tolerance.

The golden ``snr[-3]`` values were captured on this machine from the original
(joblib, qutip-default) implementation. Reproducibility spread was measured as
exactly 0.0 (bit-exact deterministic), so the baseline path is locked with
``atol=1e-12``. ``calc_ge_snr`` now defaults to a relaxed solver tolerance, so
the baseline tests pass ``solver_options=None`` to exercise the strict path.

A reduced ``max_photon=30`` photon grid is used to keep each test < 1s; the
golden values correspond to that grid, not the design.ipynb ``max_photon=70``.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.simulate.fluxonium.branch.floquet import (
    FloquetDualCouplingBranchAnalysis,
    FloquetWithTLSBranchAnalysis,
    calc_ge_snr,
)

# Shared grid for every case. max_photon=30 keeps each call well under 1s while
# leaving the branch-tracking resolution intact for these parameter points.
_COMMON = dict(
    flux=0.5,
    r_f=5.927,
    g=0.11,
    rf_w=7e-3,
    qub_dim=15,
    qub_cutoff=40,
    max_photon=30,
)

# (params, golden snr[-3]) captured from the unmodified floquet.py on this
# machine. Deterministic to the ULP (measured reproducibility spread == 0.0).
_GOLDEN: list[tuple[tuple[float, float, float], float]] = [
    ((5.5, 1.2, 0.9), 0.6161074224266799),
    ((4.5, 1.0, 0.5), 1.5201939570255476),
    ((7.0, 1.4, 1.4), 1.9192775983522998),
]


def _snr3(params: tuple[float, float, float], **overrides) -> float:
    """Run the ge-SNR pipeline and return the design metric ``sort(snr)[-3]``."""
    kwargs = {**_COMMON, **overrides}
    _, snrs = calc_ge_snr(params=params, **kwargs)
    return float(np.sort(snrs)[-3])


@pytest.mark.parametrize("params,_golden", _GOLDEN)
def test_reproducible(params: tuple[float, float, float], _golden: float) -> None:
    # Lock the determinism invariant: the same inputs give bit-identical output.
    # Uses the default (relaxed) path; determinism must hold there too.
    assert _snr3(params) == _snr3(params)


@pytest.mark.parametrize("params,golden", _GOLDEN)
def test_baseline_golden(params: tuple[float, float, float], golden: float) -> None:
    # The strict (qutip-default) path must stay bit-exact across refactors.
    assert _snr3(params, solver_options=None) == pytest.approx(golden, abs=1e-12)


@pytest.mark.parametrize("params,golden", _GOLDEN)
def test_relaxed_tol_within_tol(
    params: tuple[float, float, float], golden: float
) -> None:
    # The default relaxed-tolerance path (rtol=1e-3, atol=1e-5) must stay within
    # a tight relative tolerance of the strict golden. 2e-4 covers the ~6e-5
    # measured relative error with margin.
    assert _snr3(params) == pytest.approx(golden, rel=2e-4)


def test_tls_branch_analysis_characterization() -> None:
    photons = np.array([0.0, 0.2])
    avg_times = np.array([0.0])
    analysis = FloquetWithTLSBranchAnalysis(
        (4.5, 1.0, 0.5),
        r_f=5.0,
        g=0.03,
        E_tls=4.8,
        g_tls=0.01,
        qub_dim=4,
        qub_cutoff=12,
        solver_options=None,
    )
    fbasis_n = [
        analysis.make_floquet_basis(float(photon), precompute=avg_times)
        for photon in photons
    ]

    branch_infos = analysis.calc_branch_infos(fbasis_n, [0, 1], progress=False)
    populations = analysis.calc_branch_populations(
        fbasis_n, branch_infos, avg_times=avg_times, progress=False
    )

    assert branch_infos == {0: [0, 0], 1: [1, 1]}
    np.testing.assert_allclose(
        populations[0],
        [6.681065569224993e-05, 0.0002795982255832228],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        populations[1],
        [1.0000551277881877, 1.0001045753186426],
        atol=1e-12,
        rtol=0,
    )


def test_dual_coupling_branch_analysis_characterization() -> None:
    photons = np.array([0.0, 0.2])
    avg_times = np.array([0.0])
    analysis = FloquetDualCouplingBranchAnalysis(
        (4.5, 1.0, 0.5),
        r_f=5.0,
        g1=0.03,
        g2=0.02,
        qub_dim=4,
        qub_cutoff=12,
        solver_options=None,
    )
    fbasis_n = [
        analysis.make_floquet_basis(float(photon), precompute=avg_times)
        for photon in photons
    ]

    branch_infos = analysis.calc_branch_infos(fbasis_n, [0, 1], progress=False)
    energies = analysis.calc_branch_energies(fbasis_n, branch_infos)
    populations = analysis.calc_branch_populations(
        fbasis_n, branch_infos, avg_times, progress=False
    )

    assert branch_infos == {0: [1, 1], 1: [2, 2]}
    np.testing.assert_allclose(
        energies[0],
        [0.42538041856448916, 0.4252126135249715],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        energies[1],
        [0.6011622294113531, 0.6014513084394016],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        populations[0],
        [0.0, 0.0007636154963531296],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        populations[1],
        [1.0, 1.0002483778485474],
        atol=1e-12,
        rtol=0,
    )
