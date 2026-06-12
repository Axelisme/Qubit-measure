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
from zcu_tools.simulate.fluxonium.branch.floquet import calc_ge_snr

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
