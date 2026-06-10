"""Tests for the matrix-element vs-flux functions (BLAS-pin correctness).

calculate_{n,phi,sin_phi}_oper_vs_flux pin the BLAS backend to one thread for
the scqubits sweep — on these tiny matrices OpenBLAS multithreading is a ~150x
net loss. Pinning the thread count only changes parallel scheduling, never the
numbers, so these tests assert the pinned result is byte-for-byte (to ~1e-13)
the same as an explicit BLAS-multithreaded scqubits computation, and pin the
output shapes.
"""

from __future__ import annotations

import numpy as np
import pytest
import scqubits.settings as scq_settings
from zcu_tools.simulate.fluxonium.matrix_element import (
    calculate_n_oper_vs_flux,
    calculate_phi_oper_vs_flux,
    calculate_sin_phi_oper_vs_flux,
)

scq_settings.PROGRESSBAR_DISABLED = True

PARAMS = (5.0, 1.0, 0.5)
FLUXS = np.linspace(0.0, 0.5, 40)
RETURN_DIM = 4


def _reference_matelements(operator):
    """Explicit BLAS-MULTITHREADED scqubits sweep (no pin) — the ground truth.

    Mirrors the implementations' cutoff=40 / flux=0.5 setup so only the BLAS
    thread count differs.
    """
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*PARAMS, flux=0.5, cutoff=40, truncated_dim=RETURN_DIM)
    if operator == "n_operator":
        op = "n_operator"
    elif operator == "phi":
        op = fluxonium.phi_operator(energy_esys=False)
    else:
        op = fluxonium.sin_phi_operator(alpha=1.0, beta=0.0, energy_esys=False)
    sd = fluxonium.get_matelements_vs_paramvals(
        operator=op, param_name="flux", param_vals=FLUXS, evals_count=RETURN_DIM
    )
    return np.asarray(sd.matrixelem_table)[:, :RETURN_DIM, :RETURN_DIM]


@pytest.mark.parametrize(
    "fn,opkey",
    [
        (calculate_n_oper_vs_flux, "n_operator"),
        (calculate_phi_oper_vs_flux, "phi"),
        (calculate_sin_phi_oper_vs_flux, "sin_phi"),
    ],
)
def test_pinned_matches_blas_multithreaded(fn, opkey):
    """Assert the BLAS-pinned result is numerically identical to a multithreaded run.

    NOTE: under pytest-xdist workers tests/conftest.py pins BLAS to 1 thread
    (OMP/OPENBLAS/MKL_NUM_THREADS=1), so both ``pinned`` and ``reference`` here
    run single-threaded.  The "multithreaded" coverage of this test is only
    exercised in serial runs (``pytest tests/`` without ``-n auto``).
    """
    _, pinned = fn(PARAMS, FLUXS)
    reference = _reference_matelements(opkey)
    assert pinned.shape == (len(FLUXS), RETURN_DIM, RETURN_DIM)
    np.testing.assert_allclose(pinned, reference, atol=1e-11, rtol=0)


def test_spectrum_data_passthrough_skips_recompute():
    # passing a previously returned spectrum_data reuses its matrixelem_table
    sd, first = calculate_n_oper_vs_flux(PARAMS, FLUXS)
    sd2, second = calculate_n_oper_vs_flux(PARAMS, FLUXS, spectrum_data=sd)
    assert sd2 is sd
    np.testing.assert_array_equal(first, second)
