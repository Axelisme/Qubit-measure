"""Correctness tests for calculate_dispersive_vs_flux_fast (the numpy-only path).

calculate_dispersive_vs_flux_fast bypasses scqubits' ParameterSweep: it builds the
composite (resonator ⊗ fluxonium) Hamiltonian per flux in numpy and labels the
dressed (0/1, i) levels by their bare-product-state overlap. These tests assert it
matches the trusted scqubits ``calculate_dispersive_vs_flux`` across the
avoided-crossing region and several parameter sets / tricky flux orderings (unsorted,
duplicated, outside [0, 0.5]) that exercise the fold/dedup/reorder logic.

These call scqubits for the reference, so they are a touch slow but still seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import scqubits.settings as scq_settings
from zcu_tools.simulate.fluxonium import (
    DressedLabelingError,
    calculate_dispersive_vs_flux,
    calculate_dispersive_vs_flux_fast,
)

scq_settings.PROGRESSBAR_DISABLED = True

PARAMS = [
    (4.0, 1.0, 0.5),
    (5.0, 1.2, 0.4),
    (3.0, 1.5, 0.3),
]

FLUX_CASES = {
    "linspace_0_0.5": np.linspace(0.0, 0.5, 40),
    "unsorted": np.array([0.3, 0.1, 0.5, 0.0, 0.25, 0.4, 0.05]),
    "dups_and_above_half": np.array([0.1, 0.1, 0.7, 0.9, 0.3, 0.3, 0.5]),
}

_BARE_RF = 5.3
_G = 0.06


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("case", list(FLUX_CASES))
def test_fast_matches_scqubits(params, case):
    fluxs = FLUX_CASES[case]
    fast = calculate_dispersive_vs_flux_fast(
        params, fluxs, _BARE_RF, _G, res_dim=4, qub_dim=15, qub_cutoff=30, return_dim=2
    )
    ref = calculate_dispersive_vs_flux(
        params,
        fluxs,
        _BARE_RF,
        _G,
        progress=False,
        res_dim=4,
        qub_dim=15,
        qub_cutoff=30,
    )
    assert len(fast) == len(ref) == 2
    for i in range(2):
        # match to well below 1 kHz (the rounding floor) — measured at 0.0 in MHz
        np.testing.assert_allclose(fast[i], ref[i], atol=1e-7)


def test_fast_respects_flux_order_with_dups():
    # the fold/dedup/reorder must return one value per INPUT flux, in input order
    fluxs = np.array([0.2, 0.2, 0.4, 0.1])
    fast = calculate_dispersive_vs_flux_fast(
        (4.0, 1.0, 0.5), fluxs, _BARE_RF, _G, return_dim=2
    )
    assert fast[0].shape == (4,)
    # duplicated input fluxes give identical outputs
    assert fast[0][0] == fast[0][1]


def test_return_dim_controls_number_of_lines():
    fluxs = np.linspace(0.0, 0.5, 10)
    out = calculate_dispersive_vs_flux_fast(
        (4.0, 1.0, 0.5), fluxs, _BARE_RF, _G, return_dim=3
    )
    assert len(out) == 3
    assert all(line.shape == (10,) for line in out)


def test_dressed_labeling_error_is_a_runtime_error():
    # the guard type exists and is catchable (the fallback in PredictService relies
    # on it); we don't force a collision here — that needs pathological parameters.
    assert issubclass(DressedLabelingError, RuntimeError)
