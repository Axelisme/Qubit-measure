"""Equivalence tests: calculate_energy_vs_flux_fast vs the scqubits baseline.

The fast variant precomputes the flux-independent ``cos(phi)`` / ``sin(phi)``
operators once and recombines them per flux point (instead of scqubits rebuilding
the Hamiltonian — a matrix cosine — at every flux). These tests pin that it
returns the SAME energies as the original ``calculate_energy_vs_flux`` to machine
precision, across several parameter sets and tricky flux orderings (unsorted,
duplicated, outside [0, 0.5]) that exercise the fold/dedup/reorder logic.

These call scqubits, so they are a touch slow but still seconds, not minutes.
"""

from __future__ import annotations

import numpy as np
import pytest
import scqubits.settings as scq_settings
from zcu_tools.simulate.fluxonium import (
    calculate_energy_vs_flux,
    calculate_energy_vs_flux_fast,
)

scq_settings.PROGRESSBAR_DISABLED = True

# (EJ, EC, EL) covering normal / integer-ish / wide ranges.
PARAMS = [
    (5.0, 1.0, 0.5),
    (3.0, 1.5, 0.3),
    (12.0, 0.5, 1.8),
    (2.0, 2.0, 0.1),
]

FLUX_CASES = {
    "linspace_0_0.5": np.linspace(0.0, 0.5, 60),
    "unsorted": np.array([0.3, 0.1, 0.5, 0.0, 0.25, 0.4, 0.05]),
    "dups_and_above_half": np.array([0.1, 0.1, 0.7, 0.9, 0.3, 0.3, 0.5]),
}


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("flux_label", list(FLUX_CASES))
def test_fast_matches_baseline(params, flux_label):
    fluxs = FLUX_CASES[flux_label]
    _, slow = calculate_energy_vs_flux(params, fluxs, cutoff=40, evals_count=15)
    none_sd, fast = calculate_energy_vs_flux_fast(
        params, fluxs, cutoff=40, evals_count=15
    )
    assert none_sd is None  # fast variant returns no SpectrumData
    assert fast.shape == slow.shape
    np.testing.assert_allclose(fast, slow, atol=1e-11, rtol=0)


def test_fast_shape_and_order_preserved():
    # the returned energies must line up with the INPUT flux order, not folded
    fluxs = np.array([0.4, 0.1, 0.4])  # 0.4 repeated → rows 0 and 2 equal
    _, energies = calculate_energy_vs_flux_fast(
        (5.0, 1.0, 0.5), fluxs, cutoff=40, evals_count=10
    )
    assert energies.shape == (3, 10)
    np.testing.assert_allclose(energies[0], energies[2])
    assert not np.allclose(energies[0], energies[1])


def test_fast_respects_evals_count():
    fluxs = np.linspace(0.0, 0.5, 10)
    for n in (5, 12, 20):
        _, energies = calculate_energy_vs_flux_fast(
            (5.0, 1.0, 0.5), fluxs, cutoff=40, evals_count=n
        )
        assert energies.shape == (10, n)
