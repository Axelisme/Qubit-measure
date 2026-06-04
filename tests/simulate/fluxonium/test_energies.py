"""Correctness tests for calculate_energy_vs_flux (the precompute-cos/sin path).

calculate_energy_vs_flux precomputes the flux-independent ``cos(phi)`` /
``sin(phi)`` operators once and recombines them per flux point, instead of
scqubits rebuilding the Hamiltonian (a matrix cosine) at every flux. To keep a
ground truth after that optimisation replaced the old scqubits path, these tests
compute an INDEPENDENT reference straight from ``Fluxonium`` (per-flux
``hamiltonian()`` + ``eigvalsh``) and assert our energies match to machine
precision, across several parameter sets and tricky flux orderings (unsorted,
duplicated, outside [0, 0.5]) that exercise the fold/dedup/reorder logic.

These call scqubits, so they are a touch slow but still seconds, not minutes.
"""

from __future__ import annotations

import numpy as np
import pytest
import scqubits.settings as scq_settings
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux

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


def _reference_energies(params, fluxs, cutoff, evals_count):
    """Independent scqubits ground truth: build H per flux, eigvalsh, in order.

    Deliberately the naive per-flux path (no folding/dedup) so it cannot share a
    bug with the implementation under test.
    """
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    out = np.empty((len(fluxs), evals_count), dtype=np.float64)
    for i, flux in enumerate(fluxs):
        fluxonium.flux = float(flux)
        H = fluxonium.hamiltonian()
        out[i] = np.linalg.eigvalsh(H)[:evals_count]
    return out


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("flux_label", list(FLUX_CASES))
def test_matches_scqubits_reference(params, flux_label):
    fluxs = FLUX_CASES[flux_label]
    sd, energies = calculate_energy_vs_flux(params, fluxs, cutoff=40, evals_count=15)
    reference = _reference_energies(params, fluxs, cutoff=40, evals_count=15)

    assert energies.shape == reference.shape
    np.testing.assert_allclose(energies, reference, atol=1e-11, rtol=0)

    # SpectrumData mirrors the scqubits layout: folded-unique flux grid.
    folded = fluxs % 1.0
    folded = np.where(folded < 0.5, folded, 1.0 - folded)
    expected_grid = np.unique(folded)
    np.testing.assert_allclose(sd.param_vals, expected_grid)
    assert sd.param_name == "flux"
    assert np.asarray(sd.energy_table).shape == (len(expected_grid), 15)


def test_shape_and_order_preserved():
    # the returned energies must line up with the INPUT flux order, not folded
    fluxs = np.array([0.4, 0.1, 0.4])  # 0.4 repeated → rows 0 and 2 equal
    _, energies = calculate_energy_vs_flux(
        (5.0, 1.0, 0.5), fluxs, cutoff=40, evals_count=10
    )
    assert energies.shape == (3, 10)
    np.testing.assert_allclose(energies[0], energies[2])
    assert not np.allclose(energies[0], energies[1])


def test_respects_evals_count():
    fluxs = np.linspace(0.0, 0.5, 10)
    for n in (5, 12, 20):
        _, energies = calculate_energy_vs_flux(
            (5.0, 1.0, 0.5), fluxs, cutoff=40, evals_count=n
        )
        assert energies.shape == (10, n)


def test_flux_symmetry_about_half():
    # the spectrum is even about flux = 0.5: E(0.5 + d) == E(0.5 - d)
    _, energies = calculate_energy_vs_flux(
        (5.0, 1.0, 0.5), np.array([0.3, 0.7]), cutoff=40, evals_count=10
    )
    np.testing.assert_allclose(energies[0], energies[1], atol=1e-11)


def test_spectrum_data_passthrough_skips_computation():
    # passing a previously returned spectrum_data reuses it (no recomputation):
    # the same object is returned, and energies come from its energy_table.
    fluxs = np.linspace(0.0, 0.5, 12)
    sd, _ = calculate_energy_vs_flux((5.0, 1.0, 0.5), fluxs, cutoff=40, evals_count=10)

    # bogus params/fluxs would give wrong energies IF it recomputed — it must not
    sd2, energies = calculate_energy_vs_flux(
        (99.0, 99.0, 99.0),
        np.array([0.0]),
        cutoff=40,
        evals_count=10,
        spectrum_data=sd,
    )
    assert sd2 is sd
    np.testing.assert_array_equal(energies, np.asarray(sd.energy_table))
