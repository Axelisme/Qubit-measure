from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as sc
import zcu_tools.simulate.fluxonium as fluxonium
import zcu_tools.simulate.fluxonium.coherence as coherence
from zcu_tools.simulate.fluxonium.coherence.purcell import purcell


def test_purcell_export_uses_correct_spelling() -> None:
    assert (
        fluxonium.calculate_purcell_t1_vs_flux is coherence.calculate_purcell_t1_vs_flux
    )
    assert "calculate_purcell_t1_vs_flux" in fluxonium.__all__
    assert "calculate_percell_t1_vs_flux" not in fluxonium.__all__
    assert not hasattr(fluxonium, "calculate_percell_t1_vs_flux")


def test_purcell_t1_scales_inversely_with_resonator_linewidth() -> None:
    fluxs = np.array([0.2, 0.35], dtype=np.float64)
    first = fluxonium.calculate_purcell_t1_vs_flux(
        fluxs,
        bare_rf=5.0,
        kappa=0.001,
        g=0.05,
        Temp=0.04,
        params=(4.0, 1.0, 0.5),
        progress=False,
    )
    second = fluxonium.calculate_purcell_t1_vs_flux(
        fluxs,
        bare_rf=5.0,
        kappa=0.002,
        g=0.05,
        Temp=0.04,
        params=(4.0, 1.0, 0.5),
        progress=False,
    )

    assert first.shape == second.shape == (2,)
    assert first.dtype == second.dtype == np.float64
    for result in (first, second):
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)
    np.testing.assert_allclose(
        2 * np.asarray(second), np.asarray(first), rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize(
    "temperature",
    [0.0, 0.04, 0.08],
    ids=["zero-temperature", "thermal-40mK", "thermal-80mK"],
)
def test_purcell_sweep_matches_lc_equilibrium_t1_in_si(temperature: float) -> None:
    ec_ghz, el_ghz = 0.3, 1.5
    bare_rf_ghz, coupling_ghz, linewidth_ghz = 5.0, 0.001, 0.001

    # EJ=0 is an LC oscillator: derive its frequency and charge dipole in SI,
    # independently of scqubits and the Purcell implementation.
    capacitance = sc.e**2 / (2 * sc.h * ec_ghz * 1e9)
    inductance = (sc.hbar / (2 * sc.e)) ** 2 / (sc.h * el_ghz * 1e9)
    qubit_hz = 1 / (2 * np.pi * np.sqrt(inductance * capacitance))
    charge_dipole_squared = sc.h * qubit_hz / (16 * sc.h * ec_ghz * 1e9)
    mixing_squared = (
        (coupling_ghz * 1e9) ** 2
        * charge_dipole_squared
        / (qubit_hz - bare_rf_ghz * 1e9) ** 2
    )
    spontaneous_rate_per_second = 2 * np.pi * linewidth_ghz * 1e9 * mixing_squared
    # For an LC mode coupled to a thermal bath, Gamma_eg=Gamma0*(n_B+1)
    # and Gamma_ge=Gamma0*n_B. Their sum is Gamma0*coth(h*nu_q/(2*k_B*T)).
    # Derive its reciprocal directly from SI constants, not directional helpers.
    inverse_thermal_sum = (
        1.0 if temperature == 0 else np.tanh(sc.h * qubit_hz / (2 * sc.k * temperature))
    )
    expected_ns = 1e9 * inverse_thermal_sum / spontaneous_rate_per_second
    # Resonator photons are spectators for this harmonic qubit-mode transition.
    # Even at 80 mK the omitted geometric thermal tail is below 1e-13.
    # The reverse rate makes equilibrium T1 shorter than e-to-g escape.
    try:
        actual = fluxonium.calculate_purcell_t1_vs_flux(
            np.array([0.0, 0.3]),
            bare_rf=bare_rf_ghz,
            kappa=linewidth_ghz,
            g=coupling_ghz,
            Temp=temperature,
            params=(0.0, ec_ghz, el_ghz),
            progress=False,
        )
    except ZeroDivisionError:
        pytest.fail("Temp=0 must retain spontaneous decay, not divide by temperature")

    assert actual.shape == (2,)
    assert actual.dtype == np.float64
    # First-order admixture has relative corrections of order g²/Δ².
    np.testing.assert_allclose(np.asarray(actual), expected_ns, rtol=1e-5, atol=0.0)


def test_purcell_matched_pair_obeys_thermal_detailed_balance() -> None:
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator

    # An uncoupled resonator pair is an exact dressed-state pair with unit
    # lowering/raising dipole. Both evaluations use the same n=0 thermal weight,
    # rather than comparing multi-frequency aggregates with different weights.
    qubit = Fluxonium(0.0, 0.3, 1.5, flux=0.0, cutoff=20, truncated_dim=2)
    resonator = Oscillator(E_osc=1.25, truncated_dim=2)
    energies, vectors = HilbertSpace([qubit, resonator]).eigensys(evals_count=2)
    _, resonator_vectors = resonator.eigensys(evals_count=2)
    low_energies, high_energies = energies[:1], energies[1:2]
    low_vectors, high_vectors = vectors[:1], vectors[1:2]
    temperature = 0.04
    downward_ns = purcell(
        low_energies,
        low_vectors,
        high_energies,
        high_vectors,
        resonator_vectors,
        bare_rf=1.25,
        kappa=0.001,
        Temp=temperature,
        ns=np.array([0], dtype=np.int64),
        fluxonium=qubit,
        resonator=resonator,
    )
    upward_ns = purcell(
        high_energies,
        high_vectors,
        low_energies,
        low_vectors,
        resonator_vectors,
        bare_rf=1.25,
        kappa=0.001,
        Temp=temperature,
        ns=np.array([0], dtype=np.int64),
        fluxonium=qubit,
        resonator=resonator,
    )

    expected_rate_ratio = np.exp(-sc.h * 1.25e9 / (sc.k * temperature))
    np.testing.assert_allclose(
        downward_ns / upward_ns, expected_rate_ratio, rtol=1e-12, atol=0.0
    )
