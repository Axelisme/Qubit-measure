"""Single-circuit model facts, fixed-coordinate projection and physical baths."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.simulate import fluxonium
from zcu_tools.simulate.evolution import DriveSource


@pytest.mark.parametrize(("flux", "frequency"), [(0.5, 0.58), (0.0, 5.4)])
def test_model_spectrum_and_charge_match_single_fluxonium_facts(flux, frequency):
    assert hasattr(fluxonium, "SingleFluxoniumModel"), "single circuit model is missing"
    model = fluxonium.SingleFluxoniumModel(
        (4.0, 1.0, 1.0), flux, cutoff=30, retained_levels=3, temperature_k=0
    )
    energy, vectors = model.eigensystem(flux)
    assert energy[1] - energy[0] == pytest.approx(frequency, abs=0.08)
    assert_allclose(vectors.conj().T @ vectors, np.eye(3), atol=1e-12)
    assert_allclose(energy, model.energies_ghz, atol=1e-12)
    if flux == 0.5:
        assert abs(model.charge_operator()[0, 1]) == pytest.approx(0.16, abs=0.01)


@pytest.mark.parametrize("cutoff", [18, 30])
def test_cutoff_aware_helpers_reconstruct_native_hamiltonian(cutoff):
    assert hasattr(fluxonium, "calculate_oscillator_operators"), "operators are missing"
    assert hasattr(fluxonium, "calculate_eigensystem"), "eigensystem is missing"
    params = (4.0, 1.0, 1.0)
    operators = fluxonium.calculate_oscillator_operators(params, cutoff=cutoff)
    energy, vectors = fluxonium.calculate_eigensystem(
        params, 0.31, cutoff=cutoff, evals_count=4
    )
    hamiltonian = (
        operators.static_ghz
        + np.cos(2 * np.pi * 0.31) * operators.cosine_ghz
        + np.sin(2 * np.pi * 0.31) * operators.sine_ghz
    )
    assert vectors.shape == (cutoff, 4)
    assert operators.charge.shape == (cutoff, cutoff)
    assert_allclose(vectors.conj().T @ vectors, np.eye(4), atol=1e-12)
    assert_allclose(hamiltonian @ vectors, vectors * energy, atol=2e-12)
    assert_allclose(
        energy,
        fluxonium.calculate_energy(params, 0.31, cutoff=cutoff, evals_count=4),
        atol=1e-12,
    )


def test_flux_expansion_and_secular_drives_share_fixed_dc_coordinates():
    assert hasattr(fluxonium, "SingleFluxoniumModel"), "single circuit model is missing"
    model = fluxonium.SingleFluxoniumModel(
        (4.0, 1.0, 1.0), 0.5, cutoff=18, retained_levels=18, temperature_k=0
    )
    expansion = model.flux_expansion()
    x = 0.02 * model.charge_operator()
    drive = DriveSource("xy", x, -0.5 * x)
    for flux, level in [(0.5, (0,)), (0.31, (-190,)), (0.0, (-500,))]:
        reconstructed = expansion.static.copy()
        for term in expansion.terms:
            assert term.loop == 0
            factor = {
                "cos_2pi": np.cos(2 * np.pi * flux),
                "sin_2pi": np.sin(2 * np.pi * flux),
            }[term.kind]
            reconstructed += factor * term.matrix
        assert_allclose(reconstructed, model.hamiltonian(flux), atol=2e-12)
        secular = model.secular_model(level, (drive,))
        assert_allclose(
            secular.energies_ghz,
            fluxonium.calculate_energy(
                (4.0, 1.0, 1.0), flux, cutoff=18, evals_count=18
            ),
            atol=2e-12,
        )
        v = secular.eigenvectors
        assert_allclose(
            v.conj().T @ reconstructed @ v, np.diag(secular.energies_ghz), atol=2e-12
        )
        assert_allclose(secular.drives[0].in_phase_ghz, v.conj().T @ x @ v, atol=1e-12)
        assert_allclose(
            secular.drives[0].quadrature_ghz, -0.5 * v.conj().T @ x @ v, atol=1e-12
        )


def test_default_flux_excursion_retains_low_spectrum_within_ten_khz():
    assert hasattr(fluxonium, "SingleFluxoniumModel"), "single circuit model is missing"
    # At cutoff 40, 20 DC states miss the endpoint by 10.9696 kHz; 21 is the
    # smallest projection meeting the 10 kHz absolute lowest-three-level budget.
    model = fluxonium.SingleFluxoniumModel(
        (4.0, 1.0, 1.0), 0.5, cutoff=40, retained_levels=21, temperature_k=0
    )
    for flux in np.linspace(0.5, 0.7, 21):
        projected, _ = model.eigensystem(float(flux))
        direct = fluxonium.calculate_energy(
            (4.0, 1.0, 1.0), float(flux), cutoff=40, evals_count=3
        )
        assert_allclose(projected[:3], direct, rtol=0, atol=1e-5)


def test_lc_capacitive_rates_have_si_lifetime_and_thermal_detailed_balance():
    assert hasattr(fluxonium, "SingleFluxoniumModel"), "single circuit model is missing"
    model = fluxonium.SingleFluxoniumModel(
        (0.0, 0.3, 1.5), 0.5, cutoff=12, retained_levels=3, temperature_k=0.02
    )
    noise = fluxonium.FluxoniumNoise(q_cap=1e6)
    secular = model.secular_model((0,), noise=noise)
    gap = np.sqrt(8 * 0.3 * 1.5)
    # LC energy lifetime RC=Q/omega; Bose occupation modifies each direction.
    thermal_ratio = np.exp(-0.04799243073366221 * gap / 0.02)
    down = (1 / 83882.020174145) / (1 - thermal_ratio)
    assert secular.rates_per_ns[0, 1] == pytest.approx(down, rel=2e-10)
    assert secular.rates_per_ns[1, 0] == pytest.approx(down * thermal_ratio, rel=2e-10)
    assert secular.rates_per_ns[1, 2] == pytest.approx(2 * down, rel=2e-10)
    assert_allclose(np.diag(secular.rates_per_ns), 0)
    assert_allclose(secular.dephasing_per_ns, 0)
    source = model.model_source(noise=noise)
    weights = np.array([1, thermal_ratio, thermal_ratio**2])
    assert_allclose(
        source.initial_density, np.diag(weights / weights.sum()), atol=1e-12
    )


@pytest.mark.parametrize("temperature", [0.0, 0.04])
def test_purcell_preserves_existing_equilibrium_t1_and_only_couples_01(temperature):
    assert hasattr(fluxonium, "SingleFluxoniumModel"), "single circuit model is missing"
    model = fluxonium.SingleFluxoniumModel(
        (4.0, 1.0, 1.0), 0.5, cutoff=30, retained_levels=3, temperature_k=temperature
    )
    resonator = fluxonium.DispersiveResonator(6.0, 0.03, 0.006)
    secular = model.secular_model((0,), resonator=resonator)
    t1 = fluxonium.calculate_purcell_t1_vs_flux(
        np.array([0.5]),
        6.0,
        0.006 / (2 * np.pi),
        0.03,
        temperature,
        (4.0, 1.0, 1.0),
        progress=False,
    )[0]
    down, up = secular.rates_per_ns[0, 1], secular.rates_per_ns[1, 0]
    assert down > 0
    assert down + up == pytest.approx(1 / t1, rel=1e-10)
    ratio = (
        0
        if temperature == 0
        else np.exp(
            -0.04799243073366221
            * (secular.energies_ghz[1] - secular.energies_ghz[0])
            / temperature
        )
    )
    assert up / down == pytest.approx(ratio, abs=1e-12)
    assert_allclose(secular.rates_per_ns[2], 0)
    assert_allclose(secular.rates_per_ns[:, 2], 0)


def test_readout_pulls_match_independent_dressed_spectrum_and_flux_dephasing():
    assert hasattr(fluxonium, "SingleFluxoniumModel"), "single circuit model is missing"
    model = fluxonium.SingleFluxoniumModel(
        (4.0, 1.0, 1.0), 0.4, cutoff=30, retained_levels=3, temperature_k=0
    )
    resonator = fluxonium.DispersiveResonator(6.0, 0.03, 0.006)
    frequencies = model.readout_frequencies_ghz(resonator)
    reference = fluxonium.calculate_dispersive((4.0, 1.0, 1.0), 0.4, 6.0, 0.03)
    assert frequencies.shape == (3,)
    assert_allclose(frequencies[:2], reference, atol=2e-8)
    pure = model.secular_model((0,), noise=fluxonium.FluxoniumNoise(aphi=1e-6))
    assert pure.dephasing_per_ns[0, 1] > 0
    assert_allclose(pure.dephasing_per_ns, pure.dephasing_per_ns.T, atol=1e-14)
    assert_allclose(np.diag(pure.dephasing_per_ns), 0)


@pytest.mark.parametrize(
    ("params", "flux", "cutoff", "retained"),
    [
        ((4, -1, 1), 0.5, 30, 3),
        ((4, 1, 1), np.nan, 30, 3),
        ((4, 1, 1), 0.5, 2, 3),
        ((4, 1, 1), 0.5, 30, 0),
    ],
)
def test_model_rejects_invalid_circuit_flux_and_truncations(
    params, flux, cutoff, retained
):
    assert hasattr(fluxonium, "SingleFluxoniumModel"), "single circuit model is missing"
    with pytest.raises(ValueError, match="circuit|flux|retained_levels|cutoff"):
        fluxonium.SingleFluxoniumModel(
            params, flux, cutoff=cutoff, retained_levels=retained, temperature_k=0
        )
