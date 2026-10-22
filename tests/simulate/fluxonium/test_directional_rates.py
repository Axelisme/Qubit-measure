"""Directional coherence contracts, independent of per-circuit operator reuse.

Small native circuits exercise every loss direction. Raw scqubits is the channel
reference; SI LC lifetime, Boltzmann balance and parity supply physical oracles.
"""

import numpy as np
import pytest
import scipy.constants as const
from numpy.testing import assert_allclose
from scqubits.core.fluxonium import Fluxonium
from zcu_tools.simulate.fluxonium import (
    DispersiveResonator,
    SingleFluxoniumModel,
    calculate_eff_t1_with,
    calculate_eigensystem,
)
from zcu_tools.simulate.fluxonium.coherence.rates import (
    FluxoniumNoise,
    calculate_directional_loss_rates,
)


@pytest.mark.parametrize(
    "noise,channels",
    [
        (FluxoniumNoise(), []),
        (FluxoniumNoise(q_cap=1.7e6), [("t1_capacitive", {"Q_cap": 1.7e6})]),
        (FluxoniumNoise(q_ind=3e8), [("t1_inductive", {"Q_ind": 3e8})]),
        (
            FluxoniumNoise(xqp=1.3e-6),
            [("t1_quasiparticle_tunneling", {"x_qp": 1.3e-6})],
        ),
        (
            FluxoniumNoise(q_cap=1.7e6, q_ind=3e8, xqp=1.3e-6, aphi=2e-6),
            [
                ("t1_capacitive", {"Q_cap": 1.7e6}),
                ("t1_inductive", {"Q_ind": 3e8}),
                ("t1_quasiparticle_tunneling", {"x_qp": 1.3e-6}),
            ],
        ),
    ],
)
def test_directional_channels_preserve_scalar_lifetimes_and_balance(noise, channels):
    params, flux, temperature = (4.0, 1.0, 0.7), 0.31, 0.065
    esys = calculate_eigensystem(params, flux, cutoff=16, evals_count=4)
    actual = calculate_directional_loss_rates(
        params, flux, esys, noise, cutoff=16, temperature_k=temperature
    )
    raw = Fluxonium(*params, flux=flux, cutoff=16, truncated_dim=4)
    expected = np.zeros((4, 4))
    for source in range(4):
        for target in range(4):
            if source == target:
                continue
            expected[target, source] = raw.t1_effective(
                noise_channels=channels,
                common_noise_options={
                    "i": source,
                    "j": target,
                    "T": temperature,
                    "total": False,
                },
                esys=esys,
                get_rate=True,
            )
    assert_allclose(actual, expected, rtol=2e-13, atol=1e-25)
    assert_allclose(np.diag(actual), 0, atol=0)
    assert np.all(actual >= 0)
    for lower in range(3):
        for upper in range(lower + 1, 4):
            ratio = np.exp(
                -const.h
                * 1e9
                * (esys[0][upper] - esys[0][lower])
                / (const.k * temperature)
            )
            assert actual[upper, lower] == pytest.approx(
                actual[lower, upper] * ratio, rel=2e-12, abs=1e-25
            )
    # The existing scalar seam still distinguishes equilibrium from directional T1.
    down = calculate_eff_t1_with(flux, channels, temperature, raw, esys, total=False)
    equilibrium = calculate_eff_t1_with(flux, channels, temperature, raw, esys)
    assert 1 / down == pytest.approx(actual[0, 1], rel=2e-13, abs=1e-25)
    assert 1 / equilibrium == pytest.approx(
        actual[0, 1] + actual[1, 0], rel=2e-13, abs=1e-25
    )


@pytest.mark.parametrize("channel", ["capacitive", "inductive"])
def test_zero_temperature_lc_loss_has_si_lifetime_and_no_excitation(channel):
    params, quality = (0.0, 0.3, 1.5), 1e6
    esys = calculate_eigensystem(params, 0.17, cutoff=12, evals_count=3)
    noise = (
        FluxoniumNoise(q_cap=quality)
        if channel == "capacitive"
        else FluxoniumNoise(q_ind=quality)
    )
    rates = calculate_directional_loss_rates(
        params, 0.17, esys, noise, cutoff=12, temperature_k=0
    )
    capacitance = const.e**2 / (2 * params[1] * 1e9 * const.h)
    inductance = (const.hbar / (2 * const.e)) ** 2 / (params[2] * 1e9 * const.h)
    spontaneous = 1 / (quality * np.sqrt(inductance * capacitance) * 1e9)
    assert rates[0, 1] == pytest.approx(spontaneous, rel=1e-12)
    assert rates[1, 2] == pytest.approx(2 * spontaneous, rel=1e-12)
    assert_allclose(np.tril(rates), 0, atol=0)
    assert rates[0, 2] < 1e-25


def test_quasiparticle_parity_and_distinct_circuit_calls_are_isolated():
    noise = FluxoniumNoise(xqp=2e-7)
    cases = [((4.0, 1.0, 0.5), 0.5, 18), ((5.0, 0.8, 0.7), 0.29, 16)]
    results = []
    # Repeat the first case after a different flux/circuit/cutoff, not a shared cache.
    for params, flux, cutoff in [*cases, cases[0]]:
        esys = calculate_eigensystem(params, flux, cutoff=cutoff, evals_count=3)
        rates = calculate_directional_loss_rates(
            params, flux, esys, noise, cutoff=cutoff, temperature_k=0.05
        )
        results.append(rates)
    assert results[0][0, 1] < 1e-20  # forbidden at half flux, compare rate not huge T1
    assert results[0][0, 2] > 1e-10  # allowed transition is not accidentally disabled
    assert results[1][0, 1] > 1e-10
    assert_allclose(results[0], results[2], rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid", ["flux", "temperature", "order", "shape", "orthogonal"]
)
def test_directional_rates_reject_invalid_native_eigenpairs_and_scales(invalid):
    params = (4.0, 1.0, 1.0)
    energy, vectors = calculate_eigensystem(params, 0.31, cutoff=12, evals_count=3)
    flux, temperature = 0.31, 0.05
    if invalid == "flux":
        flux = np.nan
    elif invalid == "temperature":
        temperature = -0.01
    elif invalid == "order":
        energy = energy[::-1]
    elif invalid == "shape":
        vectors = vectors[:-1]
    else:
        vectors = 2 * vectors
    with pytest.raises(ValueError, match="ordered native eigenpairs"):
        calculate_directional_loss_rates(
            params,
            flux,
            (energy, vectors),
            FluxoniumNoise(),
            cutoff=12,
            temperature_k=temperature,
        )


def test_secular_projected_losses_and_nonzero_photon_dephasing():
    params, dc, temperature = (4.0, 1.0, 1.0), 0.4, 0.05
    model = SingleFluxoniumModel(
        params, dc, cutoff=18, retained_levels=3, temperature_k=temperature
    )
    noise = FluxoniumNoise(q_cap=1e6, q_ind=1e8, xqp=2e-7)
    secular = model.secular_model((20,), noise=noise)
    _, dc_vectors = calculate_eigensystem(params, dc, cutoff=18, evals_count=3)
    projected = (secular.energies_ghz, dc_vectors @ secular.eigenvectors)
    raw = Fluxonium(*params, flux=0.42, cutoff=18, truncated_dim=3)
    expected = np.zeros((3, 3))
    for source in range(3):
        for target in range(3):
            if source != target:
                expected[target, source] = raw.t1_effective(
                    noise_channels=[
                        ("t1_capacitive", {"Q_cap": 1e6}),
                        ("t1_inductive", {"Q_ind": 1e8}),
                        ("t1_quasiparticle_tunneling", {"x_qp": 2e-7}),
                    ],
                    common_noise_options={
                        "i": source,
                        "j": target,
                        "T": temperature,
                        "total": False,
                    },
                    esys=projected,
                    get_rate=True,
                )
    assert_allclose(secular.rates_per_ns, expected, rtol=2e-12, atol=1e-25)
    resonator = DispersiveResonator(6.0, 0.03, 0.006, thermal_occupation=0.02)
    thermal = model.secular_model((0,), noise=noise, resonator=resonator)
    vacuum = model.secular_model(
        (0,),
        noise=noise,
        resonator=DispersiveResonator(6.0, 0.03, 0.006, thermal_occupation=0),
    )
    # Pure residual photons do not change directional loss or Purcell rates.
    assert_allclose(thermal.rates_per_ns, vacuum.rates_per_ns, rtol=0, atol=0)
    frequencies = model.readout_frequencies_ghz(resonator)
    delta = 2 * np.pi * (frequencies[:, None] - frequencies[None, :])
    expected_gamma = 0.02 * 0.006 * delta**2 / (0.006**2 + delta**2)
    assert_allclose(thermal.dephasing_per_ns, expected_gamma, rtol=1e-7, atol=1e-14)
    assert_allclose(vacuum.dephasing_per_ns, 0, atol=0)


def test_scalar_coherence_propagates_invalid_channel_and_option_errors():
    raw = Fluxonium(4.0, 1.0, 1.0, flux=0.31, cutoff=12, truncated_dim=3)
    with pytest.raises(ValueError, match="Only t1 channels"):
        calculate_eff_t1_with(0.31, ["tphi_1_over_f_flux"], 0.05, raw)
    with pytest.raises(TypeError, match="unexpected keyword"):
        calculate_eff_t1_with(
            0.31, [("t1_capacitive", {"unknown_option": 1})], 0.05, raw
        )
    with pytest.raises(ValueError, match="distinct nonnegative"):
        calculate_eff_t1_with(0.31, ["t1_capacitive"], 0.05, raw, transition=(1, 1))
