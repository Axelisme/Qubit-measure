"""Independent LC, Boltzmann and CP dephasing references at numeric seams."""

import numpy as np
import pytest


def test_joint_capacitive_lc_directional_and_gibbs() -> None:
    from zcu_tools.simulate.baths import (
        CapacitiveLoss,
        directional_rates,
        gibbs_weights,
    )

    nu = np.sqrt(3.6)
    n = np.array(
        [
            [0, (1.5 / 9.6) ** 0.25, 0],
            [(1.5 / 9.6) ** 0.25, 0, (1.5 / 9.6) ** 0.25 * np.sqrt(2)],
            [0, (1.5 / 9.6) ** 0.25 * np.sqrt(2), 0],
        ],
        dtype=complex,
    )
    energy = np.arange(3) * nu
    bath = CapacitiveLoss(ec_ghz=0.3, quality_factor=1e6)
    cold = directional_rates(energy, n, bath, temperature_k=0)
    assert 1 / cold[0, 1] == pytest.approx(83882.02017414506, rel=1e-10)
    assert cold[1, 2] == pytest.approx(2 * cold[0, 1], rel=1e-12)
    assert np.count_nonzero(np.tril(cold)) == 0
    warm = directional_rates(energy, n, bath, temperature_k=0.08)
    assert warm[1, 0] / warm[0, 1] == pytest.approx(
        np.exp(-0.04799243073366221 * nu / 0.08), rel=1e-12
    )
    p = gibbs_weights(energy, temperature_k=0.08)
    q = warm - np.diag(warm.sum(axis=0))
    np.testing.assert_allclose(q @ p, 0, atol=1e-17, rtol=0)
    np.testing.assert_allclose(
        gibbs_weights(np.array([2.0, 2.0, 4.0]), temperature_k=0),
        [0.5, 0.5, 0],
        rtol=0,
        atol=0,
    )


def test_quasiparticle_low_temperature_is_finite_zero_temperature_limit() -> None:
    from zcu_tools.simulate.baths import QuasiparticleLoss, directional_rates

    energy = np.array([0.0, 1.0])
    operator = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    bath = QuasiparticleLoss(8.0, 1e-6)
    cold = directional_rates(energy, operator, bath, temperature_k=0.0)
    tiny = directional_rates(energy, operator, bath, temperature_k=1e-15)
    assert cold[0, 1] > 0
    np.testing.assert_allclose(tiny, cold, rtol=1e-12, atol=0)


def test_markov_dephasing_collapse_matches_echo_and_photon_rates() -> None:
    from zcu_tools.simulate.dephasing import (
        dephasing_operators,
        flux_dephasing,
        photon_dephasing,
    )

    flux = flux_dephasing(np.array([[0.0, 2.0, -1.0]]), np.array([0.003]))
    expected_flux = (
        np.sqrt(np.log(2)) * 0.003 * np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]]) / 1000
    )
    np.testing.assert_allclose(flux, expected_flux, rtol=1e-14)
    photon = photon_dephasing(
        np.array([[0.0, 0.004, -0.002]]), np.array([0.002]), np.array([0.05])
    )
    assert photon[0, 1] == pytest.approx(8e-8, rel=1e-12)
    gamma = flux + photon
    rho = np.ones((3, 3), dtype=complex) / 3
    change = np.zeros_like(rho)
    for c in dephasing_operators(gamma):
        loss = c.conj().T @ c
        change += c @ rho @ c.conj().T - (loss @ rho + rho @ loss) / 2
    np.testing.assert_allclose(change, -gamma * rho, atol=1e-16, rtol=0)
    np.testing.assert_allclose(np.diag(change), 0, atol=1e-17, rtol=0)
