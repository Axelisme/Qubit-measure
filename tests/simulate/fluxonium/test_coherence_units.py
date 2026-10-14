"""Physical ns at the public scalar and sweep prediction boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as const
from scqubits.core.fluxonium import Fluxonium
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1,
    calculate_eff_t1_fast,
    calculate_eff_t1_vs_flux,
    calculate_eff_t1_vs_flux_fast,
    calculate_eff_t1_vs_flux_with,
    calculate_eff_t1_with,
)


@pytest.mark.parametrize(
    "path", ["slow", "fast", "slow-with", "slow-sweep", "fast-sweep", "slow-sweep-with"]
)
def test_lc_energy_lifetime_is_physical_ns(path: str) -> None:
    params = (0.0, 0.3, 1.5)
    q = 1e6
    # SI circuit reference: EC=e²/(2C), EL=(ħ/2e)²/L, energy lifetime=Q/ω.
    capacitance = const.e**2 / (2 * params[1] * 1e9 * const.h)
    inductance = (const.hbar / (2 * const.e)) ** 2 / (params[2] * 1e9 * const.h)
    expected_ns = q * np.sqrt(inductance * capacitance) * 1e9
    noise = [("t1_capacitive", {"Q_cap": q})]
    fluxs = np.array([0.17, 0.32])
    qubit = Fluxonium(*params, flux=0.17, cutoff=20, truncated_dim=4)
    if path == "slow":
        actual = calculate_eff_t1(0.17, noise, 0.001, params, cutoff=20, evals_count=4)
    elif path == "fast":
        actual = calculate_eff_t1_fast(0.17, params, noise, 0.001, cutoff=20, qub_dim=4)
    elif path == "slow-with":
        actual = calculate_eff_t1_with(0.17, noise, 0.001, qubit)
    elif path == "slow-sweep":
        actual = calculate_eff_t1_vs_flux(
            fluxs, noise, 0.001, params, cutoff=20, evals_count=4
        )
    elif path == "fast-sweep":
        actual = calculate_eff_t1_vs_flux_fast(
            params, fluxs, noise, 0.001, cutoff=20, qub_dim=4
        )
    else:
        spectrum = qubit.get_spectrum_vs_paramvals(
            "flux", fluxs, evals_count=4, get_eigenstates=True
        )
        actual = calculate_eff_t1_vs_flux_with(fluxs, noise, 0.001, qubit, spectrum)
    np.testing.assert_allclose(actual, expected_ns, rtol=1e-10)


@pytest.mark.parametrize(
    "channel,options",
    [
        ("t1_capacitive", {"Q_cap": 1.7e6}),
        ("t1_inductive", {"Q_ind": 3e8}),
        ("t1_charge_impedance", {"Z": 63.0}),
        ("t1_flux_bias_line", {"M": 520.0, "Z": 41.0}),
        ("t1_quasiparticle_tunneling", {"x_qp": 1.3e-6, "Delta": 4e-4}),
    ],
)
@pytest.mark.parametrize("transition", [(1, 0), (0, 1), (2, 1)])
def test_supported_channels_match_raw_scqubits_ns(channel, options, transition) -> None:
    params = (3.469, 0.952, 0.582)
    fluxs = np.array([0.12, 0.31, 0.46])
    temp = 0.065
    i, j = transition
    noise = [(channel, options)]
    qubit = Fluxonium(*params, flux=0.0, cutoff=30, truncated_dim=4)
    expected = []
    for flux in fluxs:
        qubit.flux = flux
        expected.append(
            qubit.t1_effective(
                noise_channels=noise,
                common_noise_options={"i": i, "j": j, "T": temp, "total": True},
            )
        )
    fast = calculate_eff_t1_vs_flux_fast(
        params, fluxs, noise, temp, cutoff=30, qub_dim=4, i=i, j=j
    )
    np.testing.assert_allclose(fast, expected, rtol=1e-9)
    # Slow helpers currently fix 1->0 but preserve scqubits total=False.
    if transition == (1, 0):
        slow = calculate_eff_t1_vs_flux(
            fluxs, noise, temp, params, cutoff=30, evals_count=4, total=True
        )
        np.testing.assert_allclose(slow, expected, rtol=1e-9)
        for total in (True, False):
            qubit.flux = fluxs[0]
            raw = qubit.t1_effective(
                noise_channels=noise,
                common_noise_options={"i": 1, "j": 0, "T": temp, "total": total},
            )
            scalar = calculate_eff_t1(
                fluxs[0], noise, temp, params, cutoff=30, evals_count=4, total=total
            )
            np.testing.assert_allclose(scalar, raw, rtol=1e-9)
