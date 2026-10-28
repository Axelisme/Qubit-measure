"""Population modes at the public multilevel coherence seam."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import h, k
from scipy.linalg import expm
from scqubits.core.fluxonium import Fluxonium
from zcu_tools.simulate.fluxonium import (
    FluxoniumNoise,
    Mode,
    calculate_directional_loss_rates,
    calculate_eff_t1_vs_flux_fast,
    calculate_eigensystem,
    calculate_manifold_t1_vs_flux_fast,
)


def test_two_level_manifold_has_only_pair_decay():
    params = (4.0, 1.0, 0.7)
    fluxs = np.array([0.31, 0.5, 0.17])
    channels = [
        ("t1_capacitive", {"Q_cap": 1.7e6}),
        ("t1_inductive", {"Q_ind": 3e8}),
        ("t1_quasiparticle_tunneling", {"x_qp": 1.3e-6}),
    ]
    temperatures = np.array([0.065, 0.05, 0.02])
    expected = calculate_eff_t1_vs_flux_fast(
        params, fluxs, channels, temperatures, cutoff=16, qub_dim=2
    )
    actual = calculate_manifold_t1_vs_flux_fast(
        params, fluxs, channels, temperatures, cutoff=16, qub_dim=2
    )
    assert len(actual) == len(fluxs)
    for modes, lifetime in zip(actual, expected, strict=True):
        assert len(modes) == 1  # The stationary population mode is not a decay.
        mode = modes[0]
        assert isinstance(mode, Mode)
        assert mode.t1_ns == pytest.approx(lifetime, rel=2e-12)
        assert mode.amplitude == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize(
    "channels,noise",
    [
        ([("t1_capacitive", {"Q_cap": 1.7e6})], FluxoniumNoise(q_cap=1.7e6)),
        ([("t1_inductive", {"Q_ind": 3e8})], FluxoniumNoise(q_ind=3e8)),
        (
            [("t1_quasiparticle_tunneling", {"x_qp": 1.3e-6})],
            FluxoniumNoise(xqp=1.3e-6),
        ),
        (["t1_charge_impedance"], None),
        (["t1_flux_bias_line"], None),
    ],
)
def test_manifold_modes_reconstruct_scqubits_directional_decay(channels, noise):
    params, cutoff, levels = (4.0, 1.0, 0.7), 16, 4
    fluxs = np.array([0.31, 0.23])
    temperatures = np.array([0.065, 0.13])
    actual = calculate_manifold_t1_vs_flux_fast(
        params, fluxs, channels, temperatures, cutoff=cutoff, qub_dim=levels
    )
    for flux, temperature, modes in zip(fluxs, temperatures, actual, strict=True):
        esys = calculate_eigensystem(params, flux, cutoff=cutoff, evals_count=levels)
        if noise is not None:
            rates = calculate_directional_loss_rates(
                params, flux, esys, noise, cutoff=cutoff, temperature_k=temperature
            )
        else:
            raw = Fluxonium(*params, flux=flux, cutoff=cutoff, truncated_dim=levels)
            rates = np.zeros((levels, levels))
            for source in range(levels):
                for target in range(levels):
                    if source != target:
                        rates[target, source] = raw.t1_effective(
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
        generator = rates - np.diag(rates.sum(axis=0))
        thermal = np.exp(-(esys[0] - esys[0][0]) * h * 1e9 / (k * temperature))
        thermal /= thermal.sum()
        initial = thermal.copy()
        initial[[0, 1]] = initial[[1, 0]]
        times = np.array([0, 0.1, 1, 5]) / (rates[0, 1] + rates[1, 0])
        expected = np.array(
            [
                (thermal[0] - (expm(generator * time) @ initial)[0])
                / (thermal[0] - initial[0])
                for time in times
            ]
        )
        assert len(modes) == levels - 1
        amplitudes = [mode.amplitude for mode in modes]
        assert amplitudes == sorted(amplitudes, reverse=True)
        assert sum(amplitudes) == pytest.approx(1, abs=2e-13)
        assert all(np.isfinite(mode.t1_ns) and mode.t1_ns > 0 for mode in modes)
        reconstructed = np.array(
            [
                sum(mode.amplitude * np.exp(-time / mode.t1_ns) for mode in modes)
                for time in times
            ]
        )
        assert_allclose(reconstructed, expected, rtol=1e-10, atol=3e-13)


@pytest.mark.parametrize("temperature", [0.05, 0.02])
def test_q2_manifold_includes_thermal_cascade(temperature):
    # Independent numeric oracle: Q2 thermal-control experiment at phi=0.7.
    params, fluxs = (5.71, 1.30, 0.59), np.array([0.7])
    channels = [
        ("t1_capacitive", {"Q_cap": 1e6}),
        ("t1_inductive", {"Q_ind": 1e8}),
        ("t1_quasiparticle_tunneling", {"x_qp": 2e-7}),
    ]
    modes = calculate_manifold_t1_vs_flux_fast(
        params, fluxs, channels, temperature, cutoff=40, qub_dim=6
    )[0]
    pair = calculate_eff_t1_vs_flux_fast(
        params, fluxs, channels, temperature, cutoff=40, qub_dim=6
    )[0]
    if temperature == 0.05:
        assert modes[0].t1_ns == pytest.approx(433e3, rel=0.02)
        assert pair == pytest.approx(606e3, rel=0.02)
    else:
        assert modes[0].t1_ns == pytest.approx(pair, rel=0.02)
    assert len(modes) == 5
    assert sum(mode.amplitude for mode in modes) == pytest.approx(1, abs=2e-13)


def test_zero_temperature_manifold_retains_zero_amplitude_modes():
    # An LC oscillator prepared in |1> decays at the SI 1/(Q*sqrt(LC)) rate.
    # Higher population modes exist but have exactly zero observable amplitude.
    from scipy.constants import e, hbar

    quality, params = 1e6, (0.0, 0.3, 1.5)
    modes = calculate_manifold_t1_vs_flux_fast(
        params,
        np.array([0.17]),
        [("t1_capacitive", {"Q_cap": quality})],
        0.0,
        cutoff=12,
        qub_dim=4,
    )[0]
    capacitance = e**2 / (2 * params[1] * 1e9 * h)
    inductance = (hbar / (2 * e)) ** 2 / (params[2] * 1e9 * h)
    assert len(modes) == 3
    assert modes[0].t1_ns == pytest.approx(
        quality * np.sqrt(inductance * capacitance) * 1e9, rel=1e-12
    )
    assert_allclose([mode.amplitude for mode in modes], [1, 0, 0], atol=1e-14)


def test_manifold_array_options_preserve_flux_and_temperature_alignment():
    params = (4.0, 1.0, 0.7)
    fluxs, temperatures = np.array([0.31, 0.23]), np.array([0.02, 0.13])
    qualities = np.array([1e6, 3e6])
    modes = calculate_manifold_t1_vs_flux_fast(
        params,
        fluxs,
        [("t1_capacitive", {"Q_cap": qualities})],
        temperatures,
        cutoff=16,
        qub_dim=4,
    )
    for flux, temperature, quality, actual in zip(
        fluxs, temperatures, qualities, modes, strict=True
    ):
        expected = calculate_manifold_t1_vs_flux_fast(
            params,
            np.array([flux]),
            [("t1_capacitive", {"Q_cap": quality})],
            float(temperature),
            cutoff=16,
            qub_dim=4,
        )[0]
        assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize(
    "temperature", [-0.1, np.nan, np.array([0.0]), np.array([0.1, 0.2])]
)
def test_manifold_rejects_invalid_temperatures(temperature):
    with pytest.raises(ValueError, match="Temp"):
        calculate_manifold_t1_vs_flux_fast(
            (4.0, 1.0, 0.7),
            np.array([0.31]),
            ["t1_capacitive"],
            temperature,
            cutoff=16,
            qub_dim=4,
        )


def test_manifold_empty_flux_and_zero_loss_have_no_modes():
    assert (
        calculate_manifold_t1_vs_flux_fast(
            (4.0, 1.0, 0.7),
            np.array([]),
            ["t1_capacitive"],
            0.05,
            cutoff=16,
            qub_dim=4,
        )
        == []
    )
    assert calculate_manifold_t1_vs_flux_fast(
        (4.0, 1.0, 0.7),
        np.array([0.31]),
        [],
        0.05,
        cutoff=16,
        qub_dim=4,
    ) == [[]]
