"""Multilevel population-decay modes of a thermally prepared fluxonium."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.constants import h, k

from ..operators import validate_circuit
from ._fast_common import (
    NoiseChannel,
    SpectralDensity,
    resolve_channels,
    sweep_t1_basis,
    t1_operators,
    validate_temperature,
)


class Mode(NamedTuple):
    """One non-equilibrium population mode.

    t1_ns is the positive finite decay lifetime in ns. amplitude is its signed
    real weight in the equilibrium-subtracted, unit-initial-amplitude observable.
    Zero amplitudes are retained; amplitudes sum to one across decaying modes.
    """

    t1_ns: float
    amplitude: float


def calculate_manifold_t1_vs_flux_fast(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    noise_channels: Sequence[NoiseChannel],
    Temp: float | NDArray[np.float64],
    *,
    cutoff: int = 40,
    qub_dim: int = 20,
) -> list[list[Mode]]:
    """Return all population-decay modes per flux, sorted by signed amplitude.

    params=(EJ,EC,EL) are finite GHz with EJ >=0 and EC,EL >0. fluxs is finite
    1D reduced flux Phi/Phi0. Temp is nonnegative finite scalar K or a positive
    finite array matching fluxs. cutoff is an integer oscillator dimension >=2;
    qub_dim is an integer retained manifold size in [2,cutoff].
    noise_channels has the same five T1 channels and per-channel spectral options
    as calculate_eff_t1_vs_flux_fast. Pure dephasing does not enter these rates.

    All retained state pairs contribute directional rates in ns^-1. The initial
    population is Gibbs(Temp) with populations 0 and 1 exchanged by an ideal pi
    pulse. The observable is 1-p0(t). Its stationary contribution is removed,
    then its decaying amplitudes are normalized to sum to one. Each Mode holds
    a lifetime in ns and a signed real amplitude. Output preserves flux order.
    No modes are merged or discarded for small/zero amplitude. Equilibrium
    eigenvalues within 100 machine eps of the generator's spectral scale are
    excluded; a zero-loss generator returns an empty list.

    Invalid circuit, flux, temperature, manifold size or directional rates raise
    ValueError. Unsupported channels/options raise the same errors as the pair
    fast path. Non-real decay rates or amplitudes (imaginary part >1e-10 times
    their scale), unstable generators and unnormalizable amplitudes raise
    ValueError; eigensolver failures propagate. Inputs and caches are not mutated.
    """
    validate_circuit(params, cutoff)
    fluxs = np.asarray(fluxs, dtype=np.float64)
    if fluxs.ndim != 1 or not np.isfinite(fluxs).all():
        raise ValueError("fluxs must be a finite 1D reduced-flux array")
    if type(qub_dim) is not int or not 2 <= qub_dim <= cutoff:
        raise ValueError("qub_dim must be an integer in [2,cutoff]")
    validate_temperature(Temp, fluxs.shape)
    channels = resolve_channels(noise_channels, *params)
    if not len(fluxs):
        return []
    energies, operators = sweep_t1_basis(
        t1_operators(params, cutoff, qub_dim),
        fluxs,
        qub_dim,
        {key for key, _ in channels},
    )
    rates = np.zeros((len(fluxs), qub_dim, qub_dim))
    for lower in range(qub_dim):
        for upper in range(lower + 1, qub_dim):
            omega = 2 * np.pi * (energies[:, upper] - energies[:, lower])
            for key, spectrum in channels:
                matrix_element = np.abs(operators[key][:, lower, upper]) ** 2
                rates[:, lower, upper] += matrix_element * _directional_spectrum(
                    spectrum, omega, Temp
                )
                rates[:, upper, lower] += matrix_element * _directional_spectrum(
                    spectrum, -omega, Temp
                )
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("loss channels produced invalid directional rates")
    temperatures = np.broadcast_to(Temp, (len(fluxs),))
    return [
        _population_modes(rate, energy, float(temperature))
        for rate, energy, temperature in zip(rates, energies, temperatures, strict=True)
    ]


def _directional_spectrum(
    spectrum: SpectralDensity,
    omega: NDArray[np.float64],
    temperature: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate signed spectra, preserving per-flux option-array alignment."""
    if not isinstance(temperature, np.ndarray):
        return np.asarray(np.real(spectrum(omega, temperature)), dtype=np.float64)
    result = np.empty_like(omega)
    for value in np.unique(temperature):
        mask = temperature == value
        evaluated = np.broadcast_to(np.real(spectrum(omega, float(value))), omega.shape)
        result[mask] = evaluated[mask]
    return result


def _population_modes(
    rates: NDArray[np.float64], energies: NDArray[np.float64], temperature: float
) -> list[Mode]:
    """Decompose equilibrium-subtracted 1-p0 for a Gibbs pi population."""
    thermal = np.zeros(len(energies))
    if temperature == 0:
        thermal[0] = 1
    else:
        thermal = np.exp(-(energies - energies[0]) * h * 1e9 / (k * temperature))
        thermal /= thermal.sum()
    initial = thermal.copy()
    initial[[0, 1]] = initial[[1, 0]]
    generator = rates - np.diag(rates.sum(axis=0))
    raw_values, raw_vectors = np.linalg.eig(generator)
    values = np.asarray(raw_values, dtype=np.complex128)
    vectors = np.asarray(raw_vectors, dtype=np.complex128)
    scale = float(np.max(np.abs(values)))
    tolerance = 100 * np.finfo(float).eps * scale
    if np.any(np.abs(values.imag) > 1e-10 * scale):
        raise ValueError("population generator has non-real decay rates")
    if np.any(values.real > tolerance):
        raise ValueError("population generator is unstable")
    decay = values.real < -tolerance
    if not np.any(decay):
        return []
    amplitudes = -vectors[0, :] * np.linalg.solve(vectors, initial - thermal)
    amplitudes = amplitudes[decay]
    amplitude_scale = float(np.max(np.abs(amplitudes)))
    if np.any(np.abs(amplitudes.imag) > 1e-10 * amplitude_scale):
        raise ValueError("population modes have non-real amplitudes")
    real_amplitudes = amplitudes.real
    total = float(real_amplitudes.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("population amplitudes cannot be normalized")
    modes = [
        Mode(float(-1 / value), float(amplitude / total))
        for value, amplitude in zip(values.real[decay], real_amplitudes, strict=True)
    ]
    return sorted(modes, key=lambda mode: mode.amplitude, reverse=True)
