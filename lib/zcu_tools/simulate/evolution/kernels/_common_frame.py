"""Model-only NumPy builders and native common-frame apply algebra.

Small coefficient/remainder builds use NumPy directly to avoid cold JIT work.
Only repeated row admission and propagation are compiled. Frequencies use GHz.
"""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._envelope import sample_value
from ._secular import split_step


class Channels(NamedTuple):
    energies: NDArray[np.float64]
    operators: NDArray[np.complex128]
    omega: NDArray[np.float64]
    first: NDArray[np.int64]
    second: NDArray[np.int64]
    generators: NDArray[np.complex128]
    frame: NDArray[np.float64]
    signed: NDArray[np.complex128]
    denominator_min: float


def coefficients(energies, full, frequencies) -> Channels:
    size, count = len(energies), 2 * len(frequencies)
    carrier = float(np.mean(frequencies))
    labels = np.rint(energies / carrier)
    difference = energies[:, None] - energies[None, :]
    label_difference = labels[:, None] - labels[None, :]
    signs = np.tile([1, -1], len(frequencies))
    signed_frequencies = signs * np.repeat(frequencies, 2)
    signed = np.empty((count, size, size), np.complex128)
    signed[::2], signed[1::2] = full, full.conj().transpose(0, 2, 1)
    retained = label_difference + signs[:, None, None] == 0
    denominators = difference + signed_frequencies[:, None, None]
    fast = ~retained & (np.abs(signed) > 1e-16)
    minimum = float(np.min(np.abs(denominators[fast]), initial=np.inf))
    generators = np.zeros_like(signed)
    np.divide(
        signed,
        denominators,
        out=generators,
        where=fast & (np.abs(denominators) > 1e-10),
    )
    operators = np.zeros((count + count * count, size, size), np.complex128)
    operators[:count] = np.where(retained, signed, 0)
    omega = np.zeros(len(operators))
    omega[:count] = 2 * np.pi * (signed_frequencies - signs * carrier)
    first = np.empty(len(operators), np.int64)
    first[:count] = np.arange(count)
    second = np.full(len(operators), -1, np.int64)
    for a in range(count):
        da = denominators[a]
        left = np.where(~retained[a] & (np.abs(da) > 1e-10), signed[a], 0)
        left_divided = np.zeros_like(left)
        np.divide(left, da, out=left_divided, where=np.abs(da) > 1e-10)
        for b in range(count):
            k = count + a * count + b
            first[k], second[k] = a, b
            omega[k] = (
                2
                * np.pi
                * (
                    signed_frequencies[a]
                    + signed_frequencies[b]
                    - (signs[a] + signs[b]) * carrier
                )
            )
            # db[j,m] belongs to the right factor's transposed coordinates.
            db = (difference - signed_frequencies[b]).T
            right = np.where(np.abs(db) > 1e-10, signed[b], 0)
            right_divided = np.zeros_like(right)
            np.divide(right, db, out=right_divided, where=np.abs(db) > 1e-10)
            beat = (left_divided @ right + left @ right_divided) / 2
            operators[k] = np.where(
                label_difference + signs[a] + signs[b] == 0, beat, 0
            )
    return Channels(
        energies - labels * carrier,
        operators,
        omega,
        first,
        second,
        generators,
        labels * carrier,
        signed,
        minimum,
    )


def remainder_channels(channels, labels, groups, powers):
    """Aggregate absolute ordered SW remainders by equal signed monomials.

    Absolute values precede aggregation, preserving the ordered-channel bound.
    Only one matrix per commutator is transient, never a dense q**4 tensor.
    groups maps base-five exponent encodings to packed monomial indices.
    """
    energies, ops, generators, signed = (
        channels.energies,
        channels.operators,
        channels.generators,
        channels.signed,
    )
    size, count = len(energies), len(generators)
    result = np.zeros((np.max(groups) + 1, size, size))
    for a in range(count):
        for b in range(count):
            v = (ops[b] + signed[b]) / 2
            term = generators[a] @ v - v @ generators[a]
            k = groups[powers[a] + powers[b]]
            sign = 2 - 2 * (a % 2) - 2 * (b % 2)
            for i in range(size):
                for j in range(size):
                    if labels[i] - labels[j] + sign != 0:
                        result[k, i, j] += abs(term[i, j])
            for c in range(count):
                v = ops[c] / 2 + (signed[c] - ops[c]) / 3
                inner = generators[b] @ v - v @ generators[b]
                term = generators[a] @ inner - inner @ generators[a]
                k = groups[powers[a] + powers[b] + powers[c]]
                result[k] += np.abs(term)
                for d in range(count):
                    v = ops[d] / 6 + (signed[d] - ops[d]) / 8
                    inner = generators[c] @ v - v @ generators[c]
                    inner = generators[b] @ inner - inner @ generators[b]
                    term = generators[a] @ inner - inner @ generators[a]
                    k = groups[powers[a] + powers[b] + powers[c] + powers[d]]
                    result[k] += np.abs(term)
    return result


@njit(cache=True, nogil=True)
def remainder_estimates(data, gains, slopes, duration, gamma):
    energies, frequencies, remainder, monomials, degrees, generators = data
    count, size = len(generators), len(energies)
    estimates = np.zeros((3, size, size))
    derivative = np.zeros((size, size))
    bath = np.zeros_like(derivative)
    eta_matrix = np.zeros_like(derivative)
    for a in range(count):
        for i in range(size):
            for j in range(size):
                detuning = abs(energies[i] - energies[j] + frequencies[a])
                window = min(duration, 1 / (np.pi * max(detuning, 1e-14)))
                strength = abs(generators[a, i, j])
                eta_matrix[i, j] += gains[a // 2] * strength
                derivative[i, j] += 2 * slopes[a // 2] * strength * window
                bath[i, j] += 2 * gamma * gains[a // 2] * strength * window
    for k in range(len(monomials)):
        amplitude, frequency = 1.0, 0.0
        for a in range(count):
            amplitude *= gains[a // 2] ** monomials[k, a]
            frequency += frequencies[a] * monomials[k, a]
        for i in range(size):
            for j in range(size):
                detuning = abs(energies[i] - energies[j] + frequency)
                window = min(2 * np.pi * duration, 2 / max(detuning, 1e-14))
                estimates[degrees[k] - 2, i, j] += (
                    amplitude * remainder[k, i, j] * window
                )
    eta = np.max(np.sum(eta_matrix, axis=1))
    return (
        eta,
        np.max(np.sum(estimates[0], axis=1)),
        np.max(np.sum(estimates[1], axis=1)),
        np.max(np.sum(estimates[2], axis=1)),
        np.max(np.sum(derivative, axis=1)),
        np.max(np.sum(bath, axis=1)) + duration * gamma * eta**2,
    )


@njit(cache=True, nogil=True)
def amplitudes(time, args):
    gain, theta, offset, times, values, counts = args
    result = np.empty(2 * len(gain), np.complex128)
    for a in range(len(gain)):
        envelope = sample_value(
            time + offset[a], times[a, : counts[a]], values[a, : counts[a]]
        )
        z = gain[a] * envelope * np.exp(1j * theta[a])
        result[2 * a], result[2 * a + 1] = z, z.conjugate()
    return result


@njit(cache=True, nogil=True)
def hamiltonian(time, channels, args):
    z = amplitudes(time, args)
    h = np.diag(channels.energies.astype(np.complex128))
    for k in range(len(channels.operators)):
        value = z[channels.first[k]] * np.exp(1j * channels.omega[k] * time)
        if channels.second[k] >= 0:
            value *= z[channels.second[k]]
        h += channels.operators[k] * value
    return h


@njit(cache=True, nogil=True)
def unitary(h, duration):
    energies, vectors = np.linalg.eigh(h)
    return (vectors * np.exp(-2j * np.pi * duration * energies)) @ vectors.conj().T


@njit(cache=True, nogil=True)
def propagate(channels, args, control, rho, bath):
    duration, steps = control
    populations, coherences = bath
    dt = duration / steps
    a, b = (3 - 2 * np.sqrt(3)) / 12, (3 + 2 * np.sqrt(3)) / 12
    c1, c2 = 0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6
    for k in range(steps):
        h1 = hamiltonian((k + c1) * dt, channels, args)
        h2 = hamiltonian((k + c2) * dt, channels, args)
        u = unitary(a * h1 + b * h2, dt) @ unitary(b * h1 + a * h2, dt)
        rho = split_step(rho, u, populations, coherences)
    return rho
