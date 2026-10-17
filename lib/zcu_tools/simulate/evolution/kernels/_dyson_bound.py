"""Transition-wise population bound for unpruned second-order Dyson evolution."""

import numpy as np
from numpy.typing import NDArray

from ..secular import SecularModel


def population_bound(
    model: SecularModel,
    port: int,
    params: NDArray[np.float64],
    populations: NDArray[np.float64],
    source_input: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return B2 for each (frequency GHz, gain, duration ns, phase rad) row.

    populations are source-input populations in this model's eigenbasis.
    source_input marks rows starting at zero; all other rows use the universal
    density bound (d0=D=1, cij=1/2, c=2). Phases are deliberately unused:
    coherence is bounded from populations, not assumed absent. The model must
    supply Hermitian drives and a valid secular Lindblad dissipator. This bounds
    driven-minus-free populations of the unpruned second-order approximation,
    not the error against exact dynamics. No density evolution is performed.
    """
    omega = 2 * np.pi * params[:, 0]
    gain, duration = abs(params[:, 1]), params[:, 2]
    drive = model.drives[port]
    plus = (drive.in_phase_ghz - 1j * drive.quadrature_ghz) / 2
    minus = plus.conj().T
    outgoing = model.rates_per_ns.sum(axis=0)
    q_matrix = model.rates_per_ns - np.diag(outgoing)
    q = float(np.linalg.norm(q_matrix, 1))
    q_inf = float(np.linalg.norm(q_matrix, np.inf))
    diagonal = np.zeros(len(params))
    linear = np.zeros(len(params))
    for i in range(len(populations)):
        for j in range(i + 1, len(populations)):
            gamma = (outgoing[i] + outgoing[j]) / 2 + model.dephasing_per_ns[i, j]
            decay = -gamma - 2j * np.pi * (
                model.energies_ghz[i] - model.energies_ghz[j]
            )
            z_minus, z_plus = abs(-1j * omega - decay), abs(1j * omega - decay)
            u_minus = 2 * np.pi * gain * abs(minus[i, j])
            u_plus = 2 * np.pi * gain * abs(plus[i, j])
            d0 = np.where(source_input, abs(populations[j] - populations[i]), 1.0)
            difference = np.minimum(1.0, d0 + q * duration)
            # Near resonance take the finite nonoscillatory S branch. Safe
            # denominators also prevent overflow in the unused refined branch.
            resonant = (z_minus < 1e-12) | (z_plus < 1e-12)
            safe_minus = np.maximum(z_minus, 1e-12)
            safe_plus = np.maximum(z_plus, 1e-12)
            v_minus, v_plus = u_minus / safe_minus, u_plus / safe_plus
            j_minus = np.full(len(params), np.inf)
            j_plus = np.full(len(params), np.inf)
            np.divide(2 + q_inf * duration, z_minus, out=j_minus, where=z_minus > 0)
            np.divide(2 + q_inf * duration, z_plus, out=j_plus, where=z_plus > 0)
            j_minus = np.minimum(duration, j_minus)
            j_plus = np.minimum(duration, j_plus)
            beat = np.full(len(params), np.inf)
            np.divide(
                2 * difference + duration * (q_inf * difference + q),
                abs(2 * omega),
                out=beat,
                where=omega != 0,
            )
            beat = np.minimum(duration * difference, beat)
            refined = (
                2 * gamma * duration * difference * (v_minus**2 + v_plus**2)
                + 2 * u_minus * u_plus * (1 / safe_minus + 1 / safe_plus) * beat
                + 2 * d0 * (v_minus + v_plus) * (u_minus * j_minus + u_plus * j_plus)
                + q * duration**2 * (v_minus + v_plus) * (u_minus + u_plus)
            )
            simple = (u_minus + u_plus) ** 2 * duration**2 * difference
            diagonal += np.where(resonant, simple, np.minimum(simple, refined))
            cij = np.where(source_input, np.sqrt(populations[i] * populations[j]), 0.5)
            linear += 2 * cij * (u_minus * j_minus + u_plus * j_plus)
    size = len(populations)
    coherence = min(
        np.sqrt(max(0.0, size * (1 - np.sum(populations**2)))),
        max(0.0, np.sum(np.sqrt(populations)) ** 2 - 1),
    )
    c = np.where(source_input, coherence, 2.0)
    a = (
        4
        * np.pi
        * gain
        * duration
        * (
            np.linalg.norm(drive.in_phase_ghz, 2)
            + np.linalg.norm(drive.quadrature_ghz, 2)
        )
    )
    return diagonal + np.minimum(linear, c * a) + c * a**2 / 2
