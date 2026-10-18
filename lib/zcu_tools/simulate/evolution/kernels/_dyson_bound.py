"""Transition-wise population bound for unpruned second-order Dyson evolution."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import ModelSource, SecularModel
from ._dyson_model import parameters
from ._secular import coherence_decay, population_generator


def population_bound(
    model: SecularModel,
    port: int,
    params: NDArray[np.float64],
    initial: NDArray[np.complex128],
    known_input: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return B2 for each (frequency GHz, gain, duration ns, phase rad) row.

    initial is the full known density in this model's eigenbasis.
    known_input marks source or engine-certified prefix inputs; they use actual pair magnitudes
    abs(initial[i,j]) and the off-diagonal trace norm, without zeroing small
    coherence. All other rows use the universal density bound (d0=D=1,
    cij=1/2, c=2). Coherence phases are deliberately unused. The model must
    supply Hermitian drives and a valid secular Lindblad dissipator. This bounds
    driven-minus-free populations of the unpruned second-order approximation,
    not the error against exact dynamics. No density evolution is performed.
    """
    populations = np.maximum(np.diag(initial).real, 0.0)
    off_diagonal = initial - np.diag(np.diag(initial))
    coherence = float(np.linalg.norm(off_diagonal, "nuc"))
    omega = 2 * np.pi * params[:, 0]
    gain, duration = abs(params[:, 1]), params[:, 2]
    drive = model.drives[port]
    plus = (drive.in_phase_ghz - 1j * drive.quadrature_ghz) / 2
    minus = plus.conj().T
    coherence_loss = coherence_decay(model)
    q_matrix = population_generator(model.rates_per_ns)
    q = float(np.linalg.norm(q_matrix, 1))
    q_inf = float(np.linalg.norm(q_matrix, np.inf))
    diagonal = np.zeros(len(params))
    linear = np.zeros(len(params))
    for i in range(len(populations)):
        for j in range(i + 1, len(populations)):
            gamma = coherence_loss[i, j]
            decay = -gamma - 2j * np.pi * (
                model.energies_ghz[i] - model.energies_ghz[j]
            )
            z_minus, z_plus = abs(-1j * omega - decay), abs(1j * omega - decay)
            u_minus = 2 * np.pi * gain * abs(minus[i, j])
            u_plus = 2 * np.pi * gain * abs(plus[i, j])
            d0 = np.where(known_input, abs(populations[j] - populations[i]), 1.0)
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
            cij = np.where(known_input, abs(initial[i, j]), 0.5)
            linear += 2 * cij * (u_minus * j_minus + u_plus * j_plus)
    c = np.where(known_input, coherence, 2.0)
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


def accepts_population_bound(
    table: SegmentTable,
    rows: NDArray[np.int64],
    source: ModelSource,
    model: SecularModel,
    port: int,
    budget: float,
) -> NDArray[np.bool_]:
    """Test B2 using only source or engine-certified known inputs, never workers.

    No-drive shared prefixes retain their B2 exemption. Driven groups without
    a computed input use the arbitrary-input bound until the engine resolves
    their prefix. Basis conversion and scalar bounds do not evolve density.
    """
    params = parameters(table, rows)
    groups = table.prefix_group
    selected: dict[int | None, list[int]] = {}
    for index, row in enumerate(rows):
        group = groups[row] if groups is not None else None
        selected.setdefault(group, []).append(index)
    accepted = np.zeros(len(rows), dtype=bool)
    dc = source.model((0,) * len(model.level))
    for group, indices in selected.items():
        positions = np.array(indices, dtype=np.int64)
        if group is not None and table.prefix_no_drive[group]:
            accepted[positions] = True
            continue
        initial = table.prefix_inputs.get(group) if group is not None else None
        known = np.full(len(positions), initial is not None, dtype=bool)
        known |= table.start_ns[rows[positions]] == 0
        if initial is None:
            initial = source.initial_density
        if not np.array_equal(model.eigenvectors, dc.eigenvectors):
            overlap = model.eigenvectors.conj().T @ dc.eigenvectors
            initial = overlap @ initial @ overlap.conj().T
        bound = population_bound(model, port, params[positions], initial, known)
        accepted[positions] = bound + budget / 8 <= min(0.02, (1.5 * budget) ** (2 / 3))
    return accepted
