"""Parent virtual sums independently include states outside the evolution M."""

import numpy as np


def test_virtual_transitions_include_full_local_parent_and_counterrotation() -> None:
    import scqubits as scq
    from zcu_tools.simulate.ftf import (
        FluxoniumCircuit,
        JointModel,
        JointParameters,
        Resonator,
        SquidCircuit,
    )

    f1 = FluxoniumCircuit(3.0, 0.8, 0.6, 12, 3)
    f2 = FluxoniumCircuit(3.3, 0.7, 0.55, 12, 2)
    c = SquidCircuit(10.0, 0.3, 0.2, 4, 2)
    flux = np.array([0.35, 0.22, 0.4])
    model = JointModel(
        JointParameters(f1, c, f2, 0.0, 0.0, 0.0, 6), flux, temperature_k=0.0
    )
    ref1 = scq.Fluxonium(EJ=3.0, EC=0.8, EL=0.6, cutoff=12, flux=0.35, truncated_dim=3)
    ref2 = scq.Fluxonium(EJ=3.3, EC=0.7, EL=0.55, cutoff=12, flux=0.4, truncated_dim=2)
    e1, v1 = np.linalg.eigh(ref1.hamiltonian())
    e2 = np.linalg.eigvalsh(ref2.hamiltonian())[:2]
    n1 = v1[:, :3].conj().T @ ref1.n_operator() @ v1[:, :3]
    numbers = np.arange(-4, 5)
    shift = np.eye(9, k=1, dtype=complex)
    hc = (
        np.diag(4 * 0.3 * numbers**2)
        - 10 * np.cos(np.pi * 0.22) * (shift + shift.T) / 2
        + 2 * np.sin(np.pi * 0.22) * (shift - shift.T) / (2j)
    )
    ec, vc = np.linalg.eigh(hc)
    nc = vc[:, :2].conj().T @ np.diag(numbers) @ vc[:, :2]
    states = sorted(
        np.ndindex(3, 2, 2),
        key=lambda state: e1[state[0]] + ec[state[1]] + e2[state[2]],
    )[:6]
    resonators = (
        Resonator(6.0, (0.03, 0.0, 0.0), 0.001, 0.0001, 0.0),
        Resonator(6.2, (0.0, 0.02, 0.0), 0.001, 0.0002, 0.0),
    )
    expected = np.empty((6, 2))
    for i, (a, b, _d) in enumerate(states):
        for r, (energy, charge, state, coupling, frequency) in enumerate(
            ((e1[:3], n1, a, 0.03, 6.0), (ec[:2], nc, b, 0.02, 6.2))
        ):
            gap = energy[state] - energy
            expected[i, r] = frequency + np.sum(
                coupling**2
                * np.abs(charge[state]) ** 2
                * (1 / (gap + frequency) + 1 / (gap - frequency))
            )
    np.testing.assert_allclose(
        model.readout_frequencies_ghz(flux, resonators), expected, atol=1e-12
    )
