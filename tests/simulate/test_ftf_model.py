"""FTF fixed-coordinate circuit and coupled Gibbs contracts."""

import numpy as np
from scipy.linalg import expm


def test_native_embedding_preserves_physical_operators_across_truncations() -> None:
    """Compare physical native coordinates, not matching energy-index labels."""
    import scqubits as scq
    from zcu_tools.simulate.ftf import (
        FluxoniumCircuit,
        JointModel,
        JointParameters,
        SquidCircuit,
    )

    f = FluxoniumCircuit(3.0, 0.8, 0.6, 6, 3)
    c = SquidCircuit(10.0, 0.3, 0.2, 2, 3)
    flux = np.array([0.35, 0.22, 0.4])
    model = JointModel(
        JointParameters(f, c, f, 0.05, 0.06, -0.04, 8), flux, temperature_k=0.04
    )
    assert callable(getattr(model, "native_embedding", None)), (
        "authors need a physical native-coordinate map for cross-basis observables"
    )
    embedding = model.native_embedding()
    assert embedding.shape == (6, 5, 6, 8)
    vectors = embedding.reshape(180, 8)
    np.testing.assert_allclose(
        vectors.conj().T @ vectors, np.eye(8), rtol=0, atol=1e-12
    )

    # Independent circuit package supplies oscillator H/n; the SQUID uses a
    # charge-basis hopping Hamiltonian with its fixed-theta complex phase.
    circuits = [
        scq.Fluxonium(EJ=3, EC=0.8, EL=0.6, flux=value, cutoff=6)
        for value in (flux[0], flux[2])
    ]
    numbers = np.arange(-2, 3)
    hopping = -0.5 * 10 * (np.cos(np.pi * flux[1]) + 0.2j * np.sin(np.pi * flux[1]))
    hc = np.diag(4 * 0.3 * numbers**2).astype(complex)
    hc += hopping * np.eye(5, k=1) + hopping.conjugate() * np.eye(5, k=-1)
    h = (
        np.kron(np.kron(circuits[0].hamiltonian(), np.eye(5)), np.eye(6))
        + np.kron(np.kron(np.eye(6), hc), np.eye(6))
        + np.kron(np.kron(np.eye(6), np.eye(5)), circuits[1].hamiltonian())
    )
    n1, n2 = (circuit.n_operator() for circuit in circuits)
    nc = np.diag(numbers)
    h += (
        0.05 * np.kron(np.kron(n1, nc), np.eye(6))
        + 0.06 * np.kron(np.kron(np.eye(6), nc), n2)
        - 0.04 * np.kron(np.kron(n1, np.eye(5)), n2)
    )
    np.testing.assert_allclose(
        vectors.conj().T @ h @ vectors, np.diag(model.energies_ghz), rtol=0, atol=2e-12
    )
    charge_square = np.kron(np.kron(np.eye(6), nc @ nc), np.eye(6))
    np.testing.assert_allclose(
        vectors.conj().T @ charge_square @ vectors,
        model.charge_squared_operator(1),
        rtol=0,
        atol=1e-12,
    )
    larger = JointModel(
        JointParameters(f, c, f, 0.05, 0.06, -0.04, 12), flux, temperature_k=0.04
    )
    overlap = vectors.conj().T @ larger.native_embedding().reshape(180, 12)
    np.testing.assert_allclose(np.abs(overlap), np.eye(8, 12), rtol=0, atol=1e-12)


def test_fixed_squid_static_spectrum_and_native_charge_square() -> None:
    from zcu_tools.simulate.ftf import (
        FluxoniumCircuit,
        JointModel,
        JointParameters,
        SquidCircuit,
    )

    f = FluxoniumCircuit(ej_ghz=3.0, ec_ghz=0.8, el_ghz=0.6, native_cutoff=10, levels=3)
    c = SquidCircuit(
        ejmax_ghz=10.0, ec_ghz=0.3, asymmetry=0.2, charge_cutoff=4, levels=4
    )
    p = JointParameters(
        f, c, f, g1c_ghz=0.0, g2c_ghz=0.0, g12_ghz=0.0, retained_levels=36
    )
    dc = np.array([0.3, 0.27, 0.4])
    model = JointModel(p, dc, temperature_k=0.04)
    n = np.arange(-4, 5)
    ej = 10 * np.sqrt(np.cos(np.pi * 0.27) ** 2 + 0.2**2 * np.sin(np.pi * 0.27) ** 2)
    reference = np.diag(4 * 0.3 * n**2) - ej / 2 * (np.eye(9, k=1) + np.eye(9, k=-1))
    np.testing.assert_allclose(
        model.local_energies_ghz[1],
        np.linalg.eigvalsh(reference)[:4],
        atol=1e-12,
        rtol=0,
    )
    charge = model.charge_operator(1)
    square = model.charge_squared_operator(1)
    assert np.linalg.norm(square - charge @ charge) > 0.1
    np.testing.assert_allclose(
        model.hamiltonian(dc), np.diag(model.energies_ghz), atol=1e-12, rtol=0
    )
    initial = model.initial_state()
    u = expm(-2j * np.pi * model.hamiltonian(dc) * 7.2)
    np.testing.assert_allclose(
        u @ initial.density @ u.conj().T, initial.density, atol=1e-12, rtol=0
    )


def test_signed_direct_coupling_and_parent_readout_states() -> None:
    from zcu_tools.simulate.ftf import (
        FluxoniumCircuit,
        JointModel,
        JointParameters,
        Resonator,
        SquidCircuit,
    )

    f1 = FluxoniumCircuit(3.0, 0.8, 0.6, 8, 3, q_cap=1e5)
    f2 = FluxoniumCircuit(3.3, 0.7, 0.55, 8, 3)
    c = SquidCircuit(10.0, 0.3, 0.2, 3, 3)
    flux = np.array([0.35, 0.22, 0.4])
    plus = JointModel(
        JointParameters(f1, c, f2, 0.05, 0.06, 0.04, 8), flux, temperature_k=0.06
    )
    minus = JointModel(
        JointParameters(f1, c, f2, 0.05, 0.06, -0.04, 8), flux, temperature_k=0.06
    )
    assert np.linalg.norm(plus.energies_ghz - minus.energies_ghz) > 1e-6
    changed = flux + np.array([0.0, 0.06, 0.0])
    assert np.linalg.norm(plus.hamiltonian(changed) - plus.hamiltonian(flux)) > 1e-3
    resonators = (
        Resonator(6.0, (0.03, 0.01, 0.0), 0.001, 0.0001, 0.0),
        Resonator(6.2, (0.0, 0.01, 0.03), 0.001, 0.0002, 0.0),
    )
    frequencies = plus.readout_frequencies_ghz(flux, resonators)
    assert frequencies.shape == (8, 2)
    assert np.ptp(frequencies[:, 0]) > 1e-6
    assert np.ptp(frequencies[4:, 1]) > 1e-6
    rates = plus.rates(flux)
    assert np.count_nonzero(rates) > 4
    q = rates - np.diag(rates.sum(axis=0))
    np.testing.assert_allclose(
        q @ np.diag(plus.initial_state().density).real, 0, atol=1e-16, rtol=0
    )
