"""Exact-input, immutable eigensystem and bath results in a single joint model."""

import numpy as np
import pytest
from zcu_tools.simulate.ftf import (
    FluxoniumCircuit,
    JointModel,
    JointParameters,
    SquidCircuit,
)


@pytest.fixture
def joint():
    f1 = FluxoniumCircuit(3.1, 0.8, 0.7, 8, 2, q_cap=2e4)
    f2 = FluxoniumCircuit(3.7, 0.9, 0.6, 8, 2, q_ind=3e4)
    coupler = SquidCircuit(9, 0.4, 0.17, 3, 2, q_cap=3e4)
    return JointModel(
        JointParameters(f1, coupler, f2, 0.08, -0.06, 0.02, 5),
        np.array([0.34, 0.21, 0.43]),
        temperature_k=0.06,
    )


def test_exact_eigensystem_inputs_reuse_immutable_results_without_flux_aliasing(joint):
    flux = joint.dc_flux.copy()
    energy, vectors = joint.eigensystem(flux)
    other_flux = np.nextafter(flux, np.inf)
    other_energy, other_vectors = joint.eigensystem(other_flux)
    repeated_energy, repeated_vectors = joint.eigensystem(flux.copy())
    assert repeated_energy is energy
    assert repeated_vectors is vectors
    assert other_energy is not energy
    assert other_vectors is not vectors
    assert not energy.flags.writeable
    assert not vectors.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        vectors[0, 0] = 0
    reference_energy, reference_vectors = np.linalg.eigh(joint.hamiltonian(flux))
    np.testing.assert_array_equal(energy, reference_energy)
    np.testing.assert_array_equal(vectors, reference_vectors)


def test_exact_bath_inputs_reuse_immutable_rates_without_flux_or_joint_aliasing(joint):
    rates = joint.rates(joint.dc_flux)
    other_flux = np.nextafter(joint.dc_flux, np.inf)
    other_rates = joint.rates(other_flux)
    assert joint.rates(joint.dc_flux.copy()) is rates
    assert other_rates is not rates
    assert not rates.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        rates[0, 1] = 123
    other_joint = JointModel(joint.parameters, other_flux, temperature_k=0.06)
    assert other_joint.rates(other_flux) is not rates
