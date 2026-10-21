"""Joint-Fock spectra through the explicit numeric request, without acquisition."""

import numpy as np
import pytest

from zcu_tools.simulate import evolution


def test_vacuum_spectrum_keeps_finite_pi_and_probe_preparations_independent():
    assert hasattr(evolution, "JointFockRequest"), "Explicit joint request is missing"
    assert hasattr(evolution, "joint_fock_spectrum"), "Joint numeric entry is missing"
    from zcu_tools.simulate.evolution import JointFockRequest, joint_fock_spectrum

    request = JointFockRequest(
        frequencies_ghz=np.array([[0.08, 0.12], [0.081, 0.121]]),
        kappa_external_per_ns=np.array([0.01, 0.02]),
        kappa_internal_per_ns=np.array([0.002, 0.003]),
        resonator_carriers_ghz=np.array([0.09, 0.10]),
        qubit_carriers_ghz=np.array([0.02, 0.02]),
        qubit_frequency_ghz=0.02,
        incident_per_sqrt_ns=0.0,
        down_per_ns=0.0,
        up_per_ns=0.0,
        gamma2_per_ns=0.0,
        pi_duration_ns=10.0,
        pi_rabi_rad_per_ns=np.pi / 10,
        prefill_duration_ns=0.0,
        probe_duration_ns=10.0,
        probe_rabi_rad_per_ns=np.pi / 20,
    )
    result = joint_fock_spectrum(request)
    # A resonant pi/2 probe maps both the ground and finite-pi branches to Pe=1/2.
    assert result.populations.shape == (2, 2, 2, 2)  # rf, preparation, qf, g/e
    np.testing.assert_allclose(result.populations[..., 1], 0.5, atol=3e-6)
    np.testing.assert_allclose(result.populations.sum(axis=-1), 1, atol=1e-12)
    np.testing.assert_allclose(result.trace_error, 0, atol=1e-12)
    assert np.max(result.hermitian_error) < 1e-12
    assert np.min(result.min_eigenvalue) > -1e-6
    # Repeated calls and repeated probe coordinates never carry a previous state.
    np.testing.assert_array_equal(joint_fock_spectrum(request).populations, result.populations)
    assert result.cutoffs == (1, 1)  # zero sampled mean -> N=0, conservative N+1
    assert np.max(result.edge_population) == pytest.approx(0.0, abs=1e-14)
