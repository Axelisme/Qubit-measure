"""Small joint-cavity spectra against a separate physical-frame master equation."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from zcu_tools.simulate.evolution import JointFockRequest, joint_fock_spectrum


@pytest.fixture
def cavity_request() -> JointFockRequest:
    return JointFockRequest(
        frequencies_ghz=np.array([[0.08, 0.12], [0.083, 0.118]]),
        kappa_external_per_ns=np.array([0.01, 0.02]),
        kappa_internal_per_ns=np.array([0.002, 0.003]),
        resonator_carriers_ghz=np.array([0.08]),
        qubit_carriers_ghz=np.array([0.02]),
        qubit_frequency_ghz=0.02,
        incident_per_sqrt_ns=0.02,
        down_per_ns=0.0,
        up_per_ns=0.0,
        gamma2_per_ns=0.0,
        pi_duration_ns=0.0,
        pi_rabi_rad_per_ns=np.pi / 100,
        prefill_duration_ns=0.0,
        probe_duration_ns=100.0,
        probe_rabi_rad_per_ns=np.pi / 200,
    )


def test_driven_cavity_matches_independent_physical_frame_lindblad(cavity_request):
    request = cavity_request
    # Independent undisplaced 4x4 cavity truncation, dense density-matrix ODE.
    # It shares neither the total-N basis, CSR generator nor RK4 implementation.
    a = np.diag(np.sqrt(np.arange(1, 4)), 1)
    ic = np.eye(16)
    a1, a2 = np.kron(a, np.eye(4)), np.kron(np.eye(4), a)
    operators = [np.kron(np.eye(2), a1), np.kron(np.eye(2), a2)]
    v = np.sqrt(request.kappa_external_per_ns / 2)
    port = v[0] * operators[0] + v[1] * operators[1]
    h = 1j * request.incident_per_sqrt_ns * (port.conj().T - port)
    for mode, op in enumerate((a1, a2)):
        h += np.kron(2 * np.pi * np.diag(request.frequencies_ghz[:, mode] - 0.08), op.conj().T @ op)
    h += np.kron(np.array([[0, 1], [1, 0]]) * request.probe_rabi_rad_per_ns / 2, ic)
    jumps = [np.sqrt(2) * port] + [np.sqrt(k) * op for k, op in zip(request.kappa_internal_per_ns, operators, strict=True)]
    losses = [jump.conj().T @ jump for jump in jumps]

    def rhs(_time, state):
        density = state.reshape(32, 32)
        derivative = -1j * (h @ density - density @ h)
        for jump, loss in zip(jumps, losses, strict=True):
            derivative += jump @ density @ jump.conj().T - (loss @ density + density @ loss) / 2
        return derivative.ravel()

    initial = np.zeros((32, 32), complex)
    initial[0, 0] = 1
    reference = solve_ivp(rhs, (0, 100), initial.ravel(), method="DOP853", rtol=1e-10, atol=1e-12)
    assert reference.success
    pe = reference.y[:, -1].reshape(32, 32).diagonal()[16:].real.sum()
    try:
        result = joint_fock_spectrum(request)
    except ValueError as error:
        pytest.fail(f"Driven cavity spectrum is missing: {error}")
    np.testing.assert_allclose(result.populations[0, :, 0, 1], pe, atol=2e-5, rtol=0)
    vacuum = joint_fock_spectrum(replace(request, incident_per_sqrt_ns=0))
    assert abs(pe - vacuum.populations[0, 0, 0, 1]) > 1e-4
    assert np.max(result.trace_error) < 1e-11
    assert np.max(result.hermitian_error) < 1e-11
    assert np.min(result.min_eigenvalue) > -1e-7


def test_analytic_prefill_and_probe_keep_intrinsic_relaxation(cavity_request):
    request = cavity_request
    gamma, omega, pi_time = 0.001, np.pi / 100, 100.0
    frequency = np.sqrt(omega**2 - gamma**2 / 16)
    pi_pe = omega**2 / (2 * omega**2 + gamma**2) * (
        1 - np.exp(-3 * gamma * pi_time / 4) * (np.cos(frequency * pi_time) + 3 * gamma / (4 * frequency) * np.sin(frequency * pi_time))
    )
    try:
        result = joint_fock_spectrum(replace(request, down_per_ns=gamma, gamma2_per_ns=gamma / 2, pi_duration_ns=pi_time, prefill_duration_ns=20, probe_duration_ns=50, probe_rabi_rad_per_ns=0))
    except ValueError as error:
        pytest.fail(f"Driven cavity prefill is missing: {error}")
    np.testing.assert_allclose(result.populations[0, 1, 0, 1], pi_pe * np.exp(-gamma * 70), atol=1e-10)
    np.testing.assert_allclose(result.populations[0, 0, 0, 1], 0, atol=1e-12)


def test_rf_and_probe_schedule_isolation_and_owned_request(cavity_request):
    request = cavity_request
    frequencies = request.frequencies_ghz.copy()
    batched = replace(request, frequencies_ghz=frequencies, resonator_carriers_ghz=np.array([0.08, 0.09, 0.08]), qubit_carriers_ghz=np.array([0.02, 0.021, 0.02]), prefill_duration_ns=30)
    frequencies[:] = 7  # Caller mutation must not alter the prepared request.
    try:
        result = joint_fock_spectrum(batched)
    except ValueError as error:
        pytest.fail(f"Shared cavity schedule is missing: {error}")
    np.testing.assert_array_equal(result.populations[0], result.populations[2])
    np.testing.assert_array_equal(result.populations[:, :, 0], result.populations[:, :, 2])
    for rf_index, rf in enumerate((0.08, 0.09)):
        scalar = joint_fock_spectrum(replace(batched, resonator_carriers_ghz=np.array([rf]), qubit_carriers_ghz=np.array([0.021])))
        np.testing.assert_allclose(result.populations[rf_index, :, 1], scalar.populations[0, :, 0], atol=1e-13)
    assert not np.allclose(result.populations[0], result.populations[1], rtol=0, atol=1e-6)
