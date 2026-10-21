"""Small multilevel diagnostics preserve full-model virtual sums before projection."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate import evolution
from zcu_tools.simulate.evolution import (
    JointFockDiagnostic,
    JointFockRequest,
    SecularDrive,
    SecularModel,
    joint_fock_spectrum,
)


def diagnostic_request(
    *, virtual_coupling=0.03, circular=False, resonant_coupling=0.002
) -> JointFockRequest:
    assert hasattr(evolution, "JointFockDiagnostic"), (
        "Named multilevel diagnostic is missing"
    )
    from zcu_tools.simulate.evolution import JointFockDiagnostic

    x = np.zeros((5, 5), dtype=np.complex128)
    x[0, 2] = x[2, 0] = resonant_coupling
    x[2, 4] = x[4, 2] = virtual_coupling
    y = np.zeros_like(x)
    if circular:
        y[0, 2], y[2, 0] = 1j * resonant_coupling, -1j * resonant_coupling
    model = SecularModel(
        level=(0,),
        identity="synthetic-joint-fock",
        energies_ghz=np.array([0, 0.11, 0.25, 0.49, 0.95]),
        eigenvectors=np.eye(5, dtype=np.complex128),
        drives=(SecularDrive("probe", x, y),),
        rates_per_ns=np.zeros((5, 5)),
        dephasing_per_ns=np.zeros((5, 5)),
    )
    diagnostic = JointFockDiagnostic(
        model=model, drive_port=0, pi_gain=1.0, probe_gain=1.0
    )
    return JointFockRequest(
        frequencies_ghz=np.tile([0.08, 0.12], (5, 1)),
        kappa_external_per_ns=np.array([0.01, 0.02]),
        kappa_internal_per_ns=np.array([0.002, 0.003]),
        resonator_carriers_ghz=np.array([0.09]),
        qubit_carriers_ghz=np.array([0.25]),
        qubit_frequency_ghz=0.25,
        incident_per_sqrt_ns=0,
        down_per_ns=0,
        up_per_ns=0,
        gamma2_per_ns=0,
        pi_duration_ns=0,
        pi_rabi_rad_per_ns=0,
        prefill_duration_ns=0,
        probe_duration_ns=125,
        probe_rabi_rad_per_ns=0,
        diagnostic=diagnostic,
    )


def test_four_level_probe_retains_virtual_shift_from_omitted_fifth_level():
    request = diagnostic_request()
    result = joint_fock_spectrum(request)
    # Independent constant two-level Rabi formula. The fifth level contributes
    # both +/- carrier denominators to state 2; the retained pair contributes
    # only its omitted counter-rotating edge. All values here are GHz.
    shift_0 = -(0.001**2) / 0.5
    shift_2 = 0.001**2 / 0.5 - 0.015**2 * (1 / 0.45 + 1 / 0.95)
    detuning = 2 * np.pi * (shift_2 - shift_0)
    omega = 4 * np.pi * 0.001
    expected = (
        omega**2
        / (omega**2 + detuning**2)
        * np.sin(np.sqrt(omega**2 + detuning**2) * 125 / 2) ** 2
    )
    np.testing.assert_allclose(result.populations[0, :, 0, 2], expected, atol=2e-8)
    assert result.state_indices == (0, 1, 2, 3)
    np.testing.assert_allclose(result.populations[..., [1, 3]], 0, atol=1e-14)
    without_virtual = joint_fock_spectrum(diagnostic_request(virtual_coupling=0))
    assert (
        abs(result.populations[0, 0, 0, 2] - without_virtual.populations[0, 0, 0, 2])
        > 1e-3
    )
    assert np.max(result.trace_error) < 1e-11


def test_four_level_preparation_is_numeric_finite_pi_before_prefill():
    request = diagnostic_request(virtual_coupling=0, circular=True)
    # Circular quadratures remove the counter-rotating term. Omega=4*pi*0.002.
    result = joint_fock_spectrum(
        replace(
            request, pi_duration_ns=125, prefill_duration_ns=25, probe_duration_ns=0
        )
    )
    np.testing.assert_allclose(result.populations[0, :, 0, 2], [0, 1], atol=2e-8)
    assert np.max(result.initial_trace_error) < 1e-12
    assert np.max(result.initial_edge_population) == pytest.approx(0, abs=1e-14)


def test_diagnostic_cutoff_represents_displaced_physical_vacuum():
    request = diagnostic_request(virtual_coupling=0)
    request = replace(
        request,
        resonator_carriers_ghz=np.array([0.08]),
        kappa_external_per_ns=np.array([0.1, 0.1]),
        kappa_internal_per_ns=np.array([0.02, 0.02]),
        incident_per_sqrt_ns=0.2,
        prefill_duration_ns=100,
        probe_duration_ns=0,
    )
    result = joint_fock_spectrum(request)
    main = joint_fock_spectrum(
        replace(
            request, diagnostic=None, frequencies_ghz=request.frequencies_ghz[[0, 2]]
        )
    )
    assert result.cutoffs[0] > main.cutoffs[0]
    assert result.sampled_pointer_means[0, 0] > main.sampled_pointer_means[0, 0]
    assert np.max(result.vacuum_trace_error) < 1e-12
    assert np.max(result.vacuum_edge_population) < 1e-3
    assert np.max(result.trace_error) < 1e-11


def stiff_prefill_request() -> JointFockRequest:
    zeros = np.zeros((4, 4))
    model = SecularModel(
        level=(0,),
        identity="synthetic-stiff-prefill",
        energies_ghz=np.array([0, 0.02, 0.04, 0.06]),
        eigenvectors=np.eye(4, dtype=np.complex128),
        drives=(SecularDrive("q", zeros.astype(complex), zeros.astype(complex)),),
        rates_per_ns=zeros,
        dephasing_per_ns=zeros,
    )
    return JointFockRequest(
        frequencies_ghz=np.tile([7.19005, 7.07959], (4, 1)),
        kappa_external_per_ns=np.array([0.006283, 0.006283]),
        kappa_internal_per_ns=np.array([0.0006283, 0.0006283]),
        resonator_carriers_ghz=np.array([7.1894875]),
        qubit_carriers_ghz=np.array([0.04]),
        qubit_frequency_ghz=0.04,
        incident_per_sqrt_ns=0.067,
        down_per_ns=0,
        up_per_ns=0,
        gamma2_per_ns=0,
        pi_duration_ns=0,
        pi_rabi_rad_per_ns=0,
        prefill_duration_ns=1500,
        probe_duration_ns=0,
        probe_rabi_rad_per_ns=0,
        diagnostic=JointFockDiagnostic(model, 0, 0, 0, (0, 2)),
    )


def test_smaller_step_completes_vacuum_far_spectator_prefill():
    request = stiff_prefill_request()
    assert request.diagnostic is not None
    result = joint_fock_spectrum(
        replace(request, diagnostic=replace(request.diagnostic, step_ns=0.25))
    )
    assert result.cutoffs == (5,)
    # No drive or bath changes the qubit. Its exact population stays in ground.
    np.testing.assert_allclose(result.populations[..., 0], 1, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.populations[..., 1], 0, rtol=0, atol=1e-14)
    assert np.max(result.trace_error) < 1e-10
    assert np.max(result.initial_trace_error) < 1e-10
    assert np.max(result.vacuum_trace_error) < 1e-12
    assert np.max(result.vacuum_edge_population) < 1e-3


def test_default_step_rejects_nonfinite_prefill_without_retry():
    with pytest.raises(FloatingPointError, match="Nonfinite joint-Fock RK4 density"):
        joint_fock_spectrum(stiff_prefill_request())


@pytest.mark.parametrize("segment", ["pi", "probe"])
def test_step_controls_driven_segments_against_exact_rabi_and_isolates_requests(
    segment,
):
    # Circular resonant drive gives exactly Omega=1 rad/ns, with no virtual shift.
    request = diagnostic_request(
        virtual_coupling=0, circular=True, resonant_coupling=1 / (4 * np.pi)
    )
    assert request.diagnostic is not None
    fine = replace(request.diagnostic, step_ns=0.25)
    duration = 4.13  # A noninteger number of requested steps in either case.
    request = replace(
        request,
        diagnostic=fine,
        pi_duration_ns=duration if segment == "pi" else 0,
        probe_duration_ns=duration if segment == "probe" else 0,
    )
    expected = np.sin(duration / 2) ** 2
    expected_branches = [0, expected] if segment == "pi" else [expected, expected]
    result = joint_fock_spectrum(request)
    # Rotation RK4 remainder bounds population error by
    # T * Omega**5 * dt**4 * exp(Omega*dt) / 240 < 8.7e-5 for dt<=0.25.
    np.testing.assert_allclose(
        result.populations[0, :, 0, 2], expected_branches, rtol=0, atol=1e-4
    )
    coarse = joint_fock_spectrum(replace(request, diagnostic=replace(fine, step_ns=1)))
    assert abs(coarse.populations[0, 1, 0, 2] - expected) > 1e-3
    repeated = joint_fock_spectrum(request)
    np.testing.assert_array_equal(repeated.populations, result.populations)
    assert request.diagnostic is fine
    assert fine.step_ns == 0.25


def test_diagnostic_step_defaults_to_one_ns_and_validates_finite_positive_bound():
    diagnostic = diagnostic_request().diagnostic
    assert diagnostic is not None
    assert hasattr(diagnostic, "step_ns"), "Diagnostic fixed step is not exposed"
    assert diagnostic.step_ns == 1.0
    for step in (1.0, 0.25, 0.125):
        assert replace(diagnostic, step_ns=step).step_ns == step
    for step in (0.0, -0.25, 1.00001, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="step_ns"):
            replace(diagnostic, step_ns=step)


def test_diagnostic_rejects_more_than_four_dynamic_levels():
    request = diagnostic_request()
    assert request.diagnostic is not None
    with pytest.raises(ValueError, match="retained_indices"):
        joint_fock_spectrum(
            replace(
                request,
                diagnostic=replace(
                    request.diagnostic, retained_indices=(0, 1, 2, 3, 4)
                ),
            )
        )
