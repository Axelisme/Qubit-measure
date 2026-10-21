"""Fail-fast public request guards for analytic one-jump spectra."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import JointFockRequest, joint_fock_spectrum


@pytest.fixture
def vacuum_request() -> JointFockRequest:
    return JointFockRequest(
        frequencies_ghz=np.array([[0.08, 0.12], [0.083, 0.118]]),
        kappa_external_per_ns=np.array([0.01, 0.02]),
        kappa_internal_per_ns=np.array([0.002, 0.003]),
        resonator_carriers_ghz=np.array([0.09]),
        qubit_carriers_ghz=np.array([0.02]),
        qubit_frequency_ghz=0.02,
        incident_per_sqrt_ns=0,
        down_per_ns=0,
        up_per_ns=0,
        gamma2_per_ns=0,
        pi_duration_ns=0,
        pi_rabi_rad_per_ns=0,
        prefill_duration_ns=0,
        probe_duration_ns=0,
        probe_rabi_rad_per_ns=0,
    )


def test_one_jump_prefill_accepts_closed_boundary_and_rejects_above(vacuum_request):
    accepted = replace(
        vacuum_request, prefill_duration_ns=100, up_per_ns=1e-14, gamma2_per_ns=0.5e-14
    )
    populations = joint_fock_spectrum(accepted).populations
    np.testing.assert_allclose(populations.sum(axis=-1), 1, atol=1e-12)
    with pytest.raises(ValueError, match="One-jump prefill"):
        replace(accepted, up_per_ns=2e-14, gamma2_per_ns=1e-14)


@pytest.mark.parametrize(
    "frequencies", [np.zeros(2), np.array([[0.08, np.nan], [0.083, 0.118]])]
)
def test_request_rejects_bad_frequency_shape_or_nonfinite(vacuum_request, frequencies):
    with pytest.raises(ValueError, match="frequencies_ghz"):
        replace(vacuum_request, frequencies_ghz=frequencies)


@pytest.mark.parametrize(
    ("down", "up", "gamma2", "message"),
    [
        (-0.01, 0, 0, "down_per_ns must be nonnegative"),
        (0.01, 0.02, 0.001, "negative pure dephasing"),
        (0, 0, np.nan, "gamma2_per_ns must be finite"),
    ],
)
def test_request_rejects_invalid_intrinsic_rates(
    vacuum_request, down, up, gamma2, message
):
    with pytest.raises(ValueError, match=message):
        replace(vacuum_request, down_per_ns=down, up_per_ns=up, gamma2_per_ns=gamma2)
