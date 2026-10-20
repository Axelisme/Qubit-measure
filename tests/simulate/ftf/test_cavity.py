"""Driven Stark shifts and branch-distinguishability dephasing at the FTF seam."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.simulate import ftf
from zcu_tools.simulate.adc import ReadoutPulse

pytestmark = pytest.mark.sim


@pytest.fixture
def cavity():
    f = ftf.FluxoniumCircuit(3.1, 0.8, 0.7, 6, 2)
    c = ftf.SquidCircuit(9, 0.4, 0.17, 2, 2)
    joint = ftf.JointModel(
        ftf.JointParameters(f, c, f, 0.02, -0.03, 0.01, 3),
        np.array([0.34, 0.21, 0.43]), temperature_k=0,
    )
    resonators = (
        ftf.Resonator(6.2, (0.05, 0.01, 0), 0.08, 0.02, 0),
        ftf.Resonator(7.3, (0, 0.01, 0.05), 0.18, 0.04, 0),
    )
    pulse = ReadoutPulse(0, 25, 0.3j, 6.2, 0.4)
    drive_type = getattr(ftf, "CavityDrive", None)
    return joint, resonators, pulse, drive_type


def test_cavity_response_uses_joint_pulls_and_coherent_shared_line_fields(cavity):
    joint, resonators, pulse, drive_type = cavity
    assert callable(drive_type), "FTF must expose cavity photons, pulls and driven dephasing"
    drive = drive_type(joint, resonators, (pulse,))
    times = np.array([0., 4., 25., 40.])
    response = drive.response(times)
    frequencies = joint.readout_frequencies_ghz(joint.dc_flux, resonators)
    chi = frequencies - [6.2, 7.3]
    np.testing.assert_allclose(response.chi_ghz, chi, atol=1e-14)
    # Independent constant-input matrix solution in the incident carrier frame.
    v = np.array([0.2, 0.3])
    fields = np.zeros((len(times), len(frequencies), 2), dtype=complex)
    for state, frequency in enumerate(frequencies):
        generator = np.diag(np.array([0.01, 0.02]) + 2j * np.pi * (frequency - 6.2)) + np.outer(v, v)
        steady = np.linalg.solve(generator, v * pulse.amplitude * np.exp(0.4j))
        for index, time in enumerate(times):
            on = (np.eye(2) - expm(-generator * min(time, 25))) @ steady
            fields[index, state] = expm(-generator * max(time - 25, 0)) @ on
    photons = np.abs(fields) ** 2
    np.testing.assert_allclose(response.photons, photons, rtol=2e-7, atol=1e-11)
    np.testing.assert_allclose(response.stark_shift_ghz, np.sum(chi * photons, axis=-1), atol=1e-12)
    delta = fields[:, :, None, :] - fields[:, None, :, :]
    # Worked shared-line coefficients: .5*|.2*d0+.3*d1|²+.01|d0|²+.02|d1|².
    expected = .03 * np.abs(delta[..., 0]) ** 2 + .065 * np.abs(delta[..., 1]) ** 2
    expected += .06 * np.real(delta[..., 0] * delta[..., 1].conj())
    np.testing.assert_allclose(response.dephasing_per_ns, expected, rtol=2e-7, atol=1e-12)
    np.testing.assert_array_equal(np.diagonal(response.dephasing_per_ns, axis1=-2, axis2=-1), 0)
    stronger = drive_type(joint, resonators, (replace(pulse, amplitude=2*pulse.amplitude),)).response(times)
    np.testing.assert_allclose(stronger.photons, 4*response.photons, rtol=2e-7, atol=1e-11)
    np.testing.assert_allclose(stronger.stark_shift_ghz, 4*response.stark_shift_ghz, rtol=2e-7, atol=1e-12)
    np.testing.assert_allclose(stronger.dephasing_per_ns, 4*response.dephasing_per_ns, rtol=2e-7, atol=1e-12)


def test_cavity_segment_average_retains_ringdown_and_integrates_rates(cavity):
    joint, resonators, pulse, drive_type = cavity
    assert callable(drive_type), "FTF must expose cavity photons, pulls and driven dephasing"
    drive = drive_type(joint, resonators, (pulse,))
    # A dense independent quadrature checks a segment spanning switch-off.
    times = np.linspace(10., 40., 3001)
    trajectory = drive.response(times)
    average = drive.average(10., 40.)
    for actual, values in (
        (average.photons, trajectory.photons),
        (average.stark_shift_ghz, trajectory.stark_shift_ghz),
        (average.dephasing_per_ns, trajectory.dephasing_per_ns),
    ):
        np.testing.assert_allclose(actual, np.trapz(values, times, axis=0)/30, rtol=2e-5, atol=1e-12)
    assert np.max(drive.average(25., 40.).photons) > 0
    vacuum = drive_type(joint, resonators, ()).average(0., 40.)
    np.testing.assert_array_equal(vacuum.photons, 0)
    np.testing.assert_array_equal(vacuum.stark_shift_ghz, 0)
    np.testing.assert_array_equal(vacuum.dephasing_per_ns, 0)
    with pytest.raises(ValueError, match="interval"):
        drive.average(5., 5.)
