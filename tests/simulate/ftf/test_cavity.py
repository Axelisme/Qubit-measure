"""Driven Stark shifts and branch-distinguishability dephasing at the FTF seam."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import trapezoid
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
        np.array([0.34, 0.21, 0.43]),
        temperature_k=0,
    )
    resonators = (
        ftf.Resonator(6.2, (0.05, 0.01, 0), 0.08, 0.02, 0),
        ftf.Resonator(7.3, (0, 0.01, 0.05), 0.18, 0.04, 0),
    )
    pulse = ReadoutPulse(0, 25, 0.3j, 6.2, 0.4)
    drive_type = getattr(ftf, "CavityDrive", None)
    return joint, resonators, pulse, drive_type


@pytest.mark.parametrize("branch", [0, 1, None], ids=["ground", "prepared", "tie-lowest-energy"])
def test_cavity_advance_inherits_main_branch_and_keeps_ringdown(cavity, branch):
    from zcu_tools.simulate.adc import cavity_field_average, cavity_fields

    joint, resonators, pulse, _ = cavity
    drive = ftf.CavityDrive(joint, resonators, (pulse,))
    assert callable(getattr(drive, "advance", None)), "FTF must carry the main branch field between spans"
    model = ftf.readout_model(joint, resonators)
    count = len(model.state_labels)
    populations = np.zeros(count)
    if branch is None:
        populations[:2] = .5
        selected = 0
    else:
        populations[branch] = 1.
        selected = branch
    inherited = np.array([.2 + .3j, -.1j])
    average, final = drive.advance(10., 25., populations, inherited)
    fields = cavity_fields(model, (pulse,), np.array([10., 25.]), initial_field=inherited)
    expected = cavity_field_average(model, (pulse,), 10., 25., initial_field=inherited)
    np.testing.assert_allclose(average.photons, expected.photons, atol=1e-12)
    np.testing.assert_allclose(average.stark_shift_ghz, (average.chi_ghz * expected.photons).sum(axis=-1), atol=1e-12)
    np.testing.assert_allclose(final, fields[-1, selected], atol=1e-12)
    assert np.max(np.abs(fields[-1, 0] - fields[-1, 1])) > 1e-7
    ringdown, next_final = drive.advance(25., 40., populations, final)
    expected_ringdown = cavity_fields(model, (), np.array([25., 40.]), initial_field=final)
    np.testing.assert_allclose(next_final, expected_ringdown[-1, selected], atol=1e-11)
    assert np.max(ringdown.photons) > 0
    # Restarting each branch from vacuum would erase this entire span.
    np.testing.assert_array_equal(ftf.CavityDrive(joint, resonators, ()).average(25., 40.).photons, 0)


def test_cavity_response_uses_joint_pulls_and_coherent_shared_line_fields(cavity):
    joint, resonators, pulse, drive_type = cavity
    assert callable(drive_type), (
        "FTF must expose cavity photons, pulls and driven dephasing"
    )
    drive = ftf.CavityDrive(joint, resonators, (pulse,))
    times = np.array([0.0, 4.0, 25.0, 40.0])
    response = drive.response(times)
    frequencies = joint.readout_frequencies_ghz(joint.dc_flux, resonators)
    chi = frequencies - [6.2, 7.3]
    np.testing.assert_allclose(response.chi_ghz, chi, atol=1e-14)
    # Independent constant-input matrix solution in the incident carrier frame.
    v = np.array([0.2, 0.3])
    fields = np.zeros((len(times), len(frequencies), 2), dtype=complex)
    for state, frequency in enumerate(frequencies):
        generator = np.diag(
            np.array([0.01, 0.02]) + 2j * np.pi * (frequency - 6.2)
        ) + np.outer(v, v)
        steady = np.linalg.solve(generator, v * pulse.amplitude * np.exp(0.4j))
        for index, time in enumerate(times):
            on = (np.eye(2) - expm(-generator * min(time, 25))) @ steady
            fields[index, state] = expm(-generator * max(time - 25, 0)) @ on
    photons = np.abs(fields) ** 2
    np.testing.assert_allclose(response.photons, photons, rtol=2e-7, atol=1e-11)
    np.testing.assert_allclose(
        response.stark_shift_ghz, np.sum(chi * photons, axis=-1), atol=1e-12
    )
    delta = fields[:, :, None, :] - fields[:, None, :, :]
    # Worked shared-line coefficients: .5*|.2*d0+.3*d1|²+.01|d0|²+.02|d1|².
    expected = 0.03 * np.abs(delta[..., 0]) ** 2 + 0.065 * np.abs(delta[..., 1]) ** 2
    expected += 0.06 * np.real(delta[..., 0] * delta[..., 1].conj())
    np.testing.assert_allclose(
        response.dephasing_per_ns, expected, rtol=2e-7, atol=1e-12
    )
    np.testing.assert_array_equal(
        np.diagonal(response.dephasing_per_ns, axis1=-2, axis2=-1), 0
    )
    stronger = ftf.CavityDrive(
        joint, resonators, (replace(pulse, amplitude=2 * pulse.amplitude),)
    ).response(times)
    np.testing.assert_allclose(
        stronger.photons, 4 * response.photons, rtol=2e-7, atol=1e-11
    )
    np.testing.assert_allclose(
        stronger.stark_shift_ghz, 4 * response.stark_shift_ghz, rtol=2e-7, atol=1e-12
    )
    np.testing.assert_allclose(
        stronger.dephasing_per_ns, 4 * response.dephasing_per_ns, rtol=2e-7, atol=1e-12
    )


def test_cavity_flat_four_us_average_is_finite_and_matches_weighted_subspans(cavity):
    joint, _, _, _ = cavity
    # Two long-lived modes, 110 MHz apart, retain fast beats during ring-up.
    resonators = (
        ftf.Resonator(7.19, (0.05, 0.01, 0), 0.006, 0.0009, 0),
        ftf.Resonator(7.08, (0, 0.01, 0.05), 0.005, 0.0008, 0),
    )
    pulse = ReadoutPulse(0, 5500, 0.15, 7.19)
    drive = ftf.CavityDrive(joint, resonators, (pulse,))
    try:
        average = drive.average(1500, 5500)
    except RuntimeError as error:
        pytest.fail(f"legal 4 us flat cavity average must converge: {error}")
    # Unequal spans expose averaging that forgets the duration weights.
    boundaries = (1500, 2000, 2750, 3750, 4500, 5500)
    spans = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))
    pieces = [drive.average(left, right) for left, right in spans]
    for name in ("photons", "stark_shift_ghz", "dephasing_per_ns"):
        actual = getattr(average, name)
        assert np.isfinite(actual).all(), f"4 us average {name} must be finite"
        expected = (
            sum(
                (right - left) * getattr(piece, name)
                for (left, right), piece in zip(spans, pieces, strict=True)
            )
            / 4000
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-13)


def test_cavity_segment_average_retains_ringdown_and_integrates_rates(cavity):
    joint, resonators, pulse, drive_type = cavity
    assert callable(drive_type), (
        "FTF must expose cavity photons, pulls and driven dephasing"
    )
    drive = ftf.CavityDrive(joint, resonators, (pulse,))
    # A dense independent quadrature checks a segment spanning switch-off.
    times = np.linspace(10.0, 40.0, 3001)
    trajectory = drive.response(times)
    average = drive.average(10.0, 40.0)
    for actual, values in (
        (average.photons, trajectory.photons),
        (average.stark_shift_ghz, trajectory.stark_shift_ghz),
        (average.dephasing_per_ns, trajectory.dephasing_per_ns),
    ):
        np.testing.assert_allclose(
            actual, trapezoid(values, times, axis=0) / 30, rtol=2e-5, atol=1e-12
        )
    assert np.max(drive.average(25.0, 40.0).photons) > 0
    vacuum = ftf.CavityDrive(joint, resonators, ()).average(0.0, 40.0)
    np.testing.assert_array_equal(vacuum.photons, 0)
    np.testing.assert_array_equal(vacuum.stark_shift_ghz, 0)
    np.testing.assert_array_equal(vacuum.dephasing_per_ns, 0)
    with pytest.raises(ValueError, match="interval"):
        drive.average(5.0, 5.0)
