"""Coupled FTF physics through the public segment engine, not a legacy solver."""

from dataclasses import replace
from threading import Event

import numpy as np
from numpy.typing import NDArray
from zcu_tools.simulate import ftf
from zcu_tools.simulate.evolution import (
    DriveSource,
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    FluxWaveform,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels.t4 import T4
from zcu_tools.simulate.lines import FluxLine

from tests.simulate.evolution.contract.kernel_suite import KernelCase, reference

Matrix = NDArray[np.complex128]


def joint_model(*, baths: bool = True, retained: int = 4, scale: float = 1.0):
    """Small coupled-bath fixture, or its lossless scaled counterpart."""
    f1 = ftf.FluxoniumCircuit(
        3.0 * scale,
        0.8 * scale,
        0.6 * scale,
        12,
        2,
        q_cap=1e4 if baths else None,
        q_ind=2e5 if baths else None,
        xqp=1e-7 if baths else 0,
        aphi=0.001 if baths else 0,
    )
    f2 = ftf.FluxoniumCircuit(
        3.3 * scale,
        0.7 * scale,
        0.55 * scale,
        12,
        2,
        aphi=0.002 if baths else 0,
    )
    coupler = ftf.SquidCircuit(
        10.0 * scale,
        0.3 * scale,
        0.2,
        4,
        2,
        q_cap=2e4 if baths else None,
        aphi=0.001 if baths else 0,
    )
    return ftf.JointModel(
        ftf.JointParameters(
            f1, coupler, f2, 0.05 * scale, 0.06 * scale, -0.04 * scale, retained
        ),
        np.array([0.35, 0.22, 0.4]),
        temperature_k=0.08 if baths else 0.0,
    )


def resonators(*, baths: bool = True) -> tuple[ftf.Resonator, ftf.Resonator]:
    return (
        ftf.Resonator(
            6.0,
            (0.03, 0.01, 0.0) if baths else (0, 0, 0),
            0.001,
            0.0001,
            0.01 if baths else 0,
        ),
        ftf.Resonator(
            6.2,
            (0.0, 0.01, 0.03) if baths else (0, 0, 0),
            0.001,
            0.0002,
            0.02 if baths else 0,
        ),
    )


def batch(
    end: float,
    *,
    flux: FluxSchedule | None = None,
    waveform: FluxWaveform | None = None,
    tracks: tuple[DriveTrack, ...] = (),
) -> TimelineBatch:
    if flux is None:
        flux = FluxSchedule(
            np.array([0, 1]), np.array([0.0]), np.zeros((1, 3), dtype=np.int64)
        )
    return TimelineBatch(np.array([0]), tracks, flux, np.array([end]), waveform)


def evolve(
    source: ModelSource, timeline: TimelineBatch, *, path: str | None = None
) -> Matrix:
    engine = Engine(1, 1_000_000)
    try:
        result = engine.evolve(source, (timeline,), cancel=Event())
        if path is not None:
            assert result.paths.segments_by_kernel.get(path, 0) > 0
        return result.densities[0]
    finally:
        engine.close()


def strict_reference(source: ModelSource, timeline: TimelineBatch) -> Matrix:
    """Adapt one public timeline to the existing independent Lindblad oracle.

    These scenarios have at most one microwave pulse, spanning the whole row.
    T4 is only the required KernelCase placeholder; reference never calls it.
    """
    track = timeline.tracks[0] if timeline.tracks else None
    width = int(track is not None)
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=timeline.end_ns,
        level=np.zeros((1, 3), dtype=np.int64),
        tones=np.array([width]),
        tone_port=np.zeros((1, width), dtype=np.int64),
        tone_frequency_ghz=np.array([[track.frequency_ghz[0]]])
        if track
        else np.empty((1, 0)),
        tone_amplitude=np.array([[track.amplitude[0]]]) if track else np.empty((1, 0)),
        tone_theta_rad=np.array([[track.phase_rad[0]]]) if track else np.empty((1, 0)),
        tone_envelope=np.zeros((1, width), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, width)),
        envelopes=track.envelopes if track else (),
        flux_waveform=timeline.flux_waveform,
    )
    case = KernelCase(T4(), source, table, source.initial_density[None])
    if timeline.flux_waveform is not None:
        return reference(case, 0, source.initial_density, bath=timeline.flux)
    # A no-drive quantized timeline has piecewise H as well as piecewise baths.
    dc = source.model((0, 0, 0)).eigenvectors
    rho = source.initial_density.copy()
    for index, start in enumerate(timeline.flux.start_ns):
        end = (
            timeline.flux.start_ns[index + 1]
            if index + 1 < len(timeline.flux.start_ns)
            else timeline.end_ns[0]
        )
        if end <= start:
            continue
        level = timeline.flux.level[index : index + 1]
        vectors = source.model(tuple(int(v) for v in level[0])).eigenvectors
        overlap = dc.conj().T @ vectors
        local = overlap.conj().T @ rho @ overlap
        row = replace(
            table,
            start_ns=np.array([start]),
            duration_ns=np.array([end - start]),
            level=level,
        )
        local = reference(replace(case, table=row), 0, local)
        rho = overlap @ local @ overlap.conj().T
    return rho


def assert_density(rho: Matrix, *, atol: float = 1e-9) -> None:
    np.testing.assert_allclose(rho, rho.conj().T, atol=atol, rtol=0)
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=atol, rtol=0)
    assert np.linalg.eigvalsh(rho).min() > -atol


def test_coupled_gibbs_stationary_under_all_enabled_joint_baths() -> None:
    joint = joint_model()
    source = ftf.model_source(joint, (), resonators=resonators())
    weights = np.exp(
        -0.04799243073366221 * (joint.energies_ghz - joint.energies_ghz[0]) / 0.08
    )
    np.testing.assert_allclose(
        source.initial_density, np.diag(weights / weights.sum()), atol=1e-14, rtol=0
    )
    final = evolve(source, batch(12.0))
    np.testing.assert_allclose(final, source.initial_density, atol=1e-10, rtol=0)
    assert_density(final, atol=1e-10)


def test_static_engine_preserves_full_dynamic_baths_at_constant_flux() -> None:
    joint = joint_model()
    thermal = ftf.model_source(joint, (), resonators=resonators())
    # Coherences expose lost pure dephasing; Gibbs populations alone cannot.
    initial = np.full((4, 4), 0.25 + 0j, dtype=np.complex128)
    source = ModelSource(initial, thermal.model, thermal.flux_expansion())
    continuous = FluxWaveform(
        np.array([0, 2]), np.array([0.0, 2.0]), np.tile(joint.dc_flux, (2, 1))
    )
    actual = evolve(source, batch(2.0))
    expected = strict_reference(source, batch(2.0, waveform=continuous))
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    # Verify that the full-bath oracle is observably different from unitary H0.
    energy = source.model((0, 0, 0)).energies_ghz
    unitary = initial * np.exp(-2j * np.pi * 2 * (energy[:, None] - energy[None, :]))
    assert np.max(np.abs(expected - unitary)) > 0.01
    assert_density(actual)


def test_dynamic_joint_bath_segment_refinement_in_same_fixed_basis() -> None:
    joint = joint_model()
    source = ftf.model_source(
        joint,
        (DriveSource("mw", 0.001 * joint.charge_operator(0)),),
        resonators=resonators(),
    )
    # A weak real drive engages continuous H(flux). Both runs have identical
    # H and drive samples; only the instantaneous grid baths are refined.
    time = np.linspace(0.0, 3.0, 7)
    waveform = FluxWaveform(
        np.array([0, len(time)]),
        time,
        joint.dc_flux
        + np.array([0.002, 0.005, -0.001]) * np.sin(np.pi * time[:, None] / 3) ** 2,
    )
    drive = DriveTrack(
        "mw",
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.4]),
        np.array([0.3]),
        np.array([0.01]),
        (EnvelopeSpec("constant"),),
        np.array([0]),
        np.array([True]),
    )
    outputs = []
    for segment in (1.0, 0.5):
        time = np.arange(0.0, 3.0, segment)
        offsets = np.array([2, 5, -1]) * np.sin(np.pi * time[:, None] / 3) ** 2
        flux = FluxSchedule(
            np.array([0, len(time)]), time, np.rint(offsets).astype(np.int64)
        )
        timeline = batch(3.0, flux=flux, waveform=waveform, tracks=(drive,))
        actual = evolve(source, timeline, path="flux_dop853")
        expected = strict_reference(source, timeline)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
        assert_density(actual)
        outputs.append(actual)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-5, rtol=0)


def test_splitting_identical_quantized_plateaus_preserves_no_drive_dynamics() -> None:
    joint = joint_model()
    source = ftf.model_source(joint, (), resonators=resonators())
    coarse = FluxSchedule(
        np.array([0, 4]),
        np.array([0.0, 4.0, 8.0, 10.0]),
        np.array([[0, 0, 0], [2, 5, -1], [1, 2, 0], [0, 0, 0]]),
    )
    refined = FluxSchedule(
        np.array([0, 8]),
        np.array([0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0]),
        np.repeat(coarse.level, 2, axis=0),
    )
    actual = evolve(source, batch(12.0, flux=coarse))
    split = evolve(source, batch(12.0, flux=refined))
    np.testing.assert_allclose(actual, split, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        actual, strict_reference(source, batch(12.0, flux=coarse)), atol=1e-10, rtol=0
    )
    assert_density(actual)


def test_dynamic_flux_signed_complex_drive_against_larger_fixed_parent() -> None:
    joint = joint_model(baths=False, retained=4)
    # Keep every state of the 2x2x2 local parent in the strict reference.
    parent = ftf.JointModel(
        replace(joint.parameters, retained_levels=8), joint.dc_flux, temperature_k=0.0
    )
    coefficients = (0.03 + 0.01j, 0j, -0.004 + 0.003j)

    def source_for(model: ftf.JointModel) -> ModelSource:
        x = sum(
            (co.real * model.charge_operator(i) for i, co in enumerate(coefficients)),
            np.zeros((len(model.energies_ghz),) * 2, complex),
        )
        y = sum(
            (-co.imag * model.charge_operator(i) for i, co in enumerate(coefficients)),
            np.zeros_like(x),
        )
        return ftf.model_source(
            model, (DriveSource("mw", x, y),), resonators=resonators(baths=False)
        )

    time = np.linspace(0.0, 8.0, 81)
    waveform = FluxWaveform(
        np.array([0, len(time)]),
        time,
        joint.dc_flux
        + np.array([0.002, 0.005, -0.001]) * np.sin(np.pi * time[:, None] / 8) ** 2,
    )
    level = np.rint((waveform.flux - joint.dc_flux) / 0.001).astype(np.int64)
    flux = FluxSchedule(np.array([0, len(time)]), time, level)
    envelope = EnvelopeSpec(
        "sampled",
        sample_ns=time,
        samples=np.asarray(
            np.exp(-0.5 * ((time - 4) / 2) ** 2) * (1 + 0.3j * np.sin(time)),
            dtype=complex,
        ),
    )
    drive = DriveTrack(
        "mw",
        np.array([0.0]),
        np.array([8.0]),
        np.array([0.4]),
        np.array([0.3]),
        np.array([-0.2]),
        (envelope,),
        np.array([0]),
        np.array([True]),
    )
    timeline = batch(8.0, flux=flux, waveform=waveform, tracks=(drive,))
    source, parent_source = source_for(joint), source_for(parent)
    actual = evolve(source, timeline)
    parent_density = strict_reference(parent_source, timeline)
    # Compare physical coordinates, never raw energy-index eigenvector phases.
    overlap = joint.native_embedding().reshape(
        -1, 4
    ).conj().T @ parent.native_embedding().reshape(-1, 8)
    dc = source.model((0, 0, 0)).eigenvectors
    pv = parent_source.model((0, 0, 0)).eigenvectors
    projection = dc.conj().T @ overlap @ pv
    expected = projection @ parent_density @ projection.conj().T
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    for i in range(3):
        actual_charge = np.trace(actual @ dc.conj().T @ joint.charge_operator(i) @ dc)
        expected_charge = np.trace(
            parent_density @ pv.conj().T @ parent.charge_operator(i) @ pv
        )
        np.testing.assert_allclose(actual_charge, expected_charge, atol=1e-5, rtol=0)
    assert_density(actual)


def test_joint_static_coherence_phase_uses_coupled_energies() -> None:
    joint = joint_model(baths=False)
    thermal = ftf.model_source(joint, (), resonators=resonators(baths=False))
    initial = np.zeros((4, 4), dtype=np.complex128)
    initial[:2, :2] = 0.5
    source = ModelSource(initial, thermal.model, thermal.flux_expansion())
    actual = evolve(source, batch(12.0))
    energy = joint.energies_ghz
    expected = 0.5 * np.exp(-2j * np.pi * (energy[0] - energy[1]) * 12)
    np.testing.assert_allclose(actual[0, 1], expected, atol=1e-9, rtol=0)


def test_delayed_short_flux_pulse_survives_long_idle() -> None:
    # Slow circuits allow adaptive steps far longer than the 2 ns pulse.
    joint = joint_model(baths=False, retained=2, scale=1e-4)
    source = ftf.model_source(joint, (), resonators=resonators(baths=False))
    pulse = ftf.FluxTrack(
        "flux",
        np.array([0.0]),
        np.array([2.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        (EnvelopeSpec("constant"),),
        np.array([0]),
        np.array([True]),
    )
    flux = ftf.schedule(
        (pulse,),
        dc_flux=joint.dc_flux,
        lines=(FluxLine("flux", (0.0, 0.0, 0.1), delay_ns=400.0),),
        sample_period_ns=2.0,
        points=1,
    )
    actual = evolve(source, batch(1000.0, flux=flux))
    # Independent event geometry: idle [0,400), pulse [400,402), idle [402,1000).
    # The initial Gibbs state is stationary before 400 ns. Integrate only the
    # pulse with the strict oracle, then apply the analytic DC coherence phase.
    manual = FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0, 0, 100]]))
    waveform = FluxWaveform(
        np.array([0, 2]),
        np.array([0.0, 2.0]),
        np.tile(joint.dc_flux + [0, 0, 0.1], (2, 1)),
    )
    after_pulse = strict_reference(source, batch(2.0, flux=manual, waveform=waveform))
    energy = source.model((0, 0, 0)).energies_ghz
    expected = after_pulse * np.exp(
        -2j * np.pi * 598 * (energy[:, None] - energy[None, :])
    )
    assert np.max(np.abs(expected - source.initial_density)) > 1e-5
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
