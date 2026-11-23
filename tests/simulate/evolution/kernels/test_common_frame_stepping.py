"""Density accuracy for features between fixed-step quadrature samples."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.secular import SecularDrive, SecularModel


def _with_last_interval_midpoint(envelope: EnvelopeSpec) -> EnvelopeSpec:
    # Keep the waveform unchanged while using a different grid to prevent folding.
    return replace(
        envelope,
        samples=np.insert(envelope.samples, -1, envelope.samples[-2:].mean()),
        sample_ns=np.insert(envelope.sample_ns, -1, envelope.sample_ns[-2:].mean()),
    )


def test_long_resonant_drive_preserves_spectator_coherences():
    # The circular drive is exact in the rotating frame. A coherent spectator
    # exposes relative phase errors that populations and trace cannot detect.
    strength, carrier, spectator, phase = 0.2, 0.29, 0.61, 0.39
    x = np.array([[0, strength, 0], [strength, 0, 0], [0, 0, 0]], dtype=np.complex128)
    y = np.array(
        [[0, 1j * strength, 0], [-1j * strength, 0, 0], [0, 0, 0]],
        dtype=np.complex128,
    )
    model = SecularModel(
        (0,),
        "coherent-circular-resonance",
        np.array([0.0, carrier, spectator]) + 0.43,
        np.eye(3, dtype=np.complex128),
        (SecularDrive("a", x / 2, y / 2), SecularDrive("b", x / 2, y / 2)),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    initial = np.array([np.sqrt(0.5), 0.5j, 0.5])
    source = ModelSource(np.outer(initial, initial.conj()), lambda _: model)
    durations = np.array([11.0, 137.0, 503.0, 1000.0])
    count = len(durations)
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.2, 1.0], dtype=np.complex128),
        np.array([0.0, 1000.0]),
    )
    track = DriveTrack(
        "a",
        np.zeros(count),
        durations,
        np.full(count, carrier),
        np.full(count, phase),
        np.ones(count),
        (envelope,),
        np.zeros(count, dtype=np.int64),
        np.ones(count, dtype=bool),
    )
    track_b = replace(
        track, port="b", envelopes=(_with_last_interval_midpoint(envelope),)
    )
    timeline = TimelineBatch(
        np.arange(count),
        (track, track_b),
        FluxSchedule(
            np.arange(count + 1), np.zeros(count), np.zeros((count, 1), dtype=np.int64)
        ),
        durations,
    )
    # Integrate the linear envelope analytically, rather than reproducing CF4.
    angle = 2 * np.pi * strength * (0.2 * durations + 0.8 * durations**2 / 2000)
    states = np.column_stack(
        (
            initial[0] * np.cos(angle)
            - 1j * np.exp(1j * phase) * initial[1] * np.sin(angle),
            (
                initial[1] * np.cos(angle)
                - 1j * np.exp(-1j * phase) * initial[0] * np.sin(angle)
            )
            * np.exp(-2j * np.pi * carrier * durations),
            initial[2] * np.exp(-2j * np.pi * spectator * durations),
        )
    )
    expected = states[:, :, None] * states[:, None, :].conj()
    with closing(
        Engine(1, families=("common_frame", "multitone_gate", "floquet", "dop853"))
    ) as engine:
        result = engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)
    assert result.paths.segments_by_kernel == {"common_frame": count}
    np.testing.assert_allclose(result.densities, expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        result.densities, result.densities.conj().swapaxes(-1, -2), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        np.trace(result.densities, axis1=-2, axis2=-1), 1, atol=1e-12, rtol=0
    )
    assert np.min(np.linalg.eigvalsh(result.densities)) >= -1e-12


def test_narrow_envelope_knots_preserve_resonant_pulse_area():
    # A circular drive has no omitted counter-rotating channel. Its exact
    # resonant solution depends only on pulse area, independently of CF4.
    strength, carrier, duration = 0.2, 0.29, 1000.0
    x = np.array([[0, strength, 0], [strength, 0, 0], [0, 0, 0]], dtype=np.complex128)
    y = np.array(
        [[0, 1j * strength, 0], [-1j * strength, 0, 0], [0, 0, 0]],
        dtype=np.complex128,
    )
    model = SecularModel(
        (0,),
        "circular-resonance",
        np.array([0.0, carrier, 0.61]),
        np.eye(3, dtype=np.complex128),
        (SecularDrive("a", x / 2, y / 2), SecularDrive("b", x / 2, y / 2)),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    source = ModelSource(
        np.diag([1.0, 0.0, 0.0]).astype(np.complex128), lambda _: model
    )
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0, 0.0000002, 1.0000006, 0.000001, 0.001], dtype=np.complex128),
        np.array([0.0, 0.2, 0.6, 1.0, duration]),
    )
    track = DriveTrack(
        "a",
        np.zeros(4),
        np.full(4, duration),
        np.full(4, carrier),
        np.zeros(4),
        np.ones(4),
        (envelope,),
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=bool),
    )
    track_b = replace(
        track, port="b", envelopes=(_with_last_interval_midpoint(envelope),)
    )
    timeline = TimelineBatch(
        np.arange(4),
        (track, track_b),
        FluxSchedule(np.arange(5), np.zeros(4), np.zeros((4, 1), dtype=np.int64)),
        np.full(4, duration),
    )
    angle = 2 * np.pi * strength * (0.4 + 0.001 * duration / 2)
    psi = np.array(
        [
            np.cos(angle),
            -1j * np.sin(angle) * np.exp(-2j * np.pi * carrier * duration),
            0,
        ]
    )
    expected = np.outer(psi, psi.conj())
    with closing(
        Engine(1, families=("common_frame", "multitone_gate", "floquet", "dop853"))
    ) as engine:
        result = engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)
    assert result.paths.segments_by_kernel == {"common_frame": 4}
    np.testing.assert_allclose(
        result.densities, np.broadcast_to(expected, (4, 3, 3)), atol=1e-3, rtol=0
    )
