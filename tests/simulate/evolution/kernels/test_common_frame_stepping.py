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
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.secular import SecularDrive, SecularModel


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
    timeline = TimelineBatch(
        np.arange(4),
        (track, replace(track, port="b")),
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
        Engine(1, kernels=tuple(k for k in REGISTRY if k.name != "multitone_rwa"))
    ) as engine:
        result = engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)
    assert result.paths.segments_by_kernel == {"common_frame": 4}
    np.testing.assert_allclose(
        result.densities, np.broadcast_to(expected, (4, 3, 3)), atol=1e-3, rtol=0
    )
