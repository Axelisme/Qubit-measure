"""Synthetic common-frame models and independent strict lab reference."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from zcu_tools.simulate.evolution import (
    DriveTrack,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.secular import SecularDrive, SecularModel


def model_source(coherent=False, bath=True):
    x = np.array(
        [[0.007, 0.025, 0.004], [0.025, -0.003, 0.021], [0.004, 0.021, 0.002]],
        dtype=np.complex128,
    )
    rates = np.zeros((3, 3))
    if bath:
        rates[0, 1], rates[1, 2] = 0.001, 0.0015
    model = SecularModel(
        (0,),
        "common-frame-synthetic",
        np.array([0.0, 0.29, 0.603]),
        np.eye(3, dtype=complex),
        (
            SecularDrive("a", x, np.zeros_like(x)),
            SecularDrive(
                "b", np.asarray(0.7 * x, dtype=np.complex128), np.zeros_like(x)
            ),
        ),
        rates,
        np.full((3, 3), 0.0002 if bath else 0.0) * (1 - np.eye(3)),
    )
    state = np.array([1.0, 1j, 0.5], dtype=complex)
    initial = (
        np.outer(state, state.conj()) / np.vdot(state, state)
        if coherent
        else np.diag([0.7, 0.2, 0.1]).astype(complex)
    )
    return ModelSource(
        np.asarray(initial, dtype=np.complex128),
        lambda level: replace(model, level=level),
    )


def batch(durations, *, gain=1.0, frequency=0.294, shaped=False):
    durations = np.asarray(durations, dtype=float)
    count = len(durations)
    envelope = (
        EnvelopeSpec(
            "sampled",
            np.array([0.3 + 0.1j, 0.6 - 0.2j, 0.8 + 0.15j]),
            np.array([0.0, 12.0, 32.0]),
        )
        if shaped
        else EnvelopeSpec("constant")
    )
    tracks = tuple(
        DriveTrack(
            port,
            np.full(count, start),
            durations - start,
            np.full(count, carrier),
            np.full(count, phase),
            np.full(count, gain),
            (envelope,),
            np.zeros(count, dtype=np.int64),
            np.ones(count, dtype=bool),
        )
        for port, start, carrier, phase in (
            ("a", 0.0, 0.29, 0.37),
            ("b", 5.0 if shaped else 0.0, frequency, -0.63),
        )
    )
    return TimelineBatch(
        np.arange(count),
        tracks,
        FluxSchedule(
            np.arange(count + 1), np.zeros(count), np.zeros((count, 1), dtype=np.int64)
        ),
        durations,
    )


def strict_lab(source, timeline) -> NDArray[np.complex128]:
    model = source.model((0,))
    outgoing = model.rates_per_ns.sum(axis=0)
    decay = (outgoing[:, None] + outgoing[None, :]) / 2 + model.dephasing_per_ns
    generator = model.rates_per_ns - np.diag(outgoing)
    results = []
    for point, duration in enumerate(timeline.end_ns):

        def rhs(time, flat, point=point):
            rho = flat.reshape(3, 3)
            h = np.diag(model.energies_ghz).astype(complex)
            for drive, track in zip(model.drives, timeline.tracks, strict=True):
                start = track.start_ns[point]
                if start <= time <= start + track.duration_ns[point]:
                    local = time - start
                    envelope = track.envelopes[0].values(np.array([local]))[0]
                    z = (
                        track.amplitude[point]
                        * envelope
                        * np.exp(
                            1j
                            * (
                                2 * np.pi * track.frequency_ghz[point] * time
                                + track.phase_rad[point]
                            )
                        )
                    )
                    h += z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz
            result = -2j * np.pi * (h @ rho - rho @ h) - decay * rho
            np.fill_diagonal(
                result,
                np.diag(-2j * np.pi * (h @ rho - rho @ h)) + generator @ np.diag(rho),
            )
            return result.ravel()

        solved = solve_ivp(
            rhs,
            (0.0, duration),
            source.initial_density.ravel(),
            method="DOP853",
            rtol=2e-10,
            atol=2e-12,
            max_step=0.4,
        )
        assert solved.success
        results.append(solved.y[:, -1].reshape(3, 3))
    return np.array(results, dtype=np.complex128)
