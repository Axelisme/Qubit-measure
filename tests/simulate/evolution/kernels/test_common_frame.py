"""Common-frame dynamics through Engine's injected-kernel seam.

The adapter is deliberately test-owned: production dispatch and residency have
not selected this numerical variant yet. Strict lab references use SciPy's
independent density solver, including the untransformed secular bath.
"""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.kernels.common_frame import (
    admit_common_frame,
    advance_common_frame,
    build_common_frame,
)
from zcu_tools.simulate.evolution.secular import SecularDrive, SecularModel

from .._engine import RecordingKernel


class CommonFrameAdapter(RecordingKernel):
    """Supply the numerical owner under an existing injectable family slot."""

    def __init__(self):
        super().__init__("multitone_rwa")
        self.admissions = []
        self.advances = 0
        self.source = None

    def guard(self, table, rows, context):
        self.source = context.source
        accepted = []
        for row in rows:
            model = context.source.model(tuple(table.level[row]))
            count = table.tones[row]
            frame = build_common_frame(
                model,
                tuple(table.tone_port[row, :count]),
                table.tone_frequency_ghz[row, :count],
            )
            admission = admit_common_frame(frame, table, int(row))
            self.admissions.append(admission)
            accepted.append(admission.accepted)
        return np.array(accepted, dtype=bool)

    def apply(self, prepared, key_index, table, rows, states, context):
        assert self.source is not None
        for index, row in enumerate(rows):
            model = self.source.model(tuple(table.level[row]))
            count = table.tones[row]
            frame = build_common_frame(
                model,
                tuple(table.tone_port[row, :count]),
                table.tone_frequency_ghz[row, :count],
            )
            states[index] = advance_common_frame(frame, table, int(row), states[index])
            self.advances += 1
        return np.zeros(len(rows), dtype=bool)


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


def evolve(source, timeline, adapter):
    kernels = tuple(adapter if k.name == adapter.name else k for k in REGISTRY)
    with closing(Engine(1, 0, kernels=kernels)) as engine:
        return engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)


@pytest.mark.parametrize("coherent", [False, True])
def test_complex_envelope_offsets_and_held_endpoint_match_strict_lab(coherent):
    source = model_source(coherent)
    timeline = batch([16, 38, 67], shaped=True)
    adapter = CommonFrameAdapter()
    result = evolve(source, timeline, adapter)
    expected = strict_lab(source, timeline)
    assert adapter.advances > 0
    # Coherences make endpoint dressing and the DC frame restoration observable.
    np.testing.assert_allclose(result.densities, expected, atol=0.015, rtol=0)
    np.testing.assert_allclose(
        np.trace(result.densities, axis1=1, axis2=2), 1, atol=1e-12
    )


def test_rabi_period_and_decay_follow_lab_curve():
    source = model_source()
    timeline = batch(np.linspace(8, 104, 13), gain=0.6)
    adapter = CommonFrameAdapter()
    result = evolve(source, timeline, adapter)
    expected = strict_lab(source, timeline)
    actual_pop = result.densities[:, 1, 1].real
    expected_pop = expected[:, 1, 1].real
    np.testing.assert_allclose(actual_pop, expected_pop, atol=0.025, rtol=0)
    assert abs(np.argmax(actual_pop) - np.argmax(expected_pop)) <= 1
    assert np.ptp(actual_pop) > 0.15
    assert all(a.accepted for a in adapter.admissions)


@pytest.mark.parametrize(
    "gain,duration,frequency", [(0.5, 32, 0.292), (1.0, 64, 0.294), (2.0, 96, 0.298)]
)
def test_near_carrier_parameter_offsets_remain_admitted(gain, duration, frequency):
    source = model_source()
    timeline = batch([duration], gain=0.3 * gain, frequency=frequency)
    adapter = CommonFrameAdapter()
    result = evolve(source, timeline, adapter)
    assert all(a.accepted for a in adapter.admissions)
    np.testing.assert_allclose(
        result.densities.real.diagonal(axis1=1, axis2=2),
        strict_lab(source, timeline).real.diagonal(axis1=1, axis2=2),
        atol=0.1,
        rtol=0,
    )


@pytest.mark.parametrize("gain,frequency", [(15.0, 0.294), (1.0, 0.7)])
def test_over_budget_and_expensive_rows_use_existing_fallback(gain, frequency):
    source = model_source()
    timeline = batch([16], gain=gain, frequency=frequency)
    adapter = CommonFrameAdapter()
    result = evolve(source, timeline, adapter)
    assert not any(a.accepted for a in adapter.admissions)
    assert adapter.advances == 0
    with closing(
        Engine(1, 0, kernels=tuple(k for k in REGISTRY if k.name != adapter.name))
    ) as engine:
        original = engine.evolve(source, (timeline,), cancel=Event(), budget=0.01)
    np.testing.assert_array_equal(result.densities, original.densities)
