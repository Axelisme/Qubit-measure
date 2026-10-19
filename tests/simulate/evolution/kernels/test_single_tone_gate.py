"""Acquire-wide covariant single-tone gates and phase-independent reuse."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
import scipy.integrate
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.workplan import WorkPlan


def source():
    x = np.array([[0, 0.002, 0], [0.002, 0, 0.001], [0, 0.001, 0]], complex)
    model = SecularModel(
        (0,),
        "single-gate",
        np.array([0.0, 0.6, 1.3]),
        np.eye(3, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.array([[0, 1e-5, 0], [0, 0, 2e-5], [0, 0, 0]]),
        (np.ones((3, 3)) - np.eye(3)) * 1e-5,
    )
    v = np.array([1, 1j, 0.3], complex)
    v /= np.linalg.norm(v)
    return ModelSource(np.outer(v, v.conj()), lambda level: replace(model, level=level))


def envelope():
    times = np.linspace(0, 128, 129)
    return EnvelopeSpec(
        "sampled", np.exp(-0.5 * ((times - 64) / 32) ** 2).astype(complex), times
    )


def batch(count, phase=0.0):
    track = DriveTrack(
        "drive",
        np.arange(count, dtype=float) * 3.125,
        np.full(count, 128.0),
        np.full(count, 0.6),
        np.linspace(-1, 2, count) + phase,
        np.full(count, 0.08),
        (envelope(),),
        np.zeros(count, dtype=np.int64),
        np.ones(count, dtype=bool),
    )
    return TimelineBatch(
        np.arange(count),
        (track,),
        FluxSchedule(
            np.arange(count + 1), np.zeros(count), np.zeros((count, 1), dtype=np.int64)
        ),
        track.start_ns + 128.0,
    )


def test_inventory_amortizes_once_but_small_inventory_stays_sambe_even_when_warm(
    monkeypatch,
):
    calls = []
    original = scipy.integrate.solve_ivp

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.integrate, "solve_ivp", observed)
    with closing(Engine(1, 1_000_000)) as engine:
        cold_small = engine.evolve(source(), (batch(2),), cancel=Event())
        assert cold_small.paths.segments_by_kernel.get("shaped_sambe") == 2
        for phase in (0.0, 0.713):
            result = engine.evolve(source(), (batch(1024, phase),), cancel=Event())
            assert result.paths.segments_by_kernel.get("single_tone_gate") == 1024
        assert len(calls) == 4, (
            "one canonical key builds four U only once across phases"
        )
        warm_small = engine.evolve(source(), (batch(2, 0.713),), cancel=Event())
        assert (
            warm_small.paths.segments_by_kernel == cold_small.paths.segments_by_kernel
        )


@pytest.mark.parametrize("frequency", [0.6, -0.6])
def test_covariance_preserves_complex_density_for_sliced_support_and_key_domains(
    frequency,
):
    kernel = next((k for k in REGISTRY if k.name == "single_tone_gate"), None)
    assert kernel is not None, "single-tone RWA gates need a named numerical family"
    # Different starts and theta share one sliced gate; final sample hold is legal.
    table = SegmentTable(
        timeline=np.arange(3),
        start_ns=np.array([0.0, 3.125, 71.0]),
        duration_ns=np.full(3, 126.0),
        level=np.zeros((3, 1), dtype=np.int64),
        tones=np.ones(3, dtype=np.int64),
        tone_port=np.zeros((3, 1), dtype=np.int64),
        tone_frequency_ghz=np.full((3, 1), frequency),
        tone_amplitude=np.full((3, 1), 0.08),
        tone_theta_rad=np.array([[0.0], [0.713], [-1.2]]),
        tone_envelope=np.zeros((3, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.full((3, 1), 4.0),
        envelopes=(envelope(),),
    )
    rows = np.arange(3)
    src = source()
    context = KernelContext(src, 1e-3, WorkPlan())
    assert kernel.guard(table, rows, context).all()
    keys = kernel.prepare_keys(table, rows, context)
    assert len(np.unique(keys)) == 1
    prepared = kernel.prepare(keys[:1], table, rows[:1], context)
    states = np.repeat(src.initial_density[None], 3, axis=0)
    kernel.apply(
        prepared, np.zeros(3, dtype=np.int64), table, rows, states, ApplyContext(1e-3)
    )
    direct = next(k for k in REGISTRY if k.name == "dop853")
    from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Plan

    direct_keys = direct.prepare_keys(table, rows, context)
    reference = np.repeat(src.initial_density[None], 3, axis=0)
    direct.apply(
        direct.prepare(direct_keys, table, rows, context),
        rows,
        table,
        rows,
        reference,
        ApplyContext(
            1e-3, DOP853Plan(np.full(3, "lab"), np.full(3, -1, dtype=np.int64))
        ),
    )
    assert np.max(np.abs(states - reference)) < 1e-2
    for field in (
        "duration_ns",
        "tone_envelope_offset_ns",
        "tone_amplitude",
        "tone_frequency_ghz",
    ):
        values = getattr(table, field).copy()
        values[0] += 0.001
        changed = kernel.prepare_keys(replace(table, **{field: values}), rows, context)
        assert changed[0] != keys[0]
    assert not kernel.guard(table, rows, replace(context, budget=1e-4)).any()
