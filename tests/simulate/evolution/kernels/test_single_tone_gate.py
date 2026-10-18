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
        assert len(calls) == 0, "single-tone cold builds stay in compiled integration"
        warm_small = engine.evolve(source(), (batch(2, 0.713),), cancel=Event())
        assert (
            warm_small.paths.segments_by_kernel == cold_small.paths.segments_by_kernel
        )


def test_compiled_cold_cost_admits_reference_inventory_but_not_small_inventory():
    with closing(Engine(1, 1_000_000)) as engine:
        for phase in (0.0, 0.713):
            reference = engine.evolve(source(), (batch(160, phase),), cancel=Event())
            assert reference.paths.segments_by_kernel.get("single_tone_gate") == 160
            small = engine.evolve(source(), (batch(2, phase),), cancel=Event())
            assert small.paths.segments_by_kernel.get("shaped_sambe") == 2


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


@pytest.mark.parametrize("frequency", [0.6, -0.6])
def test_compiled_four_unitaries_match_scipy_at_sliced_absolute_intervals(
    frequency, monkeypatch
):
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import (
        interaction_hamiltonian,
        rwa_coefficients,
    )

    kernel = next(k for k in REGISTRY if k.name == "single_tone_gate")
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([37.0]),
        duration_ns=np.array([126.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.ones(1, dtype=np.int64),
        tone_port=np.zeros((1, 1), dtype=np.int64),
        tone_frequency_ghz=np.array([[frequency]]),
        tone_amplitude=np.array([[0.08]]),
        tone_theta_rad=np.array([[0.713]]),
        tone_envelope=np.zeros((1, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.array([[4.0]]),
        envelopes=(envelope(),),
    )
    src = source()
    rows = np.array([0])
    with closing(WorkPlan()) as plan:
        context = KernelContext(src, 1e-3, plan)
        keys = kernel.prepare_keys(table, rows, context)
        calls = []
        solve = scipy.integrate.solve_ivp

        def observed(*args, **kwargs):
            calls.append(1)
            return solve(*args, **kwargs)

        monkeypatch.setattr(scipy.integrate, "solve_ivp", observed)
        prepared = kernel.prepare(keys, table, rows, context)
        model = src.model((0,))
        coefficients = rwa_coefficients(model, (0,), (frequency,))
        energy = model.energies_ghz + prepared.charges[0, :, 0] * frequency
        energy -= energy[0]
        env = table.envelopes[0]

        def rhs(time, state):
            amplitude = 0.08 * np.interp(time + 4, env.sample_ns, env.samples)
            h = interaction_hamiltonian(
                time, energy, coefficients, np.array([amplitude])
            )
            return (-2j * np.pi * h @ state.reshape(3, 3)).ravel()

        reference = []
        boundaries = np.linspace(0, 126, 5)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            result = solve(
                rhs,
                (start, end),
                np.eye(3, dtype=complex).ravel(),
                method="DOP853",
                rtol=1e-9,
                atol=1e-11,
                max_step=4.0,
            )
            assert result.success
            endpoint = result.y[:, -1].reshape(3, 3)
            endpoint *= np.exp(-2j * np.pi * energy * end)[:, None]
            endpoint *= np.exp(2j * np.pi * energy * start)[None, :]
            reference.append(endpoint)
        np.testing.assert_allclose(prepared.unitaries[0], reference, atol=1e-7, rtol=0)
        assert not calls, "single-tone preparation must use the compiled builder"
