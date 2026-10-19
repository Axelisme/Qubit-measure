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
    FluxWaveform,
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
from zcu_tools.simulate.evolution.kernels._rwa_gate import GatePrepared
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


def test_source_history_amortizes_later_small_acquire_and_invalidates_old_memo():
    src = source()
    with closing(Engine(1, 1_000_000)) as engine:
        small = batch(2)
        cold = engine.evolve(src, (small,), cancel=Event())
        assert cold.paths.segments_by_kernel.get("shaped_sambe") == 2
        engine.evolve(src, (batch(160),), cancel=Event())
        later = engine.evolve(src, (small,), cancel=Event())
        assert later.paths.segments_by_kernel.get("single_tone_gate") == 2


def test_source_acquire_order_has_same_routes_with_and_without_cache_retention():
    routes = []
    for retain in (True, False):
        src = source()
        run = []
        with closing(Engine(1, 1_000_000)) as persistent:
            for count, phase in ((160, 0.0), (2, 0.713), (2, 0.0)):
                if retain:
                    result = persistent.evolve(
                        src, (batch(count, phase),), cancel=Event()
                    )
                else:
                    # Replacing the Engine releases both caches, not source history.
                    with closing(Engine(1, 0)) as empty:
                        result = empty.evolve(
                            src, (batch(count, phase),), cancel=Event()
                        )
                run.append(result.paths.segments_by_kernel)
                assert result.paths.segments_by_kernel.get("single_tone_gate") == count
        routes.append(run)
    assert routes[0] == routes[1]


def test_source_close_discards_usage_even_if_engine_preparations_survive():
    src = source()
    with closing(Engine(1, 1_000_000)) as engine:
        engine.evolve(src, (batch(160),), cancel=Event())
        warm = engine.evolve(src, (batch(2),), cancel=Event())
        assert warm.paths.segments_by_kernel.get("single_tone_gate") == 2
        src.close()
        reset = engine.evolve(src, (batch(2),), cancel=Event())
        assert reset.paths.segments_by_kernel.get("shaped_sambe") == 2


def test_repeated_identical_acquires_count_even_when_results_are_memoized():
    src = source()
    with closing(Engine(1, 1_000_000)) as engine:
        small = batch(2)
        result = engine.evolve(src, (small,), cancel=Event())
        for _ in range(79):
            result = engine.evolve(src, (small,), cancel=Event())
        assert result.paths.segments_by_kernel.get("single_tone_gate") == 2


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
        assert isinstance(prepared, GatePrepared)
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


@pytest.mark.parametrize("frequency", [0.6, -0.6])
def test_admission_keeps_row_flux_and_budget_outside_covariant_equivalence(frequency):
    kernel = next(k for k in REGISTRY if k.name == "single_tone_gate")
    # All rows share a playback key, but only timelines 0 and 2 have fixed flux.
    table = SegmentTable(
        np.arange(3),
        np.array([0.0, 3.125, 71.0]),
        np.full(3, 126.0),
        np.zeros((3, 1), dtype=np.int64),
        np.ones(3, dtype=np.int64),
        np.zeros((3, 1), dtype=np.int64),
        np.full((3, 1), frequency),
        np.full((3, 1), 0.08),
        np.array([[0.0], [0.713], [-1.2]]),
        np.zeros((3, 1), dtype=np.int64),
        np.full((3, 1), 4.0),
        (envelope(),),
        FluxWaveform(
            np.arange(0, 7, 2),
            np.array([0.0, 200.0] * 3),
            np.array([[0.0], [0.0], [0.0], [0.001], [0.2], [0.2]]),
        ),
    )
    with closing(WorkPlan()) as plan:
        context = KernelContext(source(), 1e-3, plan)
        assert len(np.unique(kernel.prepare_keys(table, np.arange(3), context))) == 1
        for rows in (np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([2, 0, 1])):
            np.testing.assert_array_equal(
                kernel.guard(table, rows, context), np.array([True, False, True])[rows]
            )
            assert not kernel.guard(table, rows, replace(context, budget=1e-4)).any()
        # Long held support refines the calibrated Strang estimate, not a veto.
        long = replace(table, duration_ns=np.full(3, 100_000.0))
        np.testing.assert_array_equal(
            kernel.guard(long, np.arange(3), context), [True, False, True]
        )
        long_keys = kernel.prepare_keys(long, np.array([0]), context)
        long_gate = kernel.prepare(long_keys, long, np.array([0]), context)
        assert isinstance(long_gate, GatePrepared)
        # tau=100000, Gamma=2.5e-5, V=1.6e-4: ceil(sqrt(64000)).
        assert long_gate.unitaries.shape[1] == 253
        shifted = replace(table, tone_envelope_offset_ns=np.full((3, 1), 8.0))
        assert np.all(
            kernel.prepare_keys(shifted, np.arange(3), context)
            != kernel.prepare_keys(table, np.arange(3), context)
        )


@pytest.mark.parametrize("bath_scale,segments", [(1.0, 4), (1500.0, 5)])
def test_strang_split_adapts_without_changing_the_four_step_control(
    bath_scale, segments
):
    src = source()
    model = src.model((0,))
    model = replace(
        model,
        identity=f"adaptive-bath-{bath_scale}",
        rates_per_ns=model.rates_per_ns * bath_scale,
        dephasing_per_ns=model.dephasing_per_ns * bath_scale,
    )
    src = ModelSource(src.initial_density, lambda level: replace(model, level=level))
    table = SegmentTable(
        timeline=np.arange(2),
        start_ns=np.array([0.0, 37.0]),
        duration_ns=np.full(2, 128.0),
        level=np.zeros((2, 1), dtype=np.int64),
        tones=np.ones(2, dtype=np.int64),
        tone_port=np.zeros((2, 1), dtype=np.int64),
        tone_frequency_ghz=np.full((2, 1), 0.6),
        tone_amplitude=np.full((2, 1), 0.8),
        tone_theta_rad=np.array([[0.0], [0.713]]),
        tone_envelope=np.zeros((2, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((2, 1)),
        envelopes=(envelope(),),
    )
    rows = np.arange(2)
    kernel = next(k for k in REGISTRY if k.name == "single_tone_gate")
    with closing(WorkPlan()) as plan:
        context = KernelContext(src, 1e-3, plan)
        assert kernel.guard(table, rows, context).all(), (
            "a split-only rejection must refine the secular time steps"
        )
        keys = kernel.prepare_keys(table, rows, context)
        prepared = kernel.prepare(keys[:1], table, rows[:1], context)
        assert isinstance(prepared, GatePrepared)
        assert prepared.unitaries.shape[1] == segments
        actual = np.repeat(src.initial_density[None], 2, axis=0)
        kernel.apply(
            prepared,
            np.zeros(2, dtype=np.int64),
            table,
            rows,
            actual,
            ApplyContext(1e-3),
        )
        direct = next(k for k in REGISTRY if k.name == "dop853")
        from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Plan

        reference = np.repeat(src.initial_density[None], 2, axis=0)
        direct.apply(
            direct.prepare(
                direct.prepare_keys(table, rows, context), table, rows, context
            ),
            rows,
            table,
            rows,
            reference,
            ApplyContext(
                1e-6, DOP853Plan(np.full(2, "lab"), np.full(2, -1, dtype=np.int64))
            ),
        )
        assert (
            np.max(np.abs(np.diagonal(actual - reference, axis1=-2, axis2=-1))) < 1e-2
        )


def test_mixed_inventory_does_not_charge_rejected_gate_builds_to_usable_keys():
    original = batch(161)
    track = original.tracks[0]
    gains = track.amplitude.copy()
    gains[-1] = 40.0  # Same carrier/shape domain, but outside numerical RWA admission.
    mixed = replace(original, tracks=(replace(track, amplitude=gains),))
    with closing(Engine(1, 1_000_000)) as engine:
        result = engine.evolve(source(), (mixed,), cancel=Event())
        assert result.paths.segments_by_kernel.get("single_tone_gate") == 160
        assert sum(result.paths.segments_by_kernel.values()) == 321
        # Complete-inventory cold costs still reject genuinely unamortized runs.
        small = engine.evolve(source(), (batch(2),), cancel=Event())
        assert small.paths.segments_by_kernel.get("shaped_sambe") == 2


def test_admission_uses_sum_of_quadrature_norms_not_their_maximum():
    src = source()
    model = src.model((0,))
    drive = model.drives[0]
    # Off-resonant edges eliminate the phase proxy, isolating the full norm ratio.
    model = replace(
        model,
        identity="two-quadrature-admission",
        energies_ghz=np.array([0.0, 0.9, 1.8]),
        drives=(replace(drive, quadrature_ghz=drive.in_phase_ghz),),
    )
    src = ModelSource(src.initial_density, lambda level: replace(model, level=level))
    b = batch(1)
    table = SegmentTable(
        np.array([0]),
        np.array([0.0]),
        np.array([128.0]),
        np.zeros((1, 1), dtype=np.int64),
        np.ones(1, dtype=np.int64),
        np.zeros((1, 1), dtype=np.int64),
        np.array([[0.6]]),
        np.array([[40.0]]),
        np.zeros((1, 1)),
        np.zeros((1, 1), dtype=np.int64),
        np.zeros((1, 1)),
        b.tracks[0].envelopes,
    )
    kernel = next(k for k in REGISTRY if k.name == "single_tone_gate")
    with closing(WorkPlan()) as plan:
        assert not kernel.guard(table, np.array([0]), KernelContext(src, 1e-3, plan))[0]


@pytest.mark.parametrize("workers", [1, 2])
def test_engine_covariant_density_is_bitwise_stable_under_point_permutation(workers):
    original = batch(1024)
    order = np.arange(1023, -1, -1)
    t = original.tracks[0]
    permuted = replace(
        original,
        points=original.points[order],
        tracks=(
            replace(
                t,
                **{
                    name: getattr(t, name)[order]
                    for name in (
                        "start_ns",
                        "duration_ns",
                        "frequency_ghz",
                        "phase_rad",
                        "amplitude",
                        "envelope_index",
                        "active",
                    )
                },
            ),
        ),
        end_ns=original.end_ns[order],
    )
    with closing(Engine(1, 0, chunks_per_worker=1)) as reference:
        expected = reference.evolve(source(), (original,), cancel=Event())
    with closing(Engine(workers, 0, chunks_per_worker=7)) as engine:
        actual = engine.evolve(source(), (permuted,), cancel=Event())
    assert actual.paths.segments_by_kernel == expected.paths.segments_by_kernel
    assert actual.paths.segments_by_kernel["single_tone_gate"] == 1024
    np.testing.assert_array_equal(
        actual.densities[np.argsort(actual.points)],
        expected.densities[np.argsort(expected.points)],
    )
