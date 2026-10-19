"""Joint-basis multi-tone approximation at the Engine and C5 seams."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
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
from zcu_tools.simulate.evolution.kernel import KernelContext, SegmentTable
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.workplan import WorkPlan

from tests.simulate.evolution.contract.kernel_suite import KernelCase, KernelContract


def model_source() -> ModelSource:
    x = np.array([[0, 0.002, 0], [0.002, 0, 0.001], [0, 0.001, 0]], complex)
    model = SecularModel(
        (0,),
        "joint-three",
        np.array([0.0, 0.3, 0.9]),
        np.eye(3, dtype=complex),
        (
            SecularDrive("a", x, (x * 0.2).astype(np.complex128)),
            SecularDrive(
                "b", (x * 0.7).astype(np.complex128), (-x * 0.3).astype(np.complex128)
            ),
        ),
        np.array([[0, 0.0002, 0], [0, 0, 0.0001], [0, 0, 0]]),
        np.ones((3, 3)) * 0.0001 - np.eye(3) * 0.0001,
    )
    vector = np.array([1, 1j, -1], complex) / np.sqrt(3)
    return ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )


def smooth() -> EnvelopeSpec:
    times = np.linspace(0, 128, 129)
    return EnvelopeSpec(
        "sampled", np.exp(-(((times - 64) / 32) ** 2) / 2).astype(complex), times
    )


def timeline(kind: str = "smooth", tones: int = 2) -> TimelineBatch:
    env = smooth()
    if kind == "complex":
        env = replace(env, samples=env.samples * (1 + 0.1j))
    if kind == "rough":
        env = replace(
            env, samples=np.where(np.arange(129) % 2, 1.0, 0.0).astype(complex)
        )
    frequencies = [0.3, -0.6, 0.9][:tones]
    if kind == "degenerate":
        frequencies[1] = -0.30001
    if kind == "dc":
        frequencies[1] = 0.0
    tracks = tuple(
        DriveTrack(
            "a" if slot == 0 else "b",
            np.array([7.0]),
            np.array([128.0]),
            np.array([f]),
            np.array([0.4 + slot]),
            np.array([0.08]),
            (env,),
            np.array([0]),
            np.array([True]),
        )
        for slot, f in enumerate(frequencies)
    )
    return TimelineBatch(
        np.array([0]),
        tracks,
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([135.0]),
    )


def test_smooth_multiple_drives_route_with_explicit_admission():
    routes = {}
    with closing(Engine(1, 1000000)) as engine:
        for kind in ("smooth", "complex", "rough", "degenerate", "dc"):
            result = engine.evolve(model_source(), (timeline(kind),), cancel=Event())
            routes[kind] = dict(result.paths.segments_by_kernel)
        result = engine.evolve(model_source(), (timeline(tones=3),), cancel=Event())
        routes["three"] = dict(result.paths.segments_by_kernel)
    expected = {kind: {"t4": 1, "dop853": 1} for kind in routes}
    expected["smooth"] = expected["three"] = {"t4": 1, "multitone_rwa": 1}
    assert routes == expected


@pytest.fixture
def kernel_case() -> KernelCase:
    family = next((k for k in REGISTRY if k.name == "multitone_rwa"), None)
    assert family is not None, "smooth simultaneous tones need their own RWA family"
    source = model_source()
    table = SegmentTable(
        timeline=np.arange(3),
        start_ns=np.array([0.0, 7.0, 19.0]),
        duration_ns=np.array([8.0, 16.0, 12.0]),
        level=np.zeros((3, 1), dtype=np.int64),
        tones=np.array([2, 3, 2]),
        tone_port=np.array([[0, 1, -1], [0, 1, 1], [1, 0, -1]]),
        tone_frequency_ghz=np.tile([0.3, -0.6, 0.9], (3, 1)),
        tone_amplitude=np.tile([0.08, -0.06, 0.04], (3, 1)),
        tone_theta_rad=np.array([[0.4, 1.4, 0], [1.1, -0.8, 2.2], [-0.2, 0.7, 0]]),
        tone_envelope=np.array([[0, 0, -1], [0, 0, 0], [0, 0, -1]]),
        tone_envelope_offset_ns=np.tile([40.0, 48.0, 32.0], (3, 1)),
        envelopes=(smooth(),),
    )
    return KernelCase(
        family, source, table, np.repeat(source.initial_density[None], 3, axis=0)
    )


class TestMultitoneRWA(KernelContract):
    pass


@pytest.mark.parametrize(
    "field",
    [
        "tone_frequency_ghz",
        "tone_port",
        "tone_amplitude",
        "tone_theta_rad",
        "tone_envelope_offset_ns",
        "envelopes",
        "budget",
    ],
)
def test_numerical_keys_distinguish_all_drive_content(
    kernel_case: KernelCase, field: str
):
    case = kernel_case
    rows = np.array([0])
    table = case.table
    budget = case.budget
    context = KernelContext(case.source, budget, WorkPlan())
    original = case.kernel.prepare_keys(table, rows, context)
    if field == "envelopes":
        table = replace(
            table, envelopes=(replace(smooth(), samples=smooth().samples * 0.99),)
        )
    elif field == "budget":
        budget *= 2
    else:
        values = getattr(table, field).copy()
        values[0, 1] = 0 if field == "tone_port" else values[0, 1] + 0.01
        table = replace(table, **{field: values})
    changed = case.kernel.prepare_keys(
        table, rows, KernelContext(case.source, budget, WorkPlan())
    )
    assert original.tobytes() != changed.tobytes()


def test_tight_budget_declines_approximation(kernel_case: KernelCase):
    case = kernel_case
    assert not case.kernel.guard(
        case.table, np.arange(3), KernelContext(case.source, 1e-8, WorkPlan())
    ).any()


def test_third_carrier_changes_joint_density_within_lab_budget(kernel_case: KernelCase):
    from zcu_tools.simulate.evolution.kernel import ApplyContext

    from tests.simulate.evolution.contract.kernel_suite import assert_budget, reference

    case = kernel_case
    model = case.source.model((0,))
    coupling = model.drives[1].in_phase_ghz.copy()
    coupling[0, 2] = coupling[2, 0] = 0.004
    model = replace(
        model, drives=(model.drives[0], replace(model.drives[1], in_phase_ghz=coupling))
    )
    initial = np.diag([1.0, 0.0, 0.0]).astype(complex)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    table = replace(
        case.table,
        tones=np.full(3, 3),
        tone_port=np.tile([0, 1, 1], (3, 1)),
        tone_envelope=np.zeros((3, 3), dtype=np.int64),
        duration_ns=np.full(3, 128.0),
        tone_envelope_offset_ns=np.zeros((3, 3)),
    )
    states = []
    rows = np.array([1])
    for enabled in (False, True):
        amplitude = table.tone_amplitude.copy()
        amplitude[1, 2] = 0.2 if enabled else 0.0
        selected = replace(table, tone_amplitude=amplitude)
        context = KernelContext(source, case.budget, WorkPlan())
        assert case.kernel.guard(selected, rows, context).all()
        keys = case.kernel.prepare_keys(selected, rows, context)
        prepared = case.kernel.prepare(keys, selected, rows, context)
        actual = initial[None].copy()
        case.kernel.apply(
            prepared, np.array([0]), selected, rows, actual, ApplyContext(case.budget)
        )
        expected = reference(replace(case, source=source, table=selected), 1, initial)
        assert_budget(actual[0], expected, case.budget)
        states.append(actual[0])
    assert np.max(np.abs(states[1] - states[0])) > 0.01


def test_continuous_flux_motion_declines_even_at_one_quantized_level(
    kernel_case: KernelCase,
):
    from zcu_tools.simulate.evolution import FluxWaveform

    case = kernel_case
    # The interior excursion returns to its original flux. Endpoint-only
    # inspection would incorrectly admit it while every level is still zero.
    waveform = FluxWaveform(
        np.array([0, 3, 6, 9]),
        np.tile([0.0, 10.0, 40.0], 3),
        np.tile([[0.2], [0.201], [0.2]], (3, 1)),
    )
    table = replace(case.table, flux_waveform=waveform)
    context = KernelContext(case.source, case.budget, WorkPlan())
    assert not case.kernel.guard(table, np.arange(3), context).any()
    fixed = replace(table, flux_waveform=replace(waveform, flux=np.full((9, 1), 0.2)))
    assert case.kernel.guard(fixed, np.arange(3), context).all()


def test_gate_counter_rotating_phase_rejects_long_weak_pulses(
    kernel_case: KernelCase,
):
    case = kernel_case
    model = case.source.model((0,))
    model = replace(
        model,
        energies_ghz=model.energies_ghz * 10,
        drives=tuple(
            replace(
                drive,
                in_phase_ghz=(drive.in_phase_ghz * 100).astype(np.complex128),
                quadrature_ghz=(drive.quadrature_ghz * 100).astype(np.complex128),
            )
            for drive in model.drives
        ),
    )
    source = ModelSource(
        case.source.initial_density, lambda level: replace(model, level=level)
    )
    table = replace(case.table, tone_frequency_ghz=case.table.tone_frequency_ghz * 10)
    context = KernelContext(source, case.budget, WorkPlan())
    rows = np.arange(3)
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import admit_rows

    assert admit_rows(table, rows, context).all()
    long = replace(table, duration_ns=np.full(3, 1280.0))
    assert not admit_rows(long, rows, context).any()


@pytest.mark.parametrize("frequency", [0.3, -0.3])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_window_excluded_edges_accumulate_envelope_squared_phase(frequency, scale):
    from zcu_tools.simulate.evolution.kernel import ApplyContext
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import MultitoneRWAKernel

    # Circular quadratures leave only the upper-triangular positive Fourier
    # edge and its conjugate. Both tones are outside the retained window.
    x_a = np.zeros((3, 3), dtype=np.complex128)
    x_b = np.zeros_like(x_a)
    x_a[0, 1] = x_a[1, 0] = 0.02
    x_b[1, 2] = x_b[2, 1] = 0.015
    y_a = 1j * (np.triu(x_a) - np.tril(x_a))
    y_b = 1j * (np.triu(x_b) - np.tril(x_b))
    energies = np.array([0.0, 0.5, 0.9])
    model = SecularModel(
        (0,),
        "virtual-phase",
        energies,
        np.eye(3, dtype=np.complex128),
        (SecularDrive("a", x_a, y_a), SecularDrive("b", x_b, y_b)),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    initial = np.full((3, 3), 1 / 3, dtype=np.complex128)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    times = np.arange(161, dtype=np.float64)
    samples = scale * (0.2 + 0.8 * times / times[-1])
    env = EnvelopeSpec("sampled", samples.astype(np.complex128), times)
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([27.0]),
        duration_ns=np.array([128.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array([[frequency, -0.9]]),
        tone_amplitude=np.array([[0.3, -0.4]]),
        tone_theta_rad=np.array([[0.7, -1.2]]),
        tone_envelope=np.array([[0, 0]]),
        tone_envelope_offset_ns=np.array([[0.0, 16.0]]),
        envelopes=(env,),
    )
    kernel = MultitoneRWAKernel()
    rows = np.array([0])
    context = KernelContext(source, 1e-3, WorkPlan())
    assert kernel.guard(table, rows, context).all()
    keys = kernel.prepare_keys(table, rows, context)
    prepared = kernel.prepare(keys, table, rows, context)
    actual = initial[None].copy()
    kernel.apply(prepared, rows, table, rows, actual, ApplyContext(1e-3))

    # Integrate the square of each piecewise-linear playback exactly, not
    # the square of its area or just its peak/plateau. The spectator state
    # makes both tones' individual diagonal shifts observable.
    areas = []
    for offset in (0, 16):
        values = samples[offset : offset + 129]
        areas.append(
            np.sum((values[:-1] ** 2 + values[:-1] * values[1:] + values[1:] ** 2) / 3)
        )
    phase_a = 0.3**2 * 0.02**2 / (frequency - 0.5) * areas[0]
    phase_b = (-0.4) ** 2 * 0.015**2 / (-0.9 - 0.4) * areas[1]
    integrated_shift = np.array([phase_a, -phase_a + phase_b, -phase_b])
    phases = np.exp(-2j * np.pi * (energies * 128 + integrated_shift))
    expected = initial * phases[:, None] * phases.conj()[None, :]
    np.testing.assert_allclose(actual[0], expected, atol=2e-7, rtol=0)


def test_gate_cache_and_row_rwa_share_window_excluded_shifts():
    from zcu_tools.simulate.evolution.kernel import ApplyContext
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import RWAPrepared

    # No retained Fourier edges: the analytic answer is entirely the virtual
    # shift. Agreement alone would miss both paths silently omitting it.
    energies = np.array([0.0, 0.7, 1.2])
    drives = []
    for port, target, coupling in (("a", 1, 0.01), ("b", 2, 0.008)):
        positive = np.zeros((3, 3), dtype=np.complex128)
        positive[0, target] = coupling
        drives.append(
            SecularDrive(port, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "cached-virtual-phase",
        energies,
        np.eye(3, dtype=np.complex128),
        tuple(drives),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    initial = np.full((3, 3), 1 / 3, dtype=np.complex128)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    count, duration = 24, 128.0
    times = np.array([0.0, duration / 2, duration])
    env = EnvelopeSpec("sampled", (0.5 + times / (2 * duration)).astype(complex), times)
    table = SegmentTable(
        timeline=np.arange(count),
        start_ns=np.arange(count, dtype=np.float64) * 3.125,
        duration_ns=np.full(count, duration),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.full(count, 2),
        tone_port=np.tile([0, 1], (count, 1)),
        tone_frequency_ghz=np.tile([0.6, 0.3], (count, 1)),
        # Keep the one-photon transfer below threshold: this case tests only shifts.
        tone_amplitude=np.tile([-0.3, 0.3], (count, 1)),
        tone_theta_rad=np.arange(count * 2, dtype=np.float64).reshape(count, 2) * 0.13,
        tone_envelope=np.zeros((count, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((count, 2)),
        envelopes=(env,),
    )
    integral = duration * (0.5**2 + 0.5 * 1.0 + 1.0**2) / 3
    shift_a = (-0.3 * 0.01) ** 2 / (0.6 - 0.7) * integral
    shift_b = (0.3 * 0.008) ** 2 / (0.3 - 1.2) * integral
    shifts = np.array([shift_a + shift_b, -shift_a, -shift_b])
    phase = np.exp(-2j * np.pi * (energies * duration + shifts))
    expected = initial * phase[:, None] * phase.conj()[None, :]
    results = []
    rows = np.arange(count)
    for name in ("multitone_gate", "multitone_rwa"):
        kernel = next(k for k in REGISTRY if k.name == name)
        with closing(WorkPlan()) as plan:
            context = KernelContext(source, 1e-3, plan)
            assert kernel.guard(table, rows, context).all()
            keys = kernel.prepare_keys(table, rows, context)
            _, indices, inverse = np.unique(
                keys, return_index=True, return_inverse=True
            )
            prepared = kernel.prepare(keys[indices], table, rows[indices], context)
            if name == "multitone_rwa":
                assert isinstance(prepared, RWAPrepared)
                assert not np.any(prepared.positive)
                assert not np.any(prepared.negative)
            actual = np.repeat(initial[None], count, axis=0)
            kernel.apply(prepared, inverse, table, rows, actual, ApplyContext(1e-3))
        np.testing.assert_allclose(
            actual, np.repeat(expected[None], count, axis=0), atol=2e-7, rtol=0
        )
        results.append(actual)
    np.testing.assert_allclose(results[0], results[1], atol=2e-7, rtol=0)


def constant_pair(offsets=(0.0, 0.0)):
    """Small coupled pair with a 32.049 MHz conditional double excitation."""
    energies = np.array([0.0, 6.7, 7.5, 14.232049])
    drives = []
    for name, edges in (("a", ((0, 1), (2, 3))), ("b", ((0, 2), (1, 3)))):
        positive = np.zeros((4, 4), dtype=np.complex128)
        for i, j in edges:
            positive[i, j] = 0.01
        drives.append(
            SecularDrive(name, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "conditional-pair",
        energies,
        np.eye(4, dtype=np.complex128),
        tuple(drives),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
    )
    vector = np.array([1, 1j, -1, 1], dtype=np.complex128) / 2
    source = ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )
    tracks = tuple(
        DriveTrack(
            name,
            np.array([0.0]),
            np.array([128.0]),
            np.array([frequency + offset]),
            np.array([0.3]),
            np.array([1.0]),
            (EnvelopeSpec("constant"),),
            np.array([0]),
            np.array([True]),
        )
        for name, frequency, offset in zip(("a", "b"), (6.7, 7.5), offsets, strict=True)
    )
    batch = TimelineBatch(
        np.array([0]),
        tracks,
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([128.0]),
    )
    return source, batch


@pytest.mark.parametrize("port", ["a", "b"])
def test_same_port_constant_pair_retains_original_two_tone_routing(port):
    source = model_source()
    batch = timeline()
    batch = replace(
        batch,
        tracks=tuple(
            replace(track, port=port, envelopes=(EnvelopeSpec("constant"),))
            for track in batch.tracks
        ),
    )
    original_kernels = tuple(k for k in REGISTRY if k.name != "multitone_rwa")
    with closing(Engine(1, 1000000, kernels=original_kernels)) as engine:
        original = engine.evolve(source, (batch,), cancel=Event())
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(source, (batch,), cancel=Event())
    assert set(original.paths.segments_by_kernel) & {"floquet", "dop853"}
    assert result.paths.segments_by_kernel == original.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, original.densities)


def test_same_port_sampled_pair_keeps_joint_rwa():
    batch = timeline()
    batch = replace(batch, tracks=tuple(replace(t, port="a") for t in batch.tracks))
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(model_source(), (batch,), cancel=Event())
    assert result.paths.segments_by_kernel == {"t4": 1, "multitone_rwa": 1}


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("frequency", [0.3, 0.301437, -0.301437])
def test_close_carriers_keep_joint_interference_and_coherent_density(
    sampled, frequency
):
    batch = timeline()
    batch = replace(
        batch,
        tracks=tuple(
            replace(
                track,
                frequency_ghz=np.array([f]),
                amplitude=np.array([0.8]),
                envelopes=(smooth() if sampled else EnvelopeSpec("constant"),),
            )
            for track, f in zip(batch.tracks, (0.3, frequency), strict=True)
        ),
    )
    source = model_source()
    with closing(Engine(1, 1000000)) as engine:
        actual = engine.evolve(source, (batch,), cancel=Event())
        expected = engine.evolve(source, (batch,), budget=1e-7, cancel=Event())
    assert actual.paths.segments_by_kernel == {"t4": 1, "multitone_rwa": 1}
    # RWA is approximate: use the existing constant-pair density tolerance,
    # still much tighter than the qualitative population allowance of 0.1.
    np.testing.assert_allclose(actual.densities, expected.densities, atol=0.01, rtol=0)
    assert np.max(np.abs(actual.densities[0] - source.initial_density)) > 0.01


def test_constant_pair_uses_joint_rwa_and_preserves_coherent_joint_input():
    source, batch = constant_pair()
    with closing(Engine(1, 1000000)) as engine:
        result = engine.evolve(source, (batch,), cancel=Event())
    assert result.paths.segments_by_kernel == {"multitone_rwa": 1}
    # A tight-budget Engine must retain full-drive dynamics as the reference.
    with closing(Engine(1, 1000000)) as engine:
        reference_result = engine.evolve(source, (batch,), budget=1e-7, cancel=Event())
    np.testing.assert_allclose(
        result.densities, reference_result.densities, atol=0.01, rtol=0
    )
    assert np.max(np.abs(result.densities[0] - source.initial_density)) > 0.01


@pytest.mark.parametrize("offsets", [(0.084, -0.052), (0.084, 0.084)])
def test_constant_combination_resonance_restores_contributing_edges(offsets):
    source, safe = constant_pair()
    _, risky = constant_pair(offsets)
    # The effective channel is unsafe to omit, but its four-state retained
    # graph fits. Restoring its paths must keep the full-drive joint dynamics.
    with closing(Engine(1, 1000000)) as engine:
        admitted = engine.evolve(source, (safe,), cancel=Event())
        rejected = engine.evolve(source, (risky,), cancel=Event())
    assert (admitted.paths.segments_by_kernel, rejected.paths.segments_by_kernel) == (
        {"multitone_rwa": 1},
        {"multitone_rwa": 1},
    )
    with closing(Engine(1, 1000000)) as engine:
        expected = engine.evolve(source, (risky,), budget=1e-7, cancel=Event())
    np.testing.assert_allclose(
        rejected.densities, expected.densities, atol=0.01, rtol=0
    )


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("detuning", [0.0, 0.07])
def test_omission_admission_accepts_strong_retained_rows(sampled, detuning):
    from zcu_tools.simulate.evolution.kernel import ApplyContext
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import MultitoneRWAKernel

    from tests.simulate.evolution.contract.kernel_suite import reference

    drives = []
    for port, target in (("a", 1), ("b", 2)):
        positive = np.zeros((3, 3), dtype=np.complex128)
        positive[0, target] = 0.04
        drives.append(
            SecularDrive(port, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "strong-circular",
        np.array([0.0, 0.3, 0.9]),
        np.eye(3, dtype=np.complex128),
        tuple(drives),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    initial = np.full((3, 3), 1 / 3, dtype=np.complex128)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([128.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array([[0.3 + detuning, 0.9]]),
        tone_amplitude=np.ones((1, 2)),
        tone_theta_rad=np.array([[0.7, -1.2]]),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(smooth() if sampled else EnvelopeSpec("constant"),),
    )
    kernel = MultitoneRWAKernel()
    budget = 1e-3
    with closing(WorkPlan()) as plan:
        context = KernelContext(source, budget, plan)
        rows = np.array([0])
        assert kernel.guard(table, rows, context).all()
        keys = kernel.prepare_keys(table, rows, context)
        prepared = kernel.prepare(keys, table, rows, context)
        actual = initial[None].copy()
        kernel.apply(prepared, rows, table, rows, actual, ApplyContext(budget))
    expected = reference(KernelCase(kernel, source, table, initial[None]), 0, initial)
    # Lab comparison includes the RWA DOP853 playback's budget-level error.
    np.testing.assert_allclose(actual[0], expected, atol=budget, rtol=0)


@pytest.mark.parametrize("frequencies", [(0.3, 0.9), (0.3, 0.301437)])
def test_connected_retained_graph_over_block_limit_declines(frequencies):
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import MultitoneRWAKernel

    size = 17
    positive = np.diag(np.full(size - 1, 0.001), 1).astype(np.complex128)
    drive = SecularDrive("a", positive + positive.T, 1j * (positive - positive.T))
    model = SecularModel(
        (0,),
        "connected-chain",
        np.arange(size, dtype=np.float64) * 0.3,
        np.eye(size, dtype=np.complex128),
        (drive, replace(drive, name="b")),
        np.zeros((size, size)),
        np.zeros((size, size)),
    )
    source = ModelSource(
        np.eye(size, dtype=np.complex128) / size,
        lambda level: replace(model, level=level),
    )
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([128.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array([frequencies]),
        tone_amplitude=np.ones((1, 2)),
        tone_theta_rad=np.zeros((1, 2)),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    with closing(WorkPlan()) as plan:
        assert (
            not MultitoneRWAKernel()
            .guard(table, np.array([0]), KernelContext(source, 1e-3, plan))
            .any()
        )


def test_combination_restores_strongest_parent_without_fast_weak_paths(kernel_case):
    from zcu_tools.simulate.evolution.kernel import ApplyContext
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import RWAPrepared

    from tests.simulate.evolution.contract.kernel_suite import reference

    case = kernel_case
    drives = []
    for name, edges in (("a", ((0, 1), (0, 2))), ("b", ((1, 3), (2, 3)))):
        positive = np.zeros((4, 4), dtype=np.complex128)
        for (i, j), coupling in zip(edges, (0.01, 0.00001), strict=True):
            positive[i, j] = coupling
        drives.append(
            SecularDrive(name, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "dominant-raman-parent",
        np.array([0.0, 0.6, 0.7, 1.2]),
        np.eye(4, dtype=np.complex128),
        tuple(drives),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
    )
    initial = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex128)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    table = replace(
        case.table,
        tones=np.full(3, 2),
        tone_port=np.tile([0, 1, -1], (3, 1)),
        tone_frequency_ghz=np.tile([0.3, 0.9, 0.0], (3, 1)),
        tone_amplitude=np.ones((3, 3)),
        tone_envelope=np.tile([0, 0, -1], (3, 1)),
        envelopes=(EnvelopeSpec("constant"),),
        duration_ns=np.full(3, 128.0),
    )
    rows = np.array([0])
    with closing(WorkPlan()) as plan:
        context = KernelContext(source, case.budget, plan)
        assert case.kernel.guard(table, rows, context).all()
        keys = case.kernel.prepare_keys(table, rows, context)
        prepared = case.kernel.prepare(keys, table, rows, context)
        assert isinstance(prepared, RWAPrepared)
        assert np.count_nonzero(prepared.positive) == 2
        actual = initial[None].copy()
        case.kernel.apply(
            prepared, rows, table, rows, actual, ApplyContext(case.budget)
        )
    expected = reference(
        KernelCase(case.kernel, source, table, initial[None]), 0, initial
    )
    np.testing.assert_allclose(actual[0], expected, atol=case.budget, rtol=0)


def test_residual_phase_restores_edges_even_below_transfer_threshold(kernel_case):
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import RWAPrepared

    case = kernel_case
    model = case.source.model((0,))
    positive = np.zeros((3, 3), dtype=np.complex128)
    positive[0, 1] = 0.001
    x, y = positive + positive.T, 1j * (positive - positive.T)
    model = replace(
        model,
        drives=(SecularDrive("a", x, y), SecularDrive("b", x * 0, y * 0)),
    )
    source = ModelSource(
        case.source.initial_density, lambda level: replace(model, level=level)
    )
    table = replace(
        case.table,
        tones=np.full(3, 2),
        tone_port=np.tile([0, 1, -1], (3, 1)),
        tone_frequency_ghz=np.tile([0.2, 0.9, 0.0], (3, 1)),
        tone_amplitude=np.ones((3, 3)),
        tone_envelope=np.tile([0, 0, -1], (3, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    # g/d=.01 gives transfer <.0004 at every duration. Only accumulated
    # residual phase, after the explicit second-order shift, needs restoration.
    for duration, retained in ((10.0, False), (1e12, True)):
        selected = replace(table, duration_ns=np.full(3, duration))
        with closing(WorkPlan()) as plan:
            context = KernelContext(source, case.budget, plan)
            rows = np.array([0])
            assert case.kernel.guard(selected, rows, context).all()
            keys = case.kernel.prepare_keys(selected, rows, context)
            prepared = case.kernel.prepare(keys, selected, rows, context)
            assert isinstance(prepared, RWAPrepared)
            assert bool(np.any(prepared.positive)) == retained


def phase_proxy_pair(phase_rad, *, domain="constant"):
    """Three levels with analytically known retained norms of 0.025 GHz/gain."""
    drives = []
    for port, target in (("a", 1), ("b", 2)):
        positive = np.zeros((3, 3), dtype=np.complex128)
        positive[0, target] = 0.025
        drives.append(
            SecularDrive(port, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "phase-proxy-pair",
        np.array([0.0, 0.3, 0.9]),
        np.eye(3, dtype=np.complex128),
        tuple(drives),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    initial = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    # With distinct ports both resonant Fourier operators have norm 0.025.
    # For the same port only the 0.3 GHz edge is retained.
    inverse_frequencies = 1 / 0.3 + (0 if domain == "same_port" else 1 / 0.9)
    duration = phase_rad / (2 * np.pi * 0.025**2 * inverse_frequencies)
    sampled = EnvelopeSpec(
        "sampled",
        np.ones(3, dtype=np.complex128),
        np.array([0.0, duration / 2, duration]),
    )
    tracks = tuple(
        DriveTrack(
            "a" if domain == "same_port" else port,
            np.array([0.0]),
            np.array([duration]),
            np.array([frequency]),
            np.array([0.3]),
            np.array([1.0]),
            (sampled if domain == "sampled" else EnvelopeSpec("constant"),),
            np.array([0]),
            np.array([True]),
        )
        for port, frequency in (("a", 0.3), ("b", 0.9))
    )
    return source, TimelineBatch(
        np.array([0]),
        tracks,
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([duration]),
    )


def test_retained_only_phase_does_not_reject_long_rows():
    routes = {}
    for domain in ("constant", "same_port", "sampled"):
        for phase in (0.05, 0.07, 1.0):
            source, batch = phase_proxy_pair(phase, domain=domain)
            with closing(Engine(1, 1000000)) as engine:
                result = engine.evolve(source, (batch,), cancel=Event())
            routes[domain, phase] = "multitone_rwa" in result.paths.segments_by_kernel
            if domain == "same_port":
                # Rejected rows retain the original two-tone routing and density.
                original_kernels = tuple(
                    k for k in REGISTRY if k.name != "multitone_rwa"
                )
                with closing(Engine(1, 1000000, kernels=original_kernels)) as engine:
                    original = engine.evolve(source, (batch,), cancel=Event())
                assert (
                    result.paths.segments_by_kernel == original.paths.segments_by_kernel
                )
                np.testing.assert_array_equal(result.densities, original.densities)
    assert routes == {
        (domain, phase): domain != "same_port"
        for domain in ("constant", "same_port", "sampled")
        for phase in (0.05, 0.07, 1.0)
    }
