"""Floquet's C5 seam: arbitrary densities, carrier phases and weak probes."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource, kernel

from .._engine import RecordingKernel
from ..contract.kernel_suite import (
    KernelCase,
    KernelContract,
    assert_budget,
    reference,
    synthetic_model,
)


@pytest.fixture(params=[3, 4])
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel

    size = int(request.param)
    model = synthetic_model(size)
    model = replace(
        model,
        rates_per_ns=model.rates_per_ns * 0.001,
        dephasing_per_ns=model.dephasing_per_ns * 0.001,
        drives=tuple(
            replace(
                d,
                in_phase_ghz=d.in_phase_ghz * 0.1,
                quadrature_ghz=d.quadrature_ghz * 0.1,
            )
            for d in model.drives
        ),
    )
    count = 4
    table = kernel.SegmentTable(
        timeline=np.arange(count),
        start_ns=np.array([0.0, 13.0, 2.0, 51.0]),
        duration_ns=np.array([1.1, 1.7, 0.0, 2.1]),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.full(count, 2),
        tone_port=np.zeros((count, 2), dtype=np.int64),
        tone_frequency_ghz=np.column_stack(
            (np.full(count, 3.7), [0.11, 0.23, 0.13, -0.17])
        ),
        tone_amplitude=np.column_stack(
            (np.full(count, 0.1), [0.002, -0.003, 0.004, 0.001])
        ),
        tone_theta_rad=np.array([[0.7, -0.3], [2.0, 1.4], [-1.0, 2.0], [5.0, -0.9]]),
        tone_envelope=np.zeros((count, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.full((count, 2), 2.0),
        envelopes=(EnvelopeSpec("constant"),),
    )
    vector = np.exp(1j * np.arange(size)) / np.sqrt(size)
    coherent = np.outer(vector, vector.conj())
    states = np.stack([coherent, np.eye(size) / size, coherent.conj(), coherent])
    source = ModelSource(states[0], lambda level: replace(model, level=level))
    return KernelCase(FloquetKernel(), source, table, states)


class TestFloquetContract(KernelContract):
    pass


@pytest.mark.parametrize("count", [1, 4])
def test_engine_reuses_guard_preparations_once_per_key(
    kernel_case: KernelCase, count: int
) -> None:
    """DOP853 rejection falls back to Floquet without rebuilding guard products."""
    from contextlib import closing
    from threading import Event

    from zcu_tools.simulate.evolution import (
        DriveTrack,
        Engine,
        FluxSchedule,
        TimelineBatch,
    )
    from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel

    class CountingFloquet(FloquetKernel):
        def __init__(self):
            self.prepared_keys = []

        def prepare(self, keys, table, rows, source, budget):
            self.prepared_keys.extend(key.tobytes() for key in keys)
            return super().prepare(keys, table, rows, source, budget)

    case = kernel_case
    tracks = tuple(
        DriveTrack(
            "drive",
            np.zeros(count),
            np.full(count, 1.1),
            np.full(count, frequency),
            np.full(count, phase),
            np.full(count, amplitude),
            (EnvelopeSpec("constant"),),
            np.zeros(count, dtype=np.int64),
            np.ones(count, dtype=bool),
        )
        for frequency, amplitude, phase in ((3.7, 0.1, 0.7), (0.11, 0.002, -0.3))
    )
    batch = TimelineBatch(
        np.arange(count),
        tracks,
        FluxSchedule(np.arange(count + 1), np.zeros(count), np.zeros((count, 1), int)),
        np.full(count, 1.1),
    )
    counted = CountingFloquet()
    rejected_direct = RecordingKernel("dop853", accepts=False)
    with closing(Engine(1, 0, kernels=(counted, rejected_direct))) as engine:
        actual = engine.evolve(case.source, (batch,), cancel=Event())
        assert actual.paths.segments_by_kernel == {"floquet": count}
        assert len(counted.prepared_keys) == len(set(counted.prepared_keys)) == 1
        # Evolve-local handoff must not leave hidden state on the kernel.
        engine.evolve(case.source, (batch,), cancel=Event())
        assert len(counted.prepared_keys) == 2
    rows = np.array([0], dtype=np.int64)
    prepared = case.kernel.prepare(
        case.kernel.prepare_keys(case.table, rows),
        case.table,
        rows,
        case.source,
        case.budget,
    )
    expected = case.source.initial_density[None].copy()
    case.kernel.apply(prepared, rows, case.table, rows, expected, case.budget)
    np.testing.assert_allclose(
        actual.densities, np.repeat(expected, count, axis=0), atol=1e-13
    )


def test_probe_parameters_share_prepare_but_stark_and_level_do_not(
    kernel_case: KernelCase,
) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    keys = case.kernel.prepare_keys(case.table, rows)
    assert len(np.unique(keys)) == 1
    for table in (
        replace(
            case.table, tone_frequency_ghz=case.table.tone_frequency_ghz + [1.0, 0.0]
        ),
        replace(case.table, tone_amplitude=case.table.tone_amplitude * [2.0, 1.0]),
        replace(case.table, level=np.ones((4, 1), dtype=np.int64)),
    ):
        assert not np.any(case.kernel.prepare_keys(table, rows) == keys)
    costs = case.kernel.cost(case.table, rows)
    assert costs.shape == (4,)
    assert np.all(np.isfinite(costs) & (costs >= 0))


def test_tone_order_does_not_change_physical_result(kernel_case: KernelCase) -> None:
    case = kernel_case
    swapped = replace(
        case.table,
        **{
            field: getattr(case.table, field)[:, ::-1]
            for field in (
                "tone_port",
                "tone_frequency_ghz",
                "tone_amplitude",
                "tone_theta_rad",
                "tone_envelope",
                "tone_envelope_offset_ns",
            )
        },
    )
    rows = np.arange(4, dtype=np.int64)
    assert np.array_equal(
        case.kernel.prepare_keys(swapped, rows),
        case.kernel.prepare_keys(case.table, rows),
    )
    KernelContract().test_error_within_budget(replace(case, table=swapped))


def test_nonzero_level_and_distinct_prepares(kernel_case: KernelCase) -> None:
    case = kernel_case
    levels = np.array([[0], [1], [0], [1]])
    original = case.source.model((0,))
    source = ModelSource(
        case.states[0],
        lambda level: replace(
            original,
            level=level,
            identity=f"level-{level}",
            energies_ghz=original.energies_ghz * (1 + 0.3 * level[0]),
        ),
    )
    KernelContract().test_error_within_budget(
        replace(case, source=source, table=replace(case.table, level=levels))
    )


def test_full_micromotion_with_arbitrary_start_phase(kernel_case: KernelCase) -> None:
    case = kernel_case
    original = case.source.model((0,))
    source = ModelSource(
        case.states[0],
        lambda level: replace(
            original,
            level=level,
            rates_per_ns=np.zeros_like(original.rates_per_ns),
            dephasing_per_ns=np.zeros_like(original.dephasing_per_ns),
        ),
    )
    table = replace(
        case.table,
        duration_ns=np.array([3.2, 2.3, 0.0, 1.1]),
        tone_frequency_ghz=np.tile([0.61, 0.13], (4, 1)),
        tone_amplitude=np.tile([0.2, 0.0], (4, 1)),
    )
    case = replace(case, source=source, table=table)
    rows = np.arange(4, dtype=np.int64)
    assert case.kernel.guard(table, rows, source, case.budget).all()
    keys, representatives, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(
        keys, table, rows[representatives], source, case.budget
    )
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    expected = np.stack(
        [reference(case, r, state) for r, state in enumerate(case.states)]
    )
    assert_budget(actual, expected, 2e-8)


def test_guard_rejects_strong_probe_shapes_and_untrackable_modes(
    kernel_case: KernelCase,
) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    sampled = EnvelopeSpec(
        "sampled", sample_ns=np.array([0.0, 100.0]), samples=np.ones(2, complex)
    )
    for table in (
        replace(
            case.table,
            tones=np.ones(4, dtype=np.int64),
            tone_port=np.tile([0, -1], (4, 1)),
            tone_envelope=np.tile([0, -1], (4, 1)),
        ),
        replace(case.table, envelopes=(sampled,)),
        replace(
            case.table,
            duration_ns=np.full(4, 512.0),
            tone_amplitude=np.full((4, 2), 0.5),
            tone_frequency_ghz=np.full((4, 2), 0.13),
        ),
        replace(case.table, tone_frequency_ghz=np.zeros((4, 2))),
    ):
        assert not case.kernel.guard(table, rows, case.source, case.budget).any()


def test_guard_uses_error_budget_for_omitted_harmonics(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    table = replace(case.table, duration_ns=np.full(4, 5.0))
    assert not case.kernel.guard(table, rows, case.source, 1e-12).any()


def test_probe_changes_coherent_state_and_matches_strict_reference(
    kernel_case: KernelCase,
) -> None:
    """A response larger than tolerance prevents a missing-probe implementation."""
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    keys, representatives, inverse = np.unique(
        case.kernel.prepare_keys(case.table, rows),
        return_index=True,
        return_inverse=True,
    )
    prepared = case.kernel.prepare(
        keys, case.table, rows[representatives], case.source, case.budget
    )
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, case.table, rows, actual, case.budget)
    expected = np.stack(
        [reference(case, r, state) for r, state in enumerate(case.states)]
    )
    no_probe = replace(
        case,
        table=replace(
            case.table, tone_amplitude=case.table.tone_amplitude * [1.0, 0.0]
        ),
    )
    without = np.stack(
        [reference(no_probe, r, state) for r, state in enumerate(case.states)]
    )
    assert np.max(abs(expected - without)) > 1e-4
    assert_budget(actual, expected, 2e-6)


def test_distinct_probe_port_is_not_aliased_to_stark_port(
    kernel_case: KernelCase,
) -> None:
    case = kernel_case
    original = case.source.model((0,))
    drive = original.drives[0]
    probe = replace(
        drive,
        name="probe",
        in_phase_ghz=drive.quadrature_ghz,
        quadrature_ghz=-drive.in_phase_ghz,
    )
    source = ModelSource(
        case.states[0],
        lambda level: replace(original, level=level, drives=(drive, probe)),
    )
    table = replace(case.table, tone_port=np.tile([0, 1], (4, 1)))
    case = replace(case, source=source, table=table)
    KernelContract().test_error_within_budget(case)


def test_probe_guard_failure_uses_period_propagation_in_floquet_frame(
    kernel_case: KernelCase,
) -> None:
    """Strong probes remain accepted when the Floquet approximation is valid."""
    case = kernel_case
    original = case.source.model((0,))
    probe = original.drives[0]
    stark = replace(
        probe,
        name="stark",
        in_phase_ghz=np.zeros_like(probe.in_phase_ghz),
        quadrature_ghz=np.zeros_like(probe.quadrature_ghz),
    )
    source = ModelSource(
        case.states[0],
        lambda level: replace(original, level=level, drives=(stark, probe)),
    )
    table = replace(
        case.table,
        duration_ns=np.array([4.7, 3.1, 1.9, 0.0]),
        tone_port=np.tile([0, 1], (4, 1)),
        tone_frequency_ghz=np.column_stack((np.full(4, 3.7), [0.13, 0, -0.11, 0])),
        tone_amplitude=np.tile([0.2, 0.15], (4, 1)),
    )
    case = replace(case, source=source, table=table)
    rows = np.arange(4, dtype=np.int64)
    assert case.kernel.guard(table, rows, source, case.budget).all()
    keys, representatives, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(
        keys, table, rows[representatives], source, case.budget
    )
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    expected = np.stack(
        [reference(case, r, state) for r, state in enumerate(case.states)]
    )
    without = replace(
        case, table=replace(table, tone_amplitude=table.tone_amplitude * [1, 0])
    )
    free = reference(without, 0, case.states[0])
    assert np.max(abs(np.diag(expected[0] - free))) > 0.02
    assert_budget(actual, expected, 2e-8)


@pytest.mark.parametrize("zero_first", [False, True])
def test_unused_zero_port_preserves_acceptance_and_density(
    kernel_case: KernelCase, zero_first: bool
) -> None:
    from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel

    floquet = FloquetKernel()
    case = kernel_case
    model = case.source.model((0,))
    drive = model.drives[0]
    zero = replace(
        drive,
        name="unused",
        in_phase_ghz=np.zeros_like(drive.in_phase_ghz),
        quadrature_ghz=np.zeros_like(drive.quadrature_ghz),
    )
    drives = (zero, *model.drives) if zero_first else (*model.drives, zero)
    source = ModelSource(
        case.states[0], lambda level: replace(model, level=level, drives=drives)
    )
    table = replace(case.table, tone_port=case.table.tone_port + int(zero_first))
    rows = np.arange(len(case.states), dtype=np.int64)
    for budget in (case.budget, 1e-12):
        expected = floquet.guard_prepared(case.table, rows, case.source, budget)
        actual = floquet.guard_prepared(table, rows, source, budget)
        np.testing.assert_array_equal(actual.accepted, expected.accepted)
        if budget != case.budget:
            assert not actual.accepted.all()
            continue
        assert actual.accepted.all()
        keys, first, inverse = np.unique(
            floquet.prepare_keys(table, rows),
            return_index=True,
            return_inverse=True,
        )
        prepared = floquet.prepare(keys, table, rows[first], source, budget)
        baseline = case.states.copy()
        result = case.states.copy()
        handed_off = case.states.copy()
        expected_payload = expected.preparations[0].prepared
        actual_payload = actual.preparations[0].prepared
        assert prepared.nbytes == actual_payload.nbytes > expected_payload.nbytes
        before_bytes = prepared.nbytes
        floquet.apply(expected_payload, inverse, case.table, rows, baseline, budget)
        for payload, states in ((prepared, result), (actual_payload, handed_off)):
            exceeded = floquet.apply(payload, inverse, table, rows, states, budget)
            assert not exceeded.any()
            np.testing.assert_allclose(states, baseline, rtol=0, atol=1e-13)
        assert prepared.nbytes == before_bytes


def test_micromotion_bath_matches_independent_lindblad_jumps(
    kernel_case: KernelCase,
) -> None:
    """A bath-visible response protects the transformed secular dissipator."""
    case = kernel_case
    original = case.source.model((0,))
    model = replace(
        original,
        rates_per_ns=original.rates_per_ns * 10,
        dephasing_per_ns=original.dephasing_per_ns * 10,
    )
    source = ModelSource(case.states[0], lambda level: replace(model, level=level))
    closed_source = ModelSource(
        case.states[0],
        lambda level: replace(
            model,
            level=level,
            rates_per_ns=np.zeros_like(model.rates_per_ns),
            dephasing_per_ns=np.zeros_like(model.dephasing_per_ns),
        ),
    )
    table = replace(
        case.table,
        duration_ns=np.array([0.5, 1.0, 1.5, 2.0]),
        tone_frequency_ghz=np.tile([0.61, 0.13], (4, 1)),
        tone_amplitude=np.tile([0.3, 0.0], (4, 1)),
    )
    case = replace(case, source=source, table=table)
    rows = np.arange(4, dtype=np.int64)
    assert case.kernel.guard(table, rows, source, case.budget).all()
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(table, rows),
        return_index=True,
        return_inverse=True,
    )
    prepared = case.kernel.prepare(keys, table, rows[first], source, case.budget)
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    expected = np.stack(
        [reference(case, r, state) for r, state in enumerate(case.states)]
    )
    closed = replace(case, source=closed_source)
    without_bath = np.stack(
        [reference(closed, r, state) for r, state in enumerate(case.states)]
    )
    assert np.max(abs(expected - without_bath)) > 5e-5
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-6)

    # Retaining the rotating bath extends admission without dropping its effect.
    long_table = replace(table, duration_ns=np.full(4, 128.0))
    assert case.kernel.guard(long_table, rows, source, case.budget).all()
    long_case = replace(case, table=long_table)
    prepared = case.kernel.prepare(keys, long_table, rows[first], source, case.budget)
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, long_table, rows, actual, case.budget)
    expected = np.stack(
        [reference(long_case, r, state) for r, state in enumerate(case.states)]
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-6)
    assert case.kernel.guard(long_table, rows, closed_source, case.budget).all()
