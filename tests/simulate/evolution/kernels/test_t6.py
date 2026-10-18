"""Constant-tone channels through the kernel prepare/apply contract."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.workplan import WorkPlan

from ..contract.kernel_suite import (
    KernelCase,
    KernelContract,
    assert_budget,
    reference,
    synthetic_model,
)


@pytest.fixture(params=[3, 4])
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    from zcu_tools.simulate.evolution.kernels.t6 import T6Kernel

    size: int = request.param
    model = synthetic_model(size)
    source = ModelSource(
        np.eye(size, dtype=np.complex128) / size,
        lambda level: replace(model, level=level, identity=f"model-{level}"),
    )
    count = 8
    table = SegmentTable(
        timeline=np.arange(count),
        start_ns=np.arange(count, dtype=float) * 3.7,
        duration_ns=np.array([0.0, 3.35, 3.35, 0.4, 3.2, 0.25, 1.6, 1.6]),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.ones(count, dtype=np.int64),
        tone_port=np.zeros((count, 1), dtype=np.int64),
        tone_frequency_ghz=np.array([0.7, 0.7, 0.7, 0, -0.4, 0.13, 0.7, 0.7])[:, None],
        tone_amplitude=np.array([0.25, 0.08, 0.08, -0.04, 0.15, 0.0006, 0.0006, 0.04])[
            :, None
        ],
        tone_theta_rad=np.array([0.3, 0.123, 5.678, -1.53, 0.922, 2.41, 0.66, -3.1])[
            :, None
        ],
        tone_envelope=np.zeros((count, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.ones((count, 1)) * 7,
        envelopes=(EnvelopeSpec("constant"),),
    )
    vector = np.exp(1j * np.arange(size)) / np.sqrt(size)
    state = np.outer(vector, vector.conj())
    return KernelCase(T6Kernel(), source, table, np.repeat(state[None], count, axis=0))


class TestT6Kernel(KernelContract):
    pass


@pytest.mark.parametrize(
    ("frequency", "duration", "rate_scale"),
    [
        pytest.param(0.7, 4.37, 1.0, id="coherent-periods-tail"),
        pytest.param(-0.7, 4.37, 1.0, id="negative-frequency"),
        pytest.param(0.0, 0.4, 1.0, id="zero-frequency"),
        # At late phase origins, relaxation over a period loses many digits.
        # The short physical segment itself remains accurately integrable.
        pytest.param(0.7, 0.23, 10000.0, id="ill-conditioned-phase-origin"),
    ],
)
def test_channels_for_arbitrary_coherent_inputs(
    frequency: float, duration: float, rate_scale: float
) -> None:
    from zcu_tools.simulate.evolution.kernels.t6 import T6Kernel

    model = synthetic_model(3)
    model = replace(model, rates_per_ns=model.rates_per_ns * rate_scale)
    source = ModelSource(np.eye(3, dtype=np.complex128) / 3, lambda level: model)
    # Basis projectors and real/imaginary pair superpositions span Hermitian
    # inputs. One prepared key must work for all of them, not just a thermal state.
    vectors = list(np.eye(3, dtype=complex))
    for left in range(3):
        for right in range(left + 1, 3):
            for phase in (1.0, 1j):
                vector = np.zeros(3, dtype=complex)
                vector[left], vector[right] = 1 / np.sqrt(2), phase / np.sqrt(2)
                vectors.append(vector)
    states = np.stack([np.outer(vector, vector.conj()) for vector in vectors])
    count = len(states)
    table = SegmentTable(
        timeline=np.arange(count),
        start_ns=np.arange(count, dtype=float) * 0.37,
        duration_ns=np.full(count, duration),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.ones(count, dtype=np.int64),
        tone_port=np.zeros((count, 1), dtype=np.int64),
        tone_frequency_ghz=np.full((count, 1), frequency),
        tone_amplitude=np.full((count, 1), 0.15),
        tone_theta_rad=np.linspace(-2.1, 8.2, count)[:, None],
        tone_envelope=np.zeros((count, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((count, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    case = KernelCase(T6Kernel(), source, table, states, budget=1e-6)
    rows = np.arange(count, dtype=np.int64)
    prepared = case.kernel.prepare(
        case.kernel.prepare_keys(
            table, rows[:1], KernelContext(source, case.budget, WorkPlan())
        ),
        table,
        rows[:1],
        KernelContext(source, case.budget, WorkPlan()),
    )
    actual = states.copy()
    case.kernel.apply(
        prepared,
        np.zeros(count, dtype=np.int64),
        table,
        rows,
        actual,
        ApplyContext(case.budget),
    )
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(states)]
    )
    assert_budget(actual, expected, case.budget)


def test_guard_only_accepts_one_constant_tone(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(len(case.states), dtype=np.int64)
    sampled = EnvelopeSpec("sampled", np.ones(2, dtype=complex), np.array([0.0, 20.0]))
    table = replace(case.table, envelopes=(sampled,))
    assert not case.kernel.guard(
        table, rows, KernelContext(case.source, case.budget, WorkPlan())
    ).any()
    table = replace(
        case.table,
        tones=np.zeros(len(rows), dtype=np.int64),
        tone_port=np.full((len(rows), 1), -1, dtype=np.int64),
        tone_envelope=np.full((len(rows), 1), -1, dtype=np.int64),
    )
    assert not case.kernel.guard(
        table, rows, KernelContext(case.source, case.budget, WorkPlan())
    ).any()
    table = replace(
        case.table,
        tones=np.full(len(rows), 2, dtype=np.int64),
        **{
            name: np.repeat(getattr(case.table, name), 2, axis=1)
            for name in (
                "tone_port",
                "tone_frequency_ghz",
                "tone_amplitude",
                "tone_theta_rad",
                "tone_envelope",
                "tone_envelope_offset_ns",
            )
        },
    )
    assert not case.kernel.guard(
        table, rows, KernelContext(case.source, case.budget, WorkPlan())
    ).any()


def test_keys_share_phase_but_distinguish_model_and_drive(
    kernel_case: KernelCase,
) -> None:
    case = kernel_case
    rows = np.array([1, 2], dtype=np.int64)
    keys = case.kernel.prepare_keys(
        case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
    )
    assert keys.dtype.itemsize > 0
    assert keys[0] == keys[1]
    for field in (
        "duration_ns",
        "level",
        "tone_frequency_ghz",
        "tone_amplitude",
        "tone_port",
    ):
        values = getattr(case.table, field).copy()
        values[2] += 1
        changed = replace(case.table, **{field: values})
        other = case.kernel.prepare_keys(
            changed, rows, KernelContext(case.source, case.budget, WorkPlan())
        )
        assert other[0] != other[1], field
    costs = case.kernel.cost(case.table, np.arange(len(case.states), dtype=np.int64))
    assert costs.shape == (len(case.states),)
    assert np.isfinite(costs).all()
    assert (costs >= 0).all()


def test_periods_and_tail_at_tight_budget(kernel_case: KernelCase) -> None:
    case = kernel_case
    row = np.array([1], dtype=np.int64)
    duration = case.table.duration_ns.copy()
    # Three full carrier periods plus a nonzero tail exercise channel powers.
    # Long-horizon accuracy belongs to the offline benchmark, not this contract.
    duration[1] = 4.37
    table = replace(case.table, duration_ns=duration)
    budget = 1e-7
    prepared = case.kernel.prepare(
        case.kernel.prepare_keys(
            table, row, KernelContext(case.source, budget, WorkPlan())
        ),
        table,
        row,
        KernelContext(case.source, budget, WorkPlan()),
    )
    actual = case.states[row].copy()
    case.kernel.apply(
        prepared,
        np.array([0], dtype=np.int64),
        table,
        row,
        actual,
        ApplyContext(budget),
    )
    expected = reference(replace(case, table=table), 1, case.states[1])
    assert_budget(actual[0], expected, budget)
