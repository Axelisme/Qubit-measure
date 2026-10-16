"""Constant-tone channels through the kernel prepare/apply contract."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable

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
        duration_ns=np.array([0.0, 12.35, 12.35, 4.0, 9.2, 2.5, 8.3, 8.3]),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.ones(count, dtype=np.int64),
        tone_port=np.zeros((count, 1), dtype=np.int64),
        tone_frequency_ghz=np.array([0.7, 0.7, 0.7, 0, -0.4, 0.13, 0.7, 0.7])[:, None],
        tone_amplitude=np.array([0.25, 0.08, 0.08, -0.04, 0.15, 0.0006, 0.0006, 0.04])[:, None],
        tone_theta_rad=np.array([0.3, 0.123, 5.678, -1.53, 0.922, 2.41, 0.66, -3.1])[:, None],
        tone_envelope=np.zeros((count, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.ones((count, 1)) * 7,
        envelopes=(EnvelopeSpec("constant"),),
    )
    vector = np.exp(1j * np.arange(size)) / np.sqrt(size)
    state = np.outer(vector, vector.conj())
    return KernelCase(T6Kernel(), source, table, np.repeat(state[None], count, axis=0))


class TestT6Kernel(KernelContract):
    pass


def test_guard_only_accepts_one_constant_tone(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(len(case.states), dtype=np.int64)
    sampled = EnvelopeSpec("sampled", np.ones(2, dtype=complex), np.array([0.0, 20.0]))
    table = replace(case.table, envelopes=(sampled,))
    assert not case.kernel.guard(table, rows, case.source, case.budget).any()
    table = replace(
        case.table,
        tones=np.zeros(len(rows), dtype=np.int64),
        tone_port=np.full((len(rows), 1), -1, dtype=np.int64),
        tone_envelope=np.full((len(rows), 1), -1, dtype=np.int64),
    )
    assert not case.kernel.guard(table, rows, case.source, case.budget).any()
    table = replace(
        case.table,
        tones=np.full(len(rows), 2, dtype=np.int64),
        **{name: np.repeat(getattr(case.table, name), 2, axis=1) for name in (
            "tone_port", "tone_frequency_ghz", "tone_amplitude", "tone_theta_rad",
            "tone_envelope", "tone_envelope_offset_ns",
        )},
    )
    assert not case.kernel.guard(table, rows, case.source, case.budget).any()


def test_keys_share_phase_but_distinguish_model_and_drive(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.array([1, 2], dtype=np.int64)
    keys = case.kernel.prepare_keys(case.table, rows)
    assert keys.dtype.itemsize > 0
    assert keys[0] == keys[1]
    for field in ("duration_ns", "level", "tone_frequency_ghz", "tone_amplitude", "tone_port"):
        values = getattr(case.table, field).copy()
        values[2] += 1
        changed = replace(case.table, **{field: values})
        other = case.kernel.prepare_keys(changed, rows)
        assert other[0] != other[1], field
    costs = case.kernel.cost(case.table, np.arange(len(case.states), dtype=np.int64))
    assert costs.shape == (len(case.states),)
    assert np.isfinite(costs).all()
    assert (costs >= 0).all()


def test_long_segment_at_tight_budget(kernel_case: KernelCase) -> None:
    case = kernel_case
    row = np.array([1], dtype=np.int64)
    duration = case.table.duration_ns.copy()
    duration[1] = 160.37
    table = replace(case.table, duration_ns=duration)
    budget = 1e-7
    prepared = case.kernel.prepare(
        case.kernel.prepare_keys(table, row), table, row, case.source, budget
    )
    actual = case.states[row].copy()
    case.kernel.apply(prepared, np.array([0], dtype=np.int64), table, row, actual, budget)
    expected = reference(replace(case, table=table), 1, case.states[1])
    assert_budget(actual[0], expected, budget)
