"""Shaped-gate prepare/apply accuracy and content-based reuse."""

from dataclasses import replace

import numpy as np
import pytest
import zcu_tools.simulate.evolution as evolution
from zcu_tools.simulate.evolution.kernel import SegmentTable

from tests.simulate.evolution.contract.kernel_suite import (
    KernelCase,
    KernelContract,
    assert_budget,
    reference,
    synthetic_model,
)


def shaped_table(size: int = 18) -> SegmentTable:
    """Three input states at six off-grid carrier phases share one gate."""
    times = np.linspace(0, 6, 13)
    envelope = evolution.EnvelopeSpec(
        "sampled",
        (np.sin(np.pi * times / 6) ** 2 + 0.12j * np.sin(2 * np.pi * times / 6)),
        times,
    )
    return SegmentTable(
        timeline=np.arange(size),
        start_ns=np.arange(size, dtype=float) * 2,
        duration_ns=np.full(size, 4.0),
        level=np.zeros((size, 1), dtype=np.int64),
        tones=np.ones(size, dtype=np.int64),
        tone_port=np.zeros((size, 1), dtype=np.int64),
        tone_frequency_ghz=np.full((size, 1), 0.13),
        tone_amplitude=np.full((size, 1), 0.055),
        tone_theta_rad=np.resize([0.137, 0.739, 1.531, 2.817, 4.291, 5.993], size)[
            :, None
        ],
        tone_envelope=np.zeros((size, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.full((size, 1), 0.5),
        envelopes=(envelope,),
    )


@pytest.fixture(params=[3, 4])
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    from zcu_tools.simulate.evolution.kernels.gate_cache import GateCacheKernel

    size: int = request.param
    model = synthetic_model(size)
    source = evolution.ModelSource(
        np.eye(size, dtype=np.complex128) / size,
        lambda level: replace(model, level=level),
    )
    ground = np.zeros((size, size), complex)
    ground[0, 0] = 1
    coherent = np.exp(0.7j * np.arange(size)) / np.sqrt(size)
    states = np.repeat(
        np.stack([ground, np.eye(size) / size, np.outer(coherent, coherent.conj())]),
        6,
        axis=0,
    )
    return KernelCase(GateCacheKernel(), source, shaped_table(), states)


class TestGateCache(KernelContract):
    pass


def test_guard_only_accepts_shaped_single_tone(kernel_case: KernelCase) -> None:
    case = kernel_case
    # Repetition count belongs to the engine, not this local guard.
    table = shaped_table(1)
    rows = np.array([0], dtype=np.int64)
    assert case.kernel.guard(table, rows, case.source, case.budget).tolist() == [True]
    constant = replace(table, envelopes=(evolution.EnvelopeSpec("constant"),))
    assert not case.kernel.guard(constant, rows, case.source, case.budget).any()
    silent = replace(
        table,
        tones=np.zeros(1, dtype=np.int64),
        tone_port=np.full((1, 1), -1, dtype=np.int64),
        tone_envelope=np.full((1, 1), -1, dtype=np.int64),
    )
    assert not case.kernel.guard(silent, rows, case.source, case.budget).any()
    dual = replace(
        table,
        tones=np.array([2]),
        **{
            name: np.repeat(getattr(table, name), 2, axis=1)
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
    assert not case.kernel.guard(dual, rows, case.source, case.budget).any()


def test_keys_use_content_and_exclude_start_phase(kernel_case: KernelCase) -> None:
    case = kernel_case
    table = case.table
    rows = np.arange(18, dtype=np.int64)
    keys = case.kernel.prepare_keys(table, rows)
    assert keys.dtype.kind == "V"
    assert len(np.unique(keys)) == 1
    envelope = table.envelopes[0]
    duplicate = evolution.EnvelopeSpec("sampled", envelope.samples, envelope.sample_ns)
    alias = replace(
        table,
        envelopes=(envelope, duplicate),
        tone_envelope=np.ones((18, 1), dtype=np.int64),
    )
    np.testing.assert_array_equal(case.kernel.prepare_keys(alias, rows), keys)
    for field, value in (
        ("level", np.ones((18, 1), dtype=np.int64)),
        ("duration_ns", np.full(18, 3.0)),
        ("tone_frequency_ghz", np.full((18, 1), 0.14)),
        ("tone_amplitude", np.full((18, 1), -0.055)),
        ("tone_port", np.ones((18, 1), dtype=np.int64)),
        ("tone_envelope_offset_ns", np.zeros((18, 1))),
    ):
        changed = replace(table, **{field: value})
        assert not np.any(case.kernel.prepare_keys(changed, rows) == keys), field
    changed_envelope = evolution.EnvelopeSpec(
        "sampled", envelope.samples * 0.9, envelope.sample_ns
    )
    changed = replace(table, envelopes=(changed_envelope,))
    assert not np.any(case.kernel.prepare_keys(changed, rows) == keys)


def test_prepared_entries_follow_key_mapping(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.array([0, 7, 14], dtype=np.int64)
    table = replace(case.table, duration_ns=np.resize([0.0, 2.5, 4.0], 18))
    keys, representatives, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(
        keys, table, rows[representatives], case.source, case.budget
    )
    actual = case.states[rows].copy()
    case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    changed = replace(case, table=table)
    expected = np.stack(
        [reference(changed, int(row), case.states[row]) for row in rows]
    )
    assert_budget(actual, expected, case.budget)
    np.testing.assert_allclose(actual[0], case.states[0], atol=1e-14)
