"""Interaction-picture accuracy, lab-frame chains and row partition contracts."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import ApplyContext, SegmentTable
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel

from tests.simulate.evolution._engine import batch, track
from tests.simulate.evolution.contract.kernel_suite import (
    KernelCase,
    reference,
    synthetic_model,
)


def common_controls() -> KernelCase:
    """Coherent, dissipative rows with signed gains and complex playback."""
    model = replace(synthetic_model(), energies_ghz=np.array([0.0, 0.41, 2.7]))
    vector = np.array([1.0, 0.4j, -0.3], complex)
    vector /= np.linalg.norm(vector)
    density = np.outer(vector, vector.conj())
    source = ModelSource(density, lambda _: model)
    time = np.array([0.0, 0.3, 0.8, 1.4, 2.0, 2.7, 3.0])
    envelope = EnvelopeSpec(
        "sampled", np.array([0, 0.2j, 0.6 + 0.2j, 1, 0.4 - 0.1j, 0.1, 0], complex), time
    )
    gains = np.array([0.51, 0.98, -0.7, 0.031, 0.049, 0.63])
    n = len(gains)
    table = SegmentTable(
        timeline=np.arange(n),
        start_ns=np.arange(n, dtype=float),
        duration_ns=np.full(n, 2.5),
        level=np.zeros((n, 1), dtype=np.int64),
        tones=np.ones(n, dtype=np.int64),
        tone_port=np.zeros((n, 1), dtype=np.int64),
        tone_frequency_ghz=np.full((n, 1), -0.37),
        tone_amplitude=gains[:, None],
        tone_theta_rad=np.full((n, 1), 0.43),
        tone_envelope=np.zeros((n, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.full((n, 1), 0.2),
        envelopes=(envelope,),
    )
    states = np.repeat(density[None], n, axis=0)
    # Same gain octave and controls, but very different RHS/error magnitudes.
    states[0] = np.eye(3, dtype=complex) / 3
    return KernelCase(DOP853Kernel(), source, table, states)


def evolve_rows(case: KernelCase, rows: np.ndarray) -> np.ndarray:
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(case.table, rows),
        return_index=True,
        return_inverse=True,
    )
    prepared = case.kernel.prepare(
        keys, case.table, rows[first], case.source, case.budget
    )
    actual = case.states[rows].copy()
    mask = case.kernel.apply(
        prepared, inverse, case.table, rows, actual, ApplyContext(case.budget)
    )
    assert not mask.any()
    return actual


@pytest.mark.parametrize("budget", [1e-3, 1e-8])
def test_common_control_rows_match_full_lab_frame_reference(budget: float) -> None:
    case = replace(common_controls(), budget=budget)
    actual = evolve_rows(case, np.arange(len(case.states)))
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(case.states)]
    )
    np.testing.assert_allclose(actual, expected, atol=budget, rtol=0)


def test_row_order_and_apply_partition_preserve_endpoints() -> None:
    case = common_controls()
    rows = np.arange(len(case.states))
    expected = evolve_rows(case, rows)
    permutation = np.array([4, 2, 0, 5, 1, 3])
    actual = evolve_rows(case, permutation)[np.argsort(permutation)]
    split = np.concatenate([evolve_rows(case, rows[:2]), evolve_rows(case, rows[2:])])
    singles = np.concatenate([evolve_rows(case, np.array([row])) for row in rows])
    for result in (actual, split, singles):
        np.testing.assert_array_equal(result, expected)


def test_different_playback_keys_remain_independent() -> None:
    case = common_controls()
    table = case.table
    # Every varying part of the common-control key matters physically.
    case = replace(
        case,
        table=replace(
            table,
            duration_ns=np.array([2.5, 2.1, 2.5, 2.5, 2.5, 2.5]),
            tone_frequency_ghz=np.array(
                [[-0.37], [-0.37], [0.29], [-0.37], [-0.37], [-0.37]]
            ),
            tone_theta_rad=np.array([[0.43], [0.43], [0.43], [1.2], [0.43], [0.43]]),
            tone_envelope=np.array([[0], [0], [0], [0], [1], [0]]),
            tone_envelope_offset_ns=np.array(
                [[0.2], [0.2], [0.2], [0.2], [0.2], [0.4]]
            ),
            envelopes=table.envelopes + (EnvelopeSpec("constant"),),
        ),
    )
    actual = evolve_rows(case, np.arange(len(case.states)))
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(case.states)]
    )
    np.testing.assert_allclose(actual, expected, atol=case.budget, rtol=0)


def fallback_case(mode: str) -> KernelCase:
    case = common_controls()
    table = case.table
    if mode == "zero":
        gains = table.tone_amplitude.copy()
        gains[0] = 0
        table = replace(table, tone_amplitude=gains)
    elif mode == "mixed":
        tones = table.tones.copy()
        tones[0] = 2
        playing = np.arange(2)[None, :] < tones[:, None]
        table = replace(
            table,
            tones=tones,
            tone_port=np.where(playing, 0, -1),
            tone_envelope=np.where(playing, 0, -1),
            **{
                name: np.repeat(getattr(table, name), 2, axis=1)
                for name in (
                    "tone_frequency_ghz",
                    "tone_amplitude",
                    "tone_theta_rad",
                    "tone_envelope_offset_ns",
                )
            },
        )
    elif mode == "chain":
        table = replace(
            table, timeline=np.zeros(6, dtype=np.int64), start_ns=np.arange(6) * 2.5
        )
    return replace(case, table=table)


@pytest.mark.parametrize("mode", ["zero", "mixed", "chain"])
def test_fallback_rows_match_full_lab_frame_reference(mode: str) -> None:
    case = replace(fallback_case(mode), budget=1e-8)
    actual = evolve_rows(case, np.arange(len(case.states)))
    lab_rows = np.arange(len(case.states)) if mode == "chain" else np.array([0])
    expected = []
    for row in lab_rows:
        state = expected[-1] if mode == "chain" and len(expected) else case.states[row]
        expected.append(reference(case, int(row), state))
    # Strict lab-frame integration is independent of the production RHS. The
    # tighter budget checks coherent endpoints without pinning solver roundoff.
    np.testing.assert_allclose(
        actual[lab_rows], np.stack(expected), atol=case.budget, rtol=0
    )


@pytest.mark.parametrize("mode", ["zero", "mixed"])
def test_mixed_eligibility_independent_rows_are_exact_under_apply_partition(
    mode: str,
) -> None:
    case = fallback_case(mode)
    rows = np.arange(len(case.states))
    whole = evolve_rows(case, rows)
    singles = np.concatenate([evolve_rows(case, np.array([row])) for row in rows])
    permutation = np.array([4, 2, 0, 5, 1, 3])
    reordered = evolve_rows(case, permutation)[np.argsort(permutation)]
    np.testing.assert_array_equal(whole, singles)
    np.testing.assert_array_equal(reordered, singles)


@pytest.mark.parametrize("mixed", [False, True])
def test_engine_worker_and_chunk_partition_preserve_independent_density(
    mixed: bool,
) -> None:
    case = common_controls()
    n = len(case.states)
    pulse = replace(
        track(n, case.table.envelopes[0], duration=2.5),
        amplitude=case.table.tone_amplitude[:, 0],
        frequency_ghz=case.table.tone_frequency_ghz[:, 0],
    )
    tracks = (pulse,)
    if mixed:
        extra = replace(
            track(n, case.table.envelopes[0], duration=2.5),
            amplitude=np.full(n, 0.12),
            active=np.arange(n) == 2,
        )
        tracks += (extra,)
    workload = replace(batch(n, duration=2.5), tracks=tracks)
    results = []
    for workers, chunks in ((1, 1), (1, 3), (2, 1), (2, 4)):
        with closing(
            Engine(workers, 0, chunks_per_worker=chunks, kernels=(DOP853Kernel(),))
        ) as engine:
            results.append(engine.evolve(case.source, (workload,), cancel=Event()))
    for result in results:
        assert result.paths.segments_by_kernel == {"dop853": n}
        np.testing.assert_array_equal(result.densities, results[0].densities)


def test_zero_tone_table_without_drive_slots_matches_lab_reference() -> None:
    case = common_controls()
    table = case.table
    controls = {
        name: getattr(table, name)[:, :0]
        for name in (
            "tone_port",
            "tone_frequency_ghz",
            "tone_amplitude",
            "tone_theta_rad",
            "tone_envelope",
            "tone_envelope_offset_ns",
        )
    }
    case = replace(
        case,
        budget=1e-5,
        table=replace(
            table,
            duration_ns=np.full(len(case.states), 0.4),
            tones=np.zeros(len(case.states), dtype=np.int64),
            envelopes=(),
            **controls,
        ),
    )
    actual = evolve_rows(case, np.arange(len(case.states)))
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(case.states)]
    )
    np.testing.assert_allclose(actual, expected, atol=case.budget, rtol=0)


def test_touching_rows_at_large_absolute_start_match_lab_frame_chain() -> None:
    case = common_controls()
    starts = case.table.start_ns.copy()
    starts[:2] = [1e8, np.float64(1e8 + 0.1)]
    case = replace(
        case,
        budget=1e-8,
        table=replace(
            case.table,
            timeline=np.zeros(len(case.states), dtype=np.int64),
            start_ns=starts,
            duration_ns=np.full(len(case.states), 0.1),
        ),
    )
    actual = evolve_rows(case, np.array([0, 1]))
    # Integrate in local time to avoid rounding the oracle's absolute endpoint.
    local = replace(case, table=replace(case.table, start_ns=np.zeros(len(starts))))
    first = reference(local, 0, case.states[0])
    second = reference(local, 1, first)
    np.testing.assert_allclose(
        actual, np.stack([first, second]), atol=case.budget, rtol=0
    )
