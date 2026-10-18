"""Full Lindblad fallback through the public prepare/apply kernel seam."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import KernelContext, SegmentTable
from zcu_tools.simulate.evolution.workplan import WorkPlan

from tests.simulate.evolution.contract.kernel_suite import (
    KernelCase,
    KernelContract,
    assert_budget,
    kernel_context,
    reference,
    synthetic_model,
)


def envelope(shape: str) -> EnvelopeSpec:
    """Analytic pulse shapes enter the engine as complex sampled envelopes."""
    time = np.linspace(0, 3, 31)
    gaussian = np.exp(-(((time - 1.5) / 0.65) ** 2) / 2)
    shapes = {
        "gaussian": gaussian,
        "drag": gaussian + 0.8j * (time - 1.5) * gaussian,
        "cosine": (1 - np.cos(2 * np.pi * time / 3)) / 2,
        "flat_top": np.minimum(1, np.minimum(time, 3 - time) / 0.6),
        "sampled": np.sin(time * 2.1) + 0.4j * np.cos(time * 3.7),
    }
    if shape == "constant":
        return EnvelopeSpec("constant")
    return EnvelopeSpec("sampled", shapes[shape].astype(complex), time)


@pytest.fixture(
    params=[
        (3, "constant"),
        (3, "gaussian"),
        (4, "drag"),
        (3, "cosine"),
        (3, "flat_top"),
        (3, "sampled"),
    ],
    ids=["constant-3", "gaussian-3", "drag-4", "cosine-3", "flat_top-3", "sampled-3"],
)
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel

    # Cover every envelope and both dimensions, not their Cartesian product.
    size, shape = request.param
    model = synthetic_model(size)

    def build(level: tuple[int, ...]):
        angle = 0.23 * level[0]
        vectors = np.eye(size, dtype=complex)
        vectors[:2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        return replace(
            model,
            level=level,
            identity=f"model-{size}-{level}",
            energies_ghz=model.energies_ghz * (1 + 0.1 * level[0]),
            eigenvectors=vectors,
        )

    vector = np.exp(0.7j * np.arange(size)) / np.sqrt(size)
    state = np.outer(vector, vector.conj())
    source = ModelSource(state, build)
    tones = np.array([0, 1, 2, 4])
    playing = np.arange(4)[None, :] < tones[:, None]
    table = SegmentTable(
        timeline=np.arange(4),
        start_ns=np.array([0.0, 2.0, 7.0, 1.0]),
        duration_ns=np.array([0.0, 1.1, 0.8, 1.6]),
        level=np.array([[0], [1], [0], [1]]),
        tones=tones,
        tone_port=np.where(playing, 0, -1),
        tone_frequency_ghz=np.broadcast_to([0.13, -0.31, 0.67, 1.12], (4, 4)),
        tone_amplitude=np.broadcast_to([0.17, -0.12, 0.09, 0.06], (4, 4)),
        tone_theta_rad=np.broadcast_to([0.5, -0.3, 1.2, 2.1], (4, 4)),
        tone_envelope=np.where(playing, 0, -1),
        tone_envelope_offset_ns=np.broadcast_to([0.4, 0.1, 0.7, 0.0], (4, 4)),
        envelopes=(envelope(shape),),
    )
    return KernelCase(DOP853Kernel(), source, table, np.repeat(state[None], 4, axis=0))


class TestDOP853(KernelContract):
    pass


def prepare(case: KernelCase):
    rows = np.arange(len(case.states), dtype=np.int64)
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(
            case.table, rows, KernelContext(case.source, case.budget, WorkPlan())
        ),
        return_index=True,
        return_inverse=True,
    )
    return case.kernel.prepare(
        keys,
        case.table,
        rows[first],
        KernelContext(case.source, case.budget, WorkPlan()),
    ), inverse


def test_model_keys_exclude_apply_parameters(kernel_case: KernelCase) -> None:
    case = kernel_case
    keys = case.kernel.prepare_keys(
        case.table, np.arange(4), KernelContext(case.source, case.budget, WorkPlan())
    )
    assert keys[0] == keys[2]
    assert keys[1] == keys[3]
    assert keys[0] != keys[1]
    costs = case.kernel.cost(case.table, np.arange(4))
    assert costs.shape == (4,)
    assert np.isfinite(costs).all() and (costs >= 0).all()


def test_chains_stop_at_boundaries_and_can_restart(kernel_case: KernelCase) -> None:
    case = replace(
        kernel_case,
        table=replace(
            kernel_case.table,
            timeline=np.array([5, 5, 5, 5]),
            start_ns=np.array([0.0, 0.0, 1.1, 1.9]),
        ),
    )
    prepared, inverse = prepare(case)
    actual = case.states.copy()
    # Later slots are outputs only, not fresh inputs to the same timeline.
    actual[1:] = 0
    case.kernel.apply(
        prepared,
        inverse,
        case.table,
        np.arange(4),
        actual,
        kernel_context(case.kernel.name, case.table, case.budget),
    )
    expected = []
    state = case.states[0]
    previous = case.source.model((0,))
    for row in range(4):
        model = case.source.model(tuple(case.table.level[row]))
        overlap = model.eigenvectors.conj().T @ previous.eigenvectors
        state = reference(case, row, overlap @ state @ overlap.conj().T)
        expected.append(state)
        previous = model
    assert_budget(actual, np.stack(expected), case.budget)

    split = case.states.copy()
    case.kernel.apply(
        prepared,
        inverse[:2],
        case.table,
        np.arange(2),
        split[:2],
        kernel_context(case.kernel.name, case.table, case.budget),
    )
    # Engine owns basis conversion between calls, not inside one chain.
    left, right = case.source.model((1,)), case.source.model((0,))
    overlap = right.eigenvectors.conj().T @ left.eigenvectors
    split[2] = overlap @ split[1] @ overlap.conj().T
    case.kernel.apply(
        prepared,
        inverse[2:],
        case.table,
        np.arange(2, 4),
        split[2:],
        kernel_context(case.kernel.name, case.table, case.budget),
    )
    assert_budget(actual, split, case.budget)


def test_noncontiguous_rows_of_same_timeline_are_independent(
    kernel_case: KernelCase,
) -> None:
    case = replace(
        kernel_case,
        table=replace(kernel_case.table, timeline=np.zeros(4, dtype=np.int64)),
    )
    prepared, inverse = prepare(case)
    actual = case.states.copy()
    case.kernel.apply(
        prepared,
        inverse,
        case.table,
        np.arange(4),
        actual,
        kernel_context(case.kernel.name, case.table, case.budget),
    )
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(case.states)]
    )
    assert_budget(actual, expected, case.budget)


def test_tight_budget_preserves_complex_density(kernel_case: KernelCase) -> None:
    case = replace(kernel_case, budget=1e-8)
    prepared, inverse = prepare(case)
    actual = case.states.copy()
    case.kernel.apply(
        prepared,
        inverse,
        case.table,
        np.arange(4),
        actual,
        kernel_context(case.kernel.name, case.table, case.budget),
    )
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(case.states)]
    )
    assert_budget(actual, expected, case.budget)
    np.testing.assert_allclose(actual.trace(axis1=1, axis2=2), 1, atol=1e-11)
