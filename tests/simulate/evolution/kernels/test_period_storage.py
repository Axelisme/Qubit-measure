"""Whole-payload period storage charges and shaped resident capacity."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
)
from zcu_tools.simulate.evolution.cache import (
    RESIDENT_METADATA_BYTES,
    PrepareCache,
    PreparedPayload,
)
from zcu_tools.simulate.evolution.kernel import KernelContext, SegmentTable
from zcu_tools.simulate.evolution.kernels._period_table import (
    PeriodEntry,
    PeriodPrepared,
)
from zcu_tools.simulate.evolution.kernels.dop853 import prepare_models
from zcu_tools.simulate.evolution.kernels.shaped_sambe import ShapedSambeKernel
from zcu_tools.simulate.evolution.workplan import WorkPlan


def model() -> SecularModel:
    x = np.array([[0, 0.012], [0.012, 0]], dtype=complex)
    return SecularModel(
        (0,),
        "storage",
        np.array([0.0, 0.294]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )


@pytest.mark.parametrize("view", [False, True], ids=["shared", "retained-base"])
def test_period_payload_charges_each_retained_array_allocation_once(view: bool) -> None:
    coefficients = np.zeros((1, 8, 16, 2, 2), dtype=complex)
    populations = np.eye(2)
    coherences = np.ones((2, 2))
    fallback = prepare_models([model()])
    entry = PeriodEntry(
        fallback,
        1e-8,
        0.294,
        0.3,
        coefficients[:, :4] if view else coefficients,
        populations,
        coherences,
    )
    prepared = PeriodPrepared((entry, entry))
    expected = (
        coefficients.nbytes + populations.nbytes + coherences.nbytes + fallback.nbytes
    )
    assert prepared.nbytes == expected, (
        "charge retained owners, not entries or view extents"
    )
    # Retention includes one metadata charge; bytes still counts only array owners.
    cache = PrepareCache(expected + RESIDENT_METADATA_BYTES)
    payload = PreparedPayload(prepared, (b"rise", b"fall"))
    cache.add(("shaped_sambe", "storage", 1e-3), payload)
    assert cache.bytes == expected
    assert cache.find(("shaped_sambe", "storage", 1e-3), {b"fall"}) == [payload]


def test_shaped_guard_payload_fits_compact_resident_budget() -> None:
    # Rise/fall have distinct row keys but share one physical polynomial.
    table = SegmentTable(
        np.arange(2),
        np.zeros(2),
        np.array([100.0, 150.0]),
        np.zeros((2, 1), dtype=np.int64),
        np.ones(2, dtype=np.int64),
        np.zeros((2, 1), dtype=np.int64),
        np.full((2, 1), 0.294),
        np.full((2, 1), 0.3),
        np.zeros((2, 1)),
        np.arange(2)[:, None],
        np.zeros((2, 1)),
        tuple(
            EnvelopeSpec(
                "sampled", np.array(values, dtype=complex), np.array([0.0, duration])
            )
            for values, duration in [([0, 1], 100), ([1, 0], 150)]
        ),
    )
    source = ModelSource(np.diag([1.0, 0.0]).astype(complex), lambda _: model())
    kernel = ShapedSambeKernel()
    plan = WorkPlan()
    try:
        context = KernelContext(source, 1e-3, plan)
        result = kernel.guard_prepared(table, np.arange(2), context)
        assert result.accepted.all()
        handoff = result.preparations[0]
        # One dense polynomial alone is 8192 bytes. The complete compact
        # two-key payload (including fallback models/maps/keys) fits in 4 KiB.
        assert handoff.prepared.nbytes <= 4096, "resident storage must omit zero planes"
        # Keep the 4 KiB compact-array allowance plus one allocation's metadata.
        cache = PrepareCache(4096 + RESIDENT_METADATA_BYTES)
        keys = tuple(key.tobytes() for key in handoff.keys)
        payload = PreparedPayload(handoff.prepared, keys)
        cache.add((kernel.name, "storage", 1e-3), payload)
        assert cache.find((kernel.name, "storage", 1e-3), set(keys)) == [payload]
    finally:
        plan.close()
