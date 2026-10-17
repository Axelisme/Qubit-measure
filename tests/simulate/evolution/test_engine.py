"""Engine behavior through public batches and injected C5 collaborators."""

import warnings
from dataclasses import dataclass, replace
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import (
    Cancelled,
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import Prepared, SegmentTable

from .contract.kernel_suite import synthetic_model


def source() -> ModelSource:
    model = synthetic_model()
    return ModelSource(
        np.diag([0.2, 0.3, 0.5]).astype(complex),
        lambda level: replace(model, level=level),
    )


def batch(n: int = 12, *, duration: float = 3.0) -> TimelineBatch:
    return TimelineBatch(
        np.arange(n),
        (),
        FluxSchedule(np.arange(n + 1), np.zeros(n), np.zeros((n, 1), dtype=np.int64)),
        np.full(n, duration),
    )


def track(
    n: int, envelope: EnvelopeSpec, *, start: float = 0, duration: float = 3
) -> DriveTrack:
    return DriveTrack(
        "drive",
        np.full(n, start),
        np.full(n, duration),
        np.full(n, 0.1),
        np.zeros(n),
        np.ones(n),
        (envelope,),
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
    )


@dataclass
class Payload:
    count: int

    @property
    def nbytes(self) -> int:
        return self.count * 16


class RecordingKernel:
    """Identity dynamics with public observations at the kernel boundary."""

    def __init__(self, name: str, *, accepts: bool = True, cancel: Event | None = None):
        self.name = name
        self.accepts = accepts
        self.cancel = cancel
        self.preparations = 0
        self.calls: list[
            tuple[SegmentTable, NDArray[np.int64], NDArray[np.complex128]]
        ] = []

    def guard(self, table, rows, source, budget):
        return np.full(len(rows), self.accepts)

    def prepare_keys(self, table, rows):
        keys = np.empty(
            len(rows), dtype=[("level", "i8", table.level.shape[1]), ("duration", "f8")]
        )
        keys["level"] = table.level[rows]
        keys["duration"] = table.duration_ns[rows]
        return keys

    def cost(self, table, rows):
        return np.ones(len(rows))

    def prepare(self, keys, table, rows, source, budget):
        self.preparations += len(keys)
        return Payload(len(keys))

    def apply(self, prepared: Prepared, key_index, table, rows, states, budget):
        self.calls.append((table, rows.copy(), states.copy()))
        if self.cancel is not None:
            self.cancel.set()
        return np.zeros(len(rows), dtype=np.bool_)


def run(engine: Engine, src: ModelSource, batches: tuple[TimelineBatch, ...]):
    try:
        return engine.evolve(src, batches, cancel=Event())
    finally:
        engine.close()


def test_idle_reference_and_worker_chunk_batch_independence():
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    from ._tables import idle_table
    from .contract.kernel_suite import KernelCase, reference

    src = source()
    b = batch()
    expected = reference(
        KernelCase(T4(), src, idle_table(np.array([3.0])), src.initial_density[None]),
        0,
        src.initial_density,
    )
    a = run(Engine(1, 100000, chunks_per_worker=1), src, (b,))
    subsets = tuple(replace(batch(4), points=np.arange(i, i + 4)) for i in (0, 4, 8))
    c = run(Engine(2, 0, chunks_per_worker=7), src, subsets)
    np.testing.assert_allclose(
        a.densities, np.repeat(expected[None], 12, axis=0), atol=1e-11
    )
    np.testing.assert_allclose(c.densities, a.densities, atol=1e-13)
    assert a.paths.segments_by_kernel == {"t4": 12}
    assert a.paths.points_by_class == {"t4": 12}


def test_flux_basis_changes_and_readout_truncation():
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    zero = np.zeros((2, 2))
    swap = np.array([[0, 1], [1, 0]], dtype=complex)

    def build(level):
        return SecularModel(
            level,
            str(level),
            np.zeros(2),
            np.eye(2, dtype=complex) if level == (0,) else swap,
            (),
            zero,
            zero,
        )

    src = ModelSource(np.diag([1.0, 0.0]).astype(complex), build)
    b = replace(
        batch(1),
        flux=FluxSchedule(
            np.array([0, 3]), np.array([0.0, 1.0, 100.0]), np.array([[0], [1], [99]])
        ),
        tracks=(replace(track(1, EnvelopeSpec("constant"), start=20), port="absent"),),
    )
    result = run(Engine(1, 1000, kernels=(T4(),)), src, (b,))
    np.testing.assert_allclose(result.densities[0], src.initial_density, atol=1e-14)
    assert result.paths.segments_by_kernel == {"t4": 2}


@pytest.mark.parametrize(("n", "required"), [(9, "period_table"), (10, "gate_cache")])
def test_shaped_dedup_threshold(n, required):
    shaped = EnvelopeSpec(
        "sampled", np.array([0.0, 1.0, 0.0], complex), np.array([0.0, 1.0, 3.0])
    )
    k = RecordingKernel(required)
    result = run(
        Engine(2, 1000, kernels=(k,)),
        source(),
        (replace(batch(n), tracks=(track(n, shaped),)),),
    )
    assert result.paths.segments_by_kernel == {required: n}
    assert k.preparations == 1


def test_flat_top_split_phase_offset_and_small_plateau():
    envelope = EnvelopeSpec(
        "sampled", np.array([0, 1j, 1j, 0], complex), np.array([0.0, 2.0, 12.0, 14.0])
    )
    period, t6 = RecordingKernel("period_table"), RecordingKernel("t6")
    result = run(
        Engine(1, 1000, kernels=(period, t6)),
        source(),
        (replace(batch(1, duration=14), tracks=(track(1, envelope, duration=14),)),),
    )
    assert result.paths.segments_by_kernel == {"period_table": 2, "t6": 1}
    table, rows, _ = t6.calls[0]
    assert table.start_ns[rows].tolist() == [2.0]
    assert table.duration_ns[rows].tolist() == [10.0]
    np.testing.assert_allclose(
        table.tone_theta_rad[rows, 0], 2 * np.pi * 0.1 * 2 + np.pi / 2
    )
    assert table.envelopes[table.tone_envelope[rows[0], 0]].kind == "constant"
    assert period.calls[-1][0].tone_envelope_offset_ns[
        period.calls[-1][1], 0
    ].tolist() == [12.0]
    short = replace(envelope, sample_ns=np.array([0.0, 2.0, 11.0, 14.0]))
    k = RecordingKernel("period_table")
    result = run(
        Engine(1, 0, kernels=(k,)),
        source(),
        (replace(batch(1, duration=14), tracks=(track(1, short, duration=14),)),),
    )
    assert result.paths.segments_by_kernel == {"period_table": 1}


@pytest.mark.parametrize(
    ("accept_dyson", "accept_rwa", "expected"),
    [(True, True, "dyson"), (False, True, "rwa"), (False, False, "t6")],
)
def test_constant_decision_order_and_sequence_middle(
    accept_dyson, accept_rwa, expected
):
    kernels = tuple(
        RecordingKernel(name, accepts=accepts)
        for name, accepts in [
            ("dyson", accept_dyson),
            ("rwa", accept_rwa),
            ("t6", True),
        ]
    )
    b = replace(
        batch(2, duration=6),
        tracks=(
            track(2, EnvelopeSpec("constant")),
            track(2, EnvelopeSpec("constant"), start=3),
        ),
    )
    result = run(Engine(1, 0, kernels=kernels), source(), (b,))
    counts = {expected: 2}
    counts["t6"] = counts.get("t6", 0) + 2
    assert result.paths.segments_by_kernel == counts


def test_missing_kernel_names_segment_and_flux_overlap():
    b = replace(batch(1), tracks=(track(1, EnvelopeSpec("constant")),))
    engine = Engine(1, 0, kernels=())
    try:
        with pytest.raises(RuntimeError, match=r"(?s)(segment.*dyson|dyson.*segment)"):
            engine.evolve(source(), (b,), cancel=Event())
        b = replace(
            b,
            flux=FluxSchedule(
                np.array([0, 2]), np.array([0.0, 1.0]), np.array([[0], [1]])
            ),
        )
        with pytest.raises(
            RuntimeError, match=r"(?s)(segment.*flux_dop853|flux_dop853.*segment)"
        ):
            engine.evolve(source(), (b,), cancel=Event())
    finally:
        engine.close()


def test_shared_cache_namespace_budget_eviction_and_singletons():
    k = RecordingKernel("t4")
    engine = Engine(1, 16, kernels=(k,))
    src = source()
    try:
        for duration in [3.0, 3.0, 4.0, 3.0]:
            engine.evolve(src, (batch(2, duration=duration),), cancel=Event())
        assert k.preparations == 3
        engine.evolve(src, (batch(2),), budget=1e-6, cancel=Event())
        assert k.preparations == 4
        other = ModelSource(
            src.initial_density,
            lambda level: replace(src.model(level), identity="different-source"),
        )
        engine.evolve(other, (batch(2),), cancel=Event())
        assert k.preparations == 5
        for _ in range(2):
            engine.evolve(src, (batch(1, duration=8),), cancel=Event())
        assert k.preparations == 7
    finally:
        engine.close()
        engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.evolve(src, (batch(),), cancel=Event())


def test_cancel_between_chunks_and_nonfinite_failure():
    cancel = Event()
    k = RecordingKernel("t4", cancel=cancel)
    engine = Engine(1, 1000, kernels=(k,), chunks_per_worker=4)
    try:
        with pytest.raises(Cancelled):
            engine.evolve(source(), (batch(),), cancel=cancel)
        assert len(k.calls) == 1
    finally:
        engine.close()

    class Broken(RecordingKernel):
        def apply(self, prepared, key_index, table, rows, states, budget):
            states[:] = np.nan
            return np.zeros(len(rows), dtype=np.bool_)

    engine = Engine(1, 0, kernels=(Broken("t4"),))
    try:
        with pytest.raises(RuntimeError, match=r"(?s)(point.*segment|segment.*point)"):
            engine.evolve(source(), (batch(),), cancel=Event())
    finally:
        engine.close()


def test_dop853_adjacent_rows_are_one_chain():
    k = RecordingKernel("dop853")
    # Three overlapping tones, with one replaced at t=1.
    tracks = tuple(track(1, EnvelopeSpec("constant"), duration=3) for _ in range(2)) + (
        track(1, EnvelopeSpec("constant"), duration=1),
        track(1, EnvelopeSpec("constant"), start=1, duration=2),
    )
    run(Engine(1, 0, kernels=(k,)), source(), (replace(batch(1), tracks=tracks),))
    assert len(k.calls) == 1
    table, rows, _ = k.calls[0]
    assert table.start_ns[rows].tolist() == [0.0, 1.0]
    assert table.timeline[rows].tolist() == [0, 0]


@pytest.mark.parametrize("chunks", [1, 4])
def test_exceeded_counts_warn_once_per_evolve_and_keep_results(chunks: int) -> None:
    class ReportingKernel(RecordingKernel):
        report = True

        def apply(self, prepared, key_index, table, rows, states, budget):
            super().apply(prepared, key_index, table, rows, states, budget)
            return (table.timeline[rows] % 2 == 0) & self.report

    kernels = (ReportingKernel("dyson"), ReportingKernel("t4"))
    engine = Engine(1, 1000, kernels=kernels, chunks_per_worker=chunks)
    src = source()
    b = replace(batch(6, duration=6), tracks=(track(6, EnvelopeSpec("constant")),))
    try:
        for _ in range(2):
            with pytest.warns(RuntimeWarning) as caught:
                result = engine.evolve(src, (b,), cancel=Event())
            assert len(caught) == 1
            message = str(caught[0].message)
            assert "6 segments" in message
            assert "dyson: 3" in message
            assert "t4: 3" in message
            assert "results use the uncorrected approximation" in message
            assert result.paths.exceeded_by_kernel == {"dyson": 3, "t4": 3}
            np.testing.assert_array_equal(
                result.densities, np.repeat(src.initial_density[None], 6, axis=0)
            )
        for kernel in kernels:
            kernel.report = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = engine.evolve(src, (b,), cancel=Event())
        assert not caught
        assert result.paths.exceeded_by_kernel == {}
    finally:
        engine.close()
