"""Engine behavior through public batches and injected C5 collaborators."""

import warnings
import weakref
from contextlib import closing
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
from zcu_tools.simulate.evolution.kernel import GuardResult, Prepared, SegmentTable

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

    def guard_prepared(self, table, rows, source, budget) -> GuardResult | None:
        """Keep the ordinary guard path for this test collaborator."""
        del table, rows, source, budget

    def bind_input(
        self, prepared, key_index, table, rows, source_initial, budget
    ) -> Prepared | None:
        return None

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
        Engine(
            1,
            1000,
            kernels=(period, RecordingKernel("floquet_period", accepts=False), t6),
        ),
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
            ("floquet_period", False),
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


@pytest.mark.parametrize(
    ("family", "n", "later_pulse"),
    [
        ("t4", 1, False),
        ("t6", 1, False),
        ("t6", 1, True),
        ("gate_cache", 10, False),
        ("period_table", 1, False),
    ],
)
def test_incomplete_terminal_family_fails_instead_of_rerouting(family, n, later_pulse):
    kernels = tuple(
        RecordingKernel(
            name, accepts=name == "dop853" or (name == "dyson" and later_pulse)
        )
        for name in (
            "t4",
            "dyson",
            "rwa",
            "floquet_period",
            "t6",
            "gate_cache",
            "period_table",
            "dop853",
        )
    )
    b = batch(n, duration=6 if later_pulse else 3)
    if family != "t4":
        envelope = (
            EnvelopeSpec("constant")
            if family == "t6"
            else EnvelopeSpec(
                "sampled", np.array([0, 1, 0], complex), np.array([0.0, 1.0, 3.0])
            )
        )
        tracks = (track(n, envelope),)
        if later_pulse:
            tracks += (track(n, envelope, start=3),)
        b = replace(b, tracks=tracks)
    with pytest.raises(RuntimeError, match="no valid kernel for segment"):
        run(Engine(1, 0, kernels=kernels), source(), (b,))


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


def test_gibbs_constant_pulse_at_zero_uses_dyson_for_every_driven_row():
    model = synthetic_model(3)
    model = replace(
        model,
        rates_per_ns=np.zeros((3, 3)),
        dephasing_per_ns=np.zeros((3, 3)),
    )
    weights = np.exp(-model.energies_ghz / 0.2)
    weights /= weights.sum()
    src = ModelSource(
        np.diag(weights).astype(complex), lambda level: replace(model, level=level)
    )
    n = 3
    pulse = replace(
        track(n, EnvelopeSpec("constant"), duration=8),
        frequency_ghz=np.linspace(0.36, 0.44, n),
        amplitude=np.full(n, 0.002),
    )
    b = replace(batch(n, duration=9), tracks=(pulse,))
    result = run(Engine(1, 100000), src, (b,))
    assert result.paths.segments_by_kernel == {"dyson": n, "t4": n}
    assert result.paths.exceeded_by_kernel == {}


def test_preceding_idle_flux_uses_arbitrary_input_dyson_bound():
    model = synthetic_model(3)
    x = np.zeros((3, 3), dtype=complex)
    x[1, 2] = x[2, 1] = 1
    model = replace(
        model,
        energies_ghz=np.zeros(3),
        rates_per_ns=np.zeros((3, 3)),
        dephasing_per_ns=np.zeros((3, 3)),
        drives=(replace(model.drives[0], in_phase_ghz=x, quadrature_ghz=x * 0),),
    )
    basis = np.eye(3, dtype=complex)[[1, 0, 2]]

    def build(level):
        return replace(
            model,
            level=level,
            identity=str(level),
            eigenvectors=np.eye(3, dtype=complex) if level == (0,) else basis,
        )

    src = ModelSource(np.diag([0.999, 0.001, 0.0]).astype(complex), build)
    pulse = replace(
        track(1, EnvelopeSpec("constant"), start=1, duration=1),
        frequency_ghz=np.zeros(1),
        amplitude=np.full(1, 0.003),
    )
    b = replace(
        batch(1, duration=2),
        tracks=(pulse,),
        flux=FluxSchedule(np.array([0, 2]), np.array([0.0, 1.0]), np.array([[1], [0]])),
    )
    result = run(Engine(1, 100000), src, (b,))
    assert result.paths.segments_by_kernel.get("dyson", 0) == 0
    assert result.paths.segments_by_kernel["t4"] == 1
    assert sum(result.paths.segments_by_kernel.values()) == 2
    # The identical drive without a preceding segment can use DC ground input.
    direct = replace(
        batch(1, duration=1), tracks=(replace(pulse, start_ns=np.zeros(1)),)
    )
    direct_result = run(Engine(1, 100000), src, (direct,))
    assert direct_result.paths.segments_by_kernel == {"dyson": 1}


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


@dataclass(frozen=True)
class BoundPayload:
    initial: NDArray[np.complex128]

    @property
    def nbytes(self) -> int:
        return self.initial.nbytes


class InputBindingKernel(RecordingKernel):
    """C5 collaborator observing lifetime and grouping without engine internals."""

    def __init__(self, *, decline: bool = False):
        super().__init__("dyson")
        self.decline = decline
        self.groups: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        self.bound_refs: list[weakref.ReferenceType[BoundPayload]] = []
        self.bound_rows: list[int] = []
        self.unbound_rows: list[int] = []

    def prepare_keys(self, table, rows):
        return np.zeros(len(rows), dtype="V8")

    def bind_input(self, prepared, key_index, table, rows, source_initial, budget):
        assert np.all(table.start_ns[rows] == 0)
        assert len(np.unique(table.tone_port[rows, 0])) == 1
        assert not source_initial.flags.writeable
        self.groups.append((table.tone_amplitude[rows, 0], table.duration_ns[rows]))
        if self.decline:
            return None
        bound = BoundPayload(source_initial)
        self.bound_refs.append(weakref.ref(bound))
        return bound

    def apply(self, prepared, key_index, table, rows, states, budget):
        if isinstance(prepared, BoundPayload):
            assert np.all(key_index == 0)
            self.bound_rows.extend(table.timeline[rows].tolist())
            states[:] = prepared.initial
        else:
            self.unbound_rows.extend(table.timeline[rows].tolist())
        return np.zeros(len(rows), dtype=np.bool_)


@pytest.mark.parametrize("decline", [False, True])
def test_input_bound_payload_grouping_fallback_and_evolve_lifetime(decline):
    n = 8
    k = InputBindingKernel(decline=decline)
    drive = replace(
        track(n, EnvelopeSpec("constant")),
        start_ns=np.array([0.0] * 6 + [1.0, 2.0]),
        duration_ns=(np.arange(n) + 1) / 8,
        amplitude=np.linspace(-0.001, 0.002, n),
    )
    b = replace(batch(n), tracks=(drive,))
    src = source()
    model = src.model((0,))
    other = ModelSource(
        np.diag([0.6, 0.1, 0.3]).astype(complex),
        lambda level: replace(model, level=level),
    )
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    with closing(Engine(2, 100000, kernels=(k, T4()), chunks_per_worker=3)) as engine:
        first = engine.evolve(src, (b,), cancel=Event())
        assert len(k.groups) == 1, (
            "A-stage must bind all same-key source-input rows once"
        )
        np.testing.assert_array_equal(
            np.sort(k.groups[0][0]), np.sort(drive.amplitude[:6])
        )
        np.testing.assert_array_equal(
            np.sort(k.groups[0][1]), np.sort(drive.duration_ns[:6])
        )
        assert all(ref() is None for ref in k.bound_refs), (
            "bound payload must not enter PrepareCache"
        )
        second = engine.evolve(other, (b,), cancel=Event())
        assert len(k.groups) == 2
        assert all(ref() is None for ref in k.bound_refs)
        assert set(k.bound_rows) == (set() if decline else set(range(6)))
        assert set(k.unbound_rows) == (set(range(n)) if decline else {6, 7})
        assert not np.array_equal(first.densities, second.densities)


def test_guard_handoff_only_binds_repeated_keys():
    from zcu_tools.simulate.evolution.kernel import GuardPreparation

    class GuardBindingKernel(InputBindingKernel):
        def prepare_keys(self, table, rows):
            return table.duration_ns[rows].copy().view("V8")

        def guard_prepared(self, table, rows, source, budget):
            keys, first = np.unique(self.prepare_keys(table, rows), return_index=True)
            prepared = self.prepare(keys, table, rows[first], source, budget)
            return GuardResult(
                self.guard(table, rows, source, budget),
                (GuardPreparation(keys, prepared),),
            )

    k = GuardBindingKernel()
    durations = np.array([1.0, 1.0, 2.0])
    drive = replace(track(3, EnvelopeSpec("constant")), duration_ns=durations)
    b = replace(batch(3), tracks=(drive,), end_ns=durations)
    src = source()
    with closing(Engine(1, 0, kernels=(k,))) as engine:
        result = engine.evolve(src, (b,), cancel=Event())
    assert len(k.groups) == 1, "guard-built singletons must bypass bind_input"
    np.testing.assert_array_equal(k.groups[0][1], [1.0, 1.0])
    assert k.bound_rows == [0, 1]
    assert k.unbound_rows == [2]
    assert k.preparations == 2
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], 3, axis=0)
    )


def test_shared_dyson_is_bitwise_chunk_independent_and_source_local():
    from zcu_tools.simulate.evolution.kernels.dyson import DysonKernel
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    n = 10
    drive = replace(
        track(n, EnvelopeSpec("constant")),
        duration_ns=(np.arange(n) + 1) / 16,
        frequency_ghz=np.linspace(0.11, 0.2, n),
        amplitude=np.linspace(-0.0002, 0.0002, n),
    )
    b = replace(batch(n), tracks=(drive,))
    src = source()
    model = src.model((0,))
    vector = np.exp(1j * np.arange(3)) / np.sqrt(3)
    other = ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )
    assert callable(getattr(DysonKernel(), "bind_input", None)), (
        "Dyson must share source-input paths"
    )
    outputs = []
    for workers, chunks in ((1, 1), (1, 3), (2, 1), (2, 4)):
        with closing(
            Engine(
                workers, 100000, kernels=(DysonKernel(), T4()), chunks_per_worker=chunks
            )
        ) as engine:
            result = engine.evolve(src, (b,), cancel=Event())
            changed = engine.evolve(other, (b,), cancel=Event())
        assert result.paths.segments_by_kernel["dyson"] == n
        outputs.append((result.densities, changed.densities))
    for original, changed in outputs[1:]:
        np.testing.assert_array_equal(original, outputs[0][0])
        np.testing.assert_array_equal(changed, outputs[0][1])
    assert np.max(abs(outputs[0][0] - outputs[0][1])) > 0.01
