"""One FTF point advances its independent projective histories together."""

import gzip
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from threading import Event
from typing import override

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import AcquireOutcome, IntegratedRound
from zcu_tools.simulate import adc
from zcu_tools.simulate.boundary import CancellationQuery, PulseEnvelope, RoundRequest
from zcu_tools.simulate.evolution import (
    Cancelled,
    CaptureBuffer,
    Engine,
    EvolutionResult,
    ModelSource,
    NoValidKernel,
    PathDiagnostics,
    RoutingHistory,
    StarkOrderInversion,
    TimelineBatch,
    capture_directory,
    read_capture_metadata,
    read_frozen_manifest,
    read_physical_input,
)
from zcu_tools.simulators.ftf.environment import FTFSource, plan_template
from zcu_tools.simulators.ftf.environment.numeric import resonators
from zcu_tools.simulators.ftf.physics import CavityDrive, CavityResponse

from tests.core._simulation import tick_array, ticks
from tests.program._profiles import profile_soccfg
from tests.simulators._engine import EngineDouble
from tests.simulators.ftf.environment._templates import (
    herald_template,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


class HistoryEngine(EngineDouble):
    """Play prescribed transitions and return rows in reverse order."""

    def __init__(self) -> None:
        super().__init__()
        self.interrupt: Exception | None = None
        self.attempted: list[bytes] = []
        self.completed: set[bytes] = set()
        self.reuse_checks: list[bytes] = []

    @override
    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
        capture: CaptureBuffer | None = None,
    ) -> EvolutionResult:
        result = super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
            capture=capture,
        )
        assert reuse_key is not None
        self.attempted.append(reuse_key)
        densities = []
        inversions = []
        for batch in batches:
            assert batch.start_ns is not None
            for row, (start, end) in enumerate(
                zip(batch.start_ns, batch.end_ns, strict=True)
            ):
                # The drive starts two ns later at the second physical point.
                point = int(batch.tracks[0].start_ns[row] == 4)
                initial_ground = result.densities[len(densities), 0, 0].real
                ground = (0.25 if point == 0 else 0.125) + 0.5 * initial_ground
                if point == 1 and start == 0:
                    ground = 0.375  # Prescribed pre-herald preparation.
                densities.append(np.diag([ground, 1 - ground]))
                inversions.append(
                    StarkOrderInversion(
                        (0, 1), int(batch.points[row]), (start, end), 1.0, 2.0
                    )
                )
                if point == 1 and (start, end) == (4, 6) and self.interrupt is not None:
                    # Stop after one history, before returning any batch result.
                    if isinstance(self.interrupt, Cancelled):
                        assert isinstance(cancel, Event)
                        cancel.set()
                        assert cancel.is_set()
                    raise self.interrupt
        self.completed.add(reuse_key)
        paths = PathDiagnostics(
            points_by_class={"fixture": len(result.points)},
            point_members={"fixture": tuple(sorted(result.points))},
            stark_order_inversions=tuple(inversions),
        )
        return EvolutionResult(
            result.points[::-1], np.array(densities, complex)[::-1], paths
        )

    @override
    def reuse(
        self,
        source: ModelSource,
        key: bytes,
        *,
        cancel: CancellationQuery | None = None,
        budget: float = 1e-3,
    ) -> bool:
        self.reuse_checks.append(key)
        return key in self.completed


@pytest.fixture
def projective_case():
    base, readout = herald_template()
    drive = replace(
        template(tones=1).events[0],
        start_ticks=tick_array(np.array([0.002, 0.004])),
    )
    herald = replace(base, points=base.points[::-1], events=(drive, *base.events))
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "chip": numeric.chip.model_copy(update={"temperature_k": 1.0}),
        }
    )
    return numeric, readout, replace(request(templates=(herald,)), reps=128)


@pytest.fixture
def projective_source(projective_case):
    numeric, readout, acquire = projective_case
    engine = HistoryEngine()
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=engine,
    )
    try:
        yield source, engine, acquire
    finally:
        source.close()


@pytest.fixture
def exact_cavity_span_hits(monkeypatch: pytest.MonkeyPatch):
    hits: set[tuple[float, float]] = set()
    solve = adc.CavityAverageCache.span

    def observe_span(owner, model, pulses, start_ns, end_ns, **memory):
        before = owner.hits
        result = solve(owner, model, pulses, start_ns, end_ns, **memory)
        if owner.hits > before:
            hits.add((start_ns, end_ns))
        return result

    # Acquire returns physics results; ADC's public hit counter observes reuse.
    monkeypatch.setattr(adc.CavityAverageCache, "span", observe_span)
    return hits


def _check_history_inputs(
    rows: list[tuple[TimelineBatch, int]],
    expected: NDArray[np.float64],
    memory: NDArray[np.complex128],
    cavity: CavityDrive,
    span: tuple[float, float],
) -> None:
    start, end = span
    for history, ((batch, row), ground) in enumerate(zip(rows, expected, strict=True)):
        assert batch.start_ns is not None and batch.initial_densities is not None
        assert (batch.start_ns[row], batch.end_ns[row]) == span
        populations = np.array([ground, 1 - ground])
        np.testing.assert_allclose(
            batch.initial_densities[row], np.diag(populations), rtol=0, atol=1e-15
        )
        (shift,) = batch.segment_shifts[row]
        assert (shift.start_ns, shift.end_ns) == span
        average, memory[history] = cavity.advance(
            start, end, populations, memory[history]
        )
        np.testing.assert_allclose(
            shift.energy_shift_ghz, average.stark_shift_ghz, atol=1e-15
        )
        np.testing.assert_allclose(
            shift.dephasing_per_ns, average.dephasing_per_ns, atol=1e-15
        )


def test_point_local_batches_continue_histories_and_keep_seeded_draws(
    projective_source, exact_cavity_span_hits
):
    source, engine, acquire = projective_source
    facts = source.model_facts(snapshot())
    plan = plan_template(
        source.settings,
        snapshot(),
        acquire.templates[0],
        soccfg=source.soccfg,
        readout=source.readout,
        facts=facts,
        time_base=acquire.time_base,
    )
    result = source.open_acquire(acquire).sample_round(snapshot(), 0)
    assert isinstance(result, IntegratedRound)

    pair = resonators(source.settings)
    cavities = tuple(
        CavityDrive(facts.joint, pair, reads[0].pulses)
        for reads in plan.projective_reads
    )
    initial = facts.source.initial_density.diagonal().real
    expected_inputs = {
        0: (((0, 2), [1, 0]), ((2, 4), [0.75, 0.25]), ((4, 8), [0.625, 0.375])),
        1: (
            ((0, 2), [initial[0]]),
            ((2, 4), [1, 0]),
            ((4, 6), [0.625, 0.125]),
            ((6, 10), [0.4375, 0.1875]),
        ),
    }
    calls_by_point: dict[int, list[list[tuple[TimelineBatch, int]]]] = {0: [], 1: []}
    for _, batches in engine.calls:
        rows = [(batch, row) for batch in batches for row in range(len(batch.points))]
        # All histories in one call have the same physical drive start.
        (drive_start,) = {batch.tracks[0].start_ns[row] for batch, row in rows}
        calls_by_point[{2: 0, 4: 1}[drive_start]].append(rows)
    for point, spans in expected_inputs.items():
        memory = np.zeros((2, 2), complex)
        for rows, (span, ground) in zip(calls_by_point[point], spans, strict=True):
            assert len({batch.points[row] for batch, row in rows}) == len(ground)
            # Densities distinguish these histories regardless of Engine row order.
            rows.sort(
                key=lambda item: (
                    -item[0].initial_densities[item[1], 0, 0].real
                    if item[0].initial_densities is not None
                    else 0.0
                )
            )
            if point == 1 and span == (2, 4):
                memory[1] = memory[0]
            _check_history_inputs(
                rows, np.array(ground, dtype=float), memory, cavities[point], span
            )
        assert np.any(memory[0] != memory[1])

    # Physical row 1 has caller ID 0; row 0 has caller ID 1.
    paths = source.evaluations[0].paths
    assert paths.point_members == {"fixture": (0, 1)}
    assert {(entry.point, entry.span_ns) for entry in paths.stark_order_inversions} == {
        (1 - point, span)
        for point, spans in expected_inputs.items()
        for span, _ in spans
    }
    terminal_ground = np.array([[11 / 32, 7 / 32], [9 / 16, 7 / 16]])
    expected = adc.ProjectionReadout(
        np.array([[0.375, 0.625], initial]),
        (
            np.tile(np.eye(2), (2, 1, 1)),
            np.stack((terminal_ground, 1 - terminal_ground), axis=-1),
        ),
    )
    stats = adc.round_statistics(
        expected,
        (),
        reps=acquire.reps,
        seed=acquire.root_seed,
        acquire_sequence=acquire.sequence,
        round_index=0,
    )
    np.testing.assert_array_equal(result.statistics.state_counts, stats.state_counts)
    batches = {item.channel: item.batch for item in plan.readouts}
    # ADC plan rows follow template order, while sampling follows caller ID.
    batches = {
        ch: replace(batch, content_index=batch.content_index[::-1])
        for ch, batch in batches.items()
    }
    expected_raw = adc.PreparedRawShots(expected, batches, reps=acquire.reps).bind(
        stats.raw_identity
    )
    actual_raw = result.raw.materialize(acquire.reps, acquire.avg_level)
    for ch, raw in expected_raw.materialize(acquire.reps, acquire.avg_level).items():
        np.testing.assert_array_equal(actual_raw[ch], raw)

    # The per-span diagnostics keep Engine row IDs and their history, even
    # though the aggregate diagnostics and warnings refer to acquisition points.
    solve = source.evaluations[0]
    spans_by_location = {
        (span.point, span.span_ns): span for span in solve.projective_spans
    }
    assert set(spans_by_location) == {
        (1 - point, span)
        for point, spans in expected_inputs.items()
        for span, _ in spans
    }
    for point, spans in expected_inputs.items():
        for rows, (span_ns, _) in zip(calls_by_point[point], spans, strict=True):
            diagnostic = spans_by_location[1 - point, span_ns]
            histories = {
                int(batch.points[row]): () if len(rows) == 1 else (history,)
                for history, (batch, row) in enumerate(rows)
            }
            assert dict(diagnostic.histories) == histories
            assert set(diagnostic.point_members["fixture"]) == histories.keys()
            assert {
                inversion.point for inversion in diagnostic.stark_order_inversions
            } == histories.keys()

    # Each herald begins with two distinct density rows and the same inherited
    # field. Only the cavity solve is shared; the rows and their weights are not.
    assert {(0, 2), (2, 4)} <= exact_cavity_span_hits


def test_cancel_during_projective_assembly_stops_before_next_history(
    projective_source, monkeypatch: pytest.MonkeyPatch
):
    source, _, acquire = projective_source
    cancel = Event()
    advanced_after_cancel: list[tuple[float, float]] = []
    advance = CavityDrive.advance

    def request_cancel_after_first_history(
        drive: CavityDrive,
        start_ns: float,
        end_ns: float,
        populations: NDArray[np.float64],
        initial_field: NDArray[np.complex128],
    ) -> tuple[CavityResponse, NDArray[np.complex128]]:
        span = (start_ns, end_ns)
        if cancel.is_set():
            advanced_after_cancel.append(span)
        response = advance(drive, start_ns, end_ns, populations, initial_field)
        if span == (4, 6):  # This span has two histories at the second point.
            cancel.set()
        return response

    # The acquire exception alone cannot reveal advances before Engine sees Stop.
    monkeypatch.setattr(CavityDrive, "advance", request_cancel_after_first_history)
    with pytest.raises(Cancelled):
        source.open_acquire(acquire).sample_round(snapshot(), 0, cancel=cancel)

    assert cancel.is_set()
    assert advanced_after_cancel == []


@pytest.mark.parametrize(
    "error",
    [Cancelled(), RuntimeError("continuation failed")],
    ids=["cancelled", "failed"],
)
def test_interrupted_batch_has_no_reusable_preparation(projective_source, error):
    source, engine, acquire = projective_source
    # The marked point completes its dependencies before its interrupted batch.
    old_seq = acquire.sequence
    engine.interrupt = error
    cancel = Event()
    with pytest.raises(type(error), match=str(error) or None):
        source.open_acquire(acquire).sample_round(snapshot(), 0, cancel=cancel)
    *earlier, incomplete = engine.attempted
    assert earlier
    assert set(earlier) == engine.completed
    assert incomplete not in engine.completed
    interrupted_calls = len(engine.calls)

    engine.interrupt = None
    later = replace(acquire, sequence=old_seq + 1)
    source.open_acquire(later).sample_round(snapshot(), 0)
    assert {*earlier, incomplete} <= set(engine.attempted[interrupted_calls:])
    assert engine.reuse_checks == []
    assert source.evaluations[0].projection_outcomes == (2, 2)

    # A fully retried preparation can be reused across acquires.
    calls = len(engine.calls)
    source.open_acquire(replace(later, sequence=old_seq + 2)).sample_round(
        snapshot(), 0
    )
    assert len(engine.calls) == calls
    assert set(engine.reuse_checks) == set(engine.attempted[interrupted_calls:])


class UsageHistory(RoutingHistory):
    """Observe the public accounting seam while leaving Engine to price its rows."""

    def __init__(self) -> None:
        super().__init__()
        self.inventories: list[dict[bytes, int]] = []
        self.usage: dict[bytes, int] = {}

    @override
    def record_gate_usage(
        self, inventory: Mapping[bytes, int], *, content: bytes
    ) -> dict[bytes, int]:
        totals = super().record_gate_usage(inventory, content=content)
        self.inventories.append(dict(inventory))
        self.usage.update(totals)
        return totals


def test_equivalent_projective_points_contribute_distinct_routing_usage():
    base, readout = herald_template()
    drive = replace(
        template(tones=1).events[0],
        start_ticks=tick_array(np.full(2, 0.004)),
        envelopes=(PulseEnvelope(ticks(0.001), (0.25, 1.0, 0.25)),),
    )
    equivalent = replace(
        base,
        points=np.array([1, 0], dtype=np.int64),
        events=(
            drive,
            *(
                replace(event, start_ticks=event.start_ticks[[0, 0]])
                for event in base.events
            ),
        ),
        readouts=tuple(
            replace(window, trigger_ticks=window.trigger_ticks[[0, 0]])
            for window in base.readouts
        ),
    )
    history = UsageHistory()
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "numerics": numeric.numerics.model_copy(
                update={"projection_weight_floor": 0.0}
            ),
        }
    )
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=Engine(1, backend="cpu"),
        history=history,
    )
    try:
        # Force a later acquire to call evolve again even for equal content.
        source.preparation_cache.max_bytes = 0
        acquire = request(templates=(equivalent,))
        session = source.open_acquire(acquire)
        session.sample_round(snapshot(), 0)
        expected: Counter[bytes] = Counter()
        for inventory in history.inventories:
            expected.update(inventory)
        assert (
            sum(expected.values()) == 4
        )  # Two herald histories at each of two points.
        assert history.usage == expected

        original = source.evaluations[0]
        first_inventories = len(history.inventories)
        session.sample_round(snapshot(), 1)
        assert len(history.inventories) == first_inventories
        assert source.evaluations[0] is original

        source.open_acquire(replace(acquire, sequence=8)).sample_round(snapshot(), 0)
        assert len(history.inventories) > first_inventories
        assert history.usage == expected  # Replayed content must not grow usage.
        identities = {
            (span.point, span.span_ns, states): row_id
            for span in original.projective_spans
            for row_id, states in span.histories
        }
        assert {point for point, _, _ in identities} == {0, 1}
        assert len(set(identities.values())) == 4  # The two states at each origin.
        assert identities == {
            (span.point, span.span_ns, states): row_id
            for span in source.evaluations[0].projective_spans
            for row_id, states in span.histories
        }
    finally:
        source.close()


class CaptureFailureEngine(EngineDouble):
    """Reject one history of the physically marked second row, when enabled."""

    fail = False

    @override
    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
        capture: CaptureBuffer | None = None,
    ) -> EvolutionResult:
        for batch in batches:
            if not self.fail:
                break
            assert batch.start_ns is not None and batch.initial_densities is not None
            for row, engine_point in enumerate(batch.points):
                if (
                    batch.tracks[0].start_ns[row] == 4
                    and (batch.start_ns[row], batch.end_ns[row]) == (4, 6)
                    and batch.initial_densities[row, 1, 1] == 1
                ):
                    self.error = NoValidKernel(
                        "fixture rejected excited history",
                        domain="plateau",
                        points=(int(engine_point),),
                        segments=(0,),
                        attempted=("dop853",),
                    )
        return super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
            capture=capture,
        )


@pytest.fixture
def captured_projective_source(projective_case, tmp_path, monkeypatch):
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    numeric, readout, acquire = projective_case
    numeric = numeric.model_copy(
        update={"numerics": numeric.numerics.model_copy(update={"debug_capture": True})}
    )
    engine = CaptureFailureEngine()
    with closing(
        FTFSource(
            numeric,
            DeviceManager(),
            soccfg=profile_soccfg("ftf"),
            readout=readout,
            engine=engine,
        )
    ) as source:
        yield source, engine, acquire


@pytest.mark.parametrize("fail_second_round", [False, True], ids=["success", "failure"])
def test_debug_capture_joins_each_projective_batch_to_its_origin(
    captured_projective_source, fail_second_round
):
    source, engine, acquire = captured_projective_source
    first, changed = snapshot(), snapshot(0.3)
    session = source.open_acquire(acquire)
    session.sample_round(first, 0)
    engine.fail = fail_second_round
    if fail_second_round:
        with pytest.raises(NoValidKernel, match=r"point 0.*\[4, 6\].*\(1,\)") as caught:
            session.sample_round(changed, 1)
        assert caught.value.__cause__ is engine.error
        assert caught.value.points == (0,)
        assert source.preparation_cache.resident_bytes == 0
    else:
        session.sample_round(changed, 1)
    source.finish_acquire(
        AcquireOutcome(
            (RoundRequest(acquire, first, 0), RoundRequest(acquire, changed, 1)),
            wall_s=0.0,
            compile_wall_s=0.0,
            complete=not fail_second_round,
            acquire_id="fixture-acquire",
        )
    )

    (path,) = capture_directory().glob("capture-*.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)
    spans = manifest["metadata"]["ftf"]["projective_spans"]
    assert manifest["metadata"]["complete"] is not fail_second_round

    ids = {}
    observed = {0: set(), 1: set()}
    incomplete = set()
    records = read_frozen_manifest(path)
    assert spans.keys() == {
        read_capture_metadata(record)["capture_id"] for record in records
    }
    for record in records:
        metadata = read_capture_metadata(record)
        span = spans[metadata["capture_id"]]
        frozen = read_physical_input(record.physical)
        with closing(frozen.source):
            batch = frozen.batch
            assert batch.start_ns is not None and batch.initial_densities is not None
            for row, engine_point in enumerate(batch.points):
                origin = int(batch.tracks[0].start_ns[row] == 2)
                bounds = (batch.start_ns[row], batch.end_ns[row])
                history = (
                    ()
                    if origin == 0 and bounds == (0, 2)
                    else (int(batch.initial_densities[row, 1, 1] == 1),)
                )
                assert span["point"] == origin
                assert tuple(span["span_ns"]) == bounds
                assert tuple(span["histories"][str(engine_point)]) == history
                assert ids.setdefault((origin, history), engine_point) == engine_point
                position = (origin, bounds, history)
                observed[span["round_index"]].add(position)
                if not metadata["complete"]:
                    incomplete.add((span["round_index"], position))

    assert len(ids) == len(set(ids.values())) == 5
    assert observed[0] == {
        (point, span, (state,))
        for point, point_spans in (
            (0, ((2, 4), (4, 6), (6, 10))),
            (1, ((0, 2), (2, 4), (4, 8))),
        )
        for span in point_spans
        for state in (0, 1)
    } | {(0, (0, 2), ())}
    if fail_second_round:
        assert observed[1] <= observed[0]
        assert incomplete == {(1, (0, (4, 6), state)) for state in ((0,), (1,))}
    else:
        assert observed[1] == observed[0]
        assert not incomplete
