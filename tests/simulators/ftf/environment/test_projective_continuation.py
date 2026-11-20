"""One FTF point advances its independent projective histories together."""

from collections.abc import Sequence
from dataclasses import replace
from threading import Event
from typing import override

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate import adc
from zcu_tools.simulate.boundary import CancellationQuery
from zcu_tools.simulate.evolution import (
    Cancelled,
    CaptureBuffer,
    EvolutionResult,
    ModelSource,
    PathDiagnostics,
    StarkOrderInversion,
    TimelineBatch,
)
from zcu_tools.simulators.ftf.environment import FTFSource, plan_template
from zcu_tools.simulators.ftf.environment.numeric import resonators
from zcu_tools.simulators.ftf.physics import CavityDrive

from tests.core._simulation import tick_array
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
                if (
                    len(self.attempted) > 1
                    and len(result.points) > 1
                    and self.interrupt is not None
                ):
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
def projective_source():
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
    engine = HistoryEngine()
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=engine,
    )
    acquire = replace(request(templates=(herald,)), reps=128)
    try:
        yield source, engine, acquire
    finally:
        source.close()


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
    projective_source,
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
    widths = [
        sum(len(batch.points) for batch in batches) for _, batches in engine.calls
    ]
    assert widths == [2, 2, 2, 1, 2, 2, 2], (
        "one call per ready point/span, not per history"
    )

    pair = resonators(source.settings)
    cavities = tuple(
        CavityDrive(facts.joint, pair, reads[0].pulses)
        for reads in plan.projective_reads
    )
    initial = facts.source.initial_density.diagonal().real
    points = (0, 0, 0, 1, 1, 1, 1)
    spans = ((0, 2), (2, 4), (4, 8), (0, 2), (2, 4), (4, 6), (6, 10))
    input_ground = (
        [1, 0],
        [0.75, 0.25],
        [0.625, 0.375],
        [initial[0]],
        [1, 0],
        [0.625, 0.125],
        [0.4375, 0.1875],
    )
    memory = [np.zeros((2, 2), complex), np.zeros((2, 2), complex)]
    for step, ((_, batches), point, span, ground) in enumerate(
        zip(engine.calls, points, spans, input_ground, strict=True)
    ):
        rows = [(batch, row) for batch in batches for row in range(len(batch.points))]
        assert len({batch.points[row] for batch, row in rows}) == len(rows)
        assert all(
            batch.tracks[0].start_ns[row] == 2 + 2 * point for batch, row in rows
        )
        # ID assignment is not the history order contract. Match the histories by their input densities.
        rows.sort(
            key=lambda item: (
                -item[0].initial_densities[item[1], 0, 0].real
                if item[0].initial_densities is not None
                else 0.0
            )
        )
        if step == 4:
            memory[1][1] = memory[1][0]
        _check_history_inputs(
            rows, np.array(ground, dtype=float), memory[point], cavities[point], span
        )
    assert np.any(memory[0][0] != memory[0][1])
    assert np.any(memory[1][0] != memory[1][1])

    # Physical row 1 has caller ID 0; row 0 has caller ID 1.
    paths = source.evaluations[0].paths
    assert paths.point_members == {"fixture": (0, 1)}
    assert {(entry.point, entry.span_ns) for entry in paths.stark_order_inversions} == {
        (1 - point, span) for point, span in zip(points, spans, strict=True)
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


@pytest.mark.parametrize(
    "error",
    [Cancelled(), RuntimeError("continuation failed")],
    ids=["cancelled", "failed"],
)
def test_interrupted_batch_has_no_reusable_preparation(projective_source, error):
    source, engine, acquire = projective_source
    # Complete the first span before interrupting the next two-history batch.
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
    assert engine.attempted[
        interrupted_calls : interrupted_calls + len(earlier) + 1
    ] == [*earlier, incomplete]
    assert engine.reuse_checks == []
    assert source.evaluations[0].projection_outcomes == (2, 2)

    # A fully retried preparation can be reused across acquires.
    calls = len(engine.calls)
    source.open_acquire(replace(later, sequence=old_seq + 2)).sample_round(
        snapshot(), 0
    )
    assert len(engine.calls) == calls
    assert set(engine.reuse_checks) == set(engine.attempted[interrupted_calls:])
