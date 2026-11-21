"""FTF acquire chooses a dimension from a complete pilot density before ADC."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, override

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import AcquireOutcome
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    CancellationQuery,
    EnvironmentSnapshot,
    RoundRequest,
    RoundStatistics,
)
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    Engine,
    EvolutionResult,
    ModelSource,
    PathDiagnostics,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulators.ftf.environment import FTFSource
from zcu_tools.simulators.ftf.environment import template as template_module
from zcu_tools.simulators.telemetry import RoutingRecord, RoutingTelemetry

from tests.program._profiles import profile_soccfg
from tests.simulators._engine import EngineDouble
from tests.simulators.ftf.environment._dimension_cases import dimension_bypass
from tests.simulators.ftf.environment._dimension_cases import (
    dimension_request as _request,
)
from tests.simulators.ftf.environment._dimension_cases import (
    dimension_settings as _settings,
)
from tests.simulators.ftf.environment._templates import expect_statistics, snapshot

pytestmark = pytest.mark.sim


class PilotEngine(EngineDouble):
    """Physical density answers make the FTF selection observable at the ADC."""

    def __init__(
        self, *, extent: int = 12, bad_phase: bool = False, workers: int = 1
    ) -> None:
        super().__init__(workers, backend="cpu")
        self.solve_shapes: list[tuple[int, tuple[int, ...]]] = []
        self.extent = extent
        self.bad_phase = bad_phase
        self.round_models: list[tuple[SecularModel, NDArray[np.complex128]]] = []

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
        points = np.concatenate([batch.points for batch in batches])
        n = len(source.initial_density)
        self.solve_shapes.append((n, tuple(int(p) for p in points)))
        self.round_models.append((source.model((0, 0, 0)), source.initial_density))
        if len(points) == 1 and n > 12:
            # Population at index 12 is tiny; its coherence with the ground
            # state requires retaining that row AND column despite the tiny population.
            vector = np.zeros(n, dtype=np.complex128)
            vector[0] = np.sqrt(1 - 1e-6)
            vector[self.extent] = 1e-3j
            if self.bad_phase:
                # Opposite phases with equal populations defeat a diagonal-only check.
                vector[0] = np.sqrt(0.5 - 1e-6)
                vector[1] = np.sqrt(0.5) * (1 if n == 16 else -1)
            rho = np.outer(vector, vector.conj())
        else:
            rho = np.zeros((n, n), dtype=np.complex128)
            rho[min(12, n - 1), min(12, n - 1)] = 1
        if len(points) == 1 and n == 13 and not self.bad_phase:
            # Keep the verification within budget but give it a detectable
            # population that the real upper reference lacks.
            rho *= 1 - 4e-4
            rho[1, 1] += 4e-4
        return EvolutionResult(
            points, np.repeat(rho[None], len(points), axis=0), PathDiagnostics()
        )


def _dimension_extras(logs: Path) -> dict[str, object]:
    (summary,) = logs.glob("routing-*.jsonl")
    return json.loads(summary.read_text())["extras"]


def _dimension_source(engine: Engine, logs: Path, *, upper: int = 16) -> FTFSource:
    return FTFSource(
        _settings(upper),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=engine,
        telemetry=RoutingTelemetry(logs),
    )


def _sample(
    engine: Engine, acquire: AcquireRequest, upper: int, logs: Path
) -> tuple[RoundStatistics, dict[str, object]]:
    source = _dimension_source(engine, logs, upper=upper)
    try:
        result = _execute_acquire(source, acquire)
    finally:
        source.close()

    return result, _dimension_extras(logs)


def _execute_acquire(
    source: FTFSource, acquire: AcquireRequest, state: EnvironmentSnapshot | None = None
) -> RoundStatistics:
    round_request = RoundRequest(acquire, snapshot() if state is None else state, 0)
    result = expect_statistics(source.execute(round_request))
    source.finish_acquire(AcquireOutcome((round_request,), 1.0, 0.0, True))
    return result


def _acquire_rows(logs: Path) -> list[RoutingRecord]:
    (summary,) = logs.glob("routing-*.jsonl")
    return [json.loads(line) for line in summary.read_text().splitlines()]


def test_equal_acquire_reuses_pilot_preparation_with_fresh_sampling(
    tmp_path: Path,
) -> None:
    engine = PilotEngine()
    source = _dimension_source(engine, tmp_path)
    reference = FTFSource(
        _settings(16),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=PilotEngine(),
    )
    try:
        acquire = _request()
        first = _execute_acquire(source, acquire)
        assert engine.solve_shapes == [(16, (7,)), (13, (7,)), (13, tuple(range(7)))]
        hits = source.preparation_cache.hits
        repeat = replace(acquire, sequence=acquire.sequence + 1)
        engine.solve_shapes.clear()
        second = _execute_acquire(source, repeat)
        assert source.preparation_cache.hits == hits + 1
        assert not engine.solve_shapes
        expected = _execute_acquire(reference, repeat)

        # A cache hit is not permission to bypass the Engine's reprice decision.
        engine.reuse_allowed = False
        refreshed = replace(repeat, sequence=repeat.sequence + 1)
        _execute_acquire(source, refreshed)
        assert engine.solve_shapes == [(13, tuple(range(8)))]
        engine.reuse_allowed = True
        engine.solve_shapes.clear()
        _execute_acquire(source, replace(refreshed, sequence=refreshed.sequence + 1))
        assert not engine.solve_shapes
        assert source.preparation_cache.hits == hits + 3
    finally:
        source.close()
        reference.close()
    original, reused, repriced, reused_full = _acquire_rows(tmp_path)
    original_extras = original["extras"]
    pilot_wall = original_extras["ftf.pilot_wall_s"]
    assert isinstance(pilot_wall, float) and pilot_wall > 0.0
    dimension = original_extras["ftf.selected_dimension"]
    assert isinstance(dimension, int) and dimension < 16
    assert original_extras["ftf.dimension_reused"] is False
    for row in (reused, repriced, reused_full):
        extras = row["extras"]
        assert extras["ftf.selected_dimension"] == dimension
        assert extras["ftf.dimension_reused"] is True
        assert extras["ftf.pilot_point"] is None
        assert extras["ftf.pilot_wall_s"] == 0.0
    assert reused["evolve_wall_s"] == reused_full["evolve_wall_s"] == 0.0
    assert repriced["evolve_wall_s"] > 0.0
    np.testing.assert_array_equal(second.state_counts, expected.state_counts)
    np.testing.assert_array_equal(second.channels[0].sums, expected.channels[0].sums)
    assert not np.array_equal(first.channels[0].sums, second.channels[0].sums)


def test_evicted_pilot_preparation_evolves_the_full_selected_batch(
    tmp_path: Path,
) -> None:
    engine = PilotEngine()
    source = _dimension_source(engine, tmp_path)
    acquire = _request()
    try:
        _execute_acquire(source, acquire)
        assert source.preparation_cache.resident_bytes > 0
        source.preparation_cache.max_bytes = 0
        engine.solve_shapes.clear()
        repeat = replace(acquire, sequence=acquire.sequence + 1)
        second = _execute_acquire(source, repeat)
        assert source.preparation_cache.resident_bytes == 0
    finally:
        source.close()

    assert engine.solve_shapes == [(13, tuple(range(8)))]
    np.testing.assert_array_equal(second.state_counts[:, 12], acquire.reps)
    _, row = _acquire_rows(tmp_path)
    extras = row["extras"]
    assert extras["ftf.selected_dimension"] == 13
    assert extras["ftf.dimension_reused"] is True
    assert extras["ftf.pilot_point"] is None
    assert extras["ftf.pilot_wall_s"] == 0.0
    assert row["evolve_wall_s"] > 0.0


@pytest.mark.parametrize(
    "change", ["settings", "snapshot", "revision", "engine", "failure"]
)
def test_remembered_dimension_does_not_cross_preparation_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    engine = PilotEngine()
    source = _dimension_source(engine, tmp_path)
    acquire, state = _request(), snapshot()
    try:
        _execute_acquire(source, acquire, state)
        if change == "settings":
            chip = source.settings.chip.model_copy(update={"temperature_k": 0.3})
            source.settings = source.settings.model_copy(update={"chip": chip})
        elif change == "snapshot":
            state = snapshot(0.8)
        elif change == "revision":
            # The author API has no revision setter. Change the numeric version
            # supplied to the facts owner without inspecting its private cache.
            monkeypatch.setattr(
                template_module, "SEGMENT_NUMERIC_VERSION", "changed-numerics"
            )
        elif change == "engine":
            engine.close()
            engine = PilotEngine()
            source.engine = engine
        else:
            first, *others = acquire.templates
            window = first.readouts[0]
            early = replace(window, trigger_ticks=window.trigger_ticks - 1)
            invalid_first = replace(first, readouts=(early,))
            invalid = replace(acquire, templates=(invalid_first, *others))
            with pytest.raises(ValueError, match="trigger must not precede"):
                _execute_acquire(source, invalid, state)
        engine.solve_shapes.clear()
        _execute_acquire(source, replace(acquire, sequence=acquire.sequence + 1), state)
    finally:
        source.close()

    assert engine.solve_shapes == [(16, (7,)), (13, (7,)), (13, tuple(range(7)))]
    assert _acquire_rows(tmp_path)[-1]["extras"]["ftf.dimension_reused"] is False


def test_pilot_keeps_a_state_with_small_population_but_large_coherence(
    tmp_path: Path,
) -> None:
    engine = PilotEngine()
    acquire = replace(_request(), reps=32768, loop_dims=(32768, 8))
    result, extras = _sample(engine, acquire, 16, tmp_path)
    assert engine.solve_shapes == [(16, (7,)), (13, (7,)), (13, tuple(range(7)))]
    assert result.state_counts.shape == (8, 16)
    np.testing.assert_array_equal(result.state_counts[:7, 12], acquire.reps)
    np.testing.assert_array_equal(result.state_counts[:, 13:], 0)
    # Recording the k-dimensional verification would populate state 1 on
    # this seed; using the upper reference cannot.
    assert result.state_counts[7, 1] == 0
    assert result.state_counts[7, 0] > 32000
    assert extras["ftf.selected_dimension"] == 13
    assert extras["ftf.pilot_point"] == 7
    assert extras["ftf.pilot_drop"] == 0.0
    assert extras["ftf.pilot_error"] == pytest.approx(4e-4)
    duration = extras["ftf.pilot_wall_s"]
    assert isinstance(duration, float) and duration > 0.0


@pytest.mark.parametrize("eligible", [True, False])
def test_upper_selection_reuses_the_reference_row_without_another_attempt(
    tmp_path: Path, eligible: bool
) -> None:
    engine = PilotEngine(extent=12 if eligible else 15, bad_phase=eligible)
    acquire = _request()
    result, extras = _sample(engine, acquire, 16, tmp_path)
    expected: list[tuple[int, tuple[int, ...]]] = [(16, (7,))]
    if eligible:
        expected.append((13, (7,)))
    expected.append((16, tuple(range(7))))
    assert engine.solve_shapes == expected
    np.testing.assert_array_equal(result.state_counts[:7, 12], acquire.reps)
    assert result.state_counts[7, 12] == 0
    assert extras["ftf.selected_dimension"] == 16
    assert extras["ftf.pilot_point"] == 7
    assert extras["ftf.pilot_drop"] == 0.0
    assert extras["ftf.pilot_verify_price_skipped"] is False
    if eligible:
        assert extras["ftf.pilot_error"] == pytest.approx(1.0, abs=2e-6)
        assert extras["ftf.pilot_candidate_dimension"] == 13
    else:
        assert extras["ftf.pilot_error"] is None
        assert extras["ftf.pilot_candidate_dimension"] is None
        assert extras["ftf.pilot_estimated_verify_s"] is None


@pytest.mark.parametrize(("workers", "waves"), [(7, 1), (4, 2)])
def test_price_not_amortized_skips_verification_and_keeps_upper_result(
    tmp_path: Path,
    workers: int,
    waves: int,
) -> None:
    engine = PilotEngine(extent=11, workers=workers)
    acquire = _request()
    result, extras = _sample(engine, acquire, 18, tmp_path)

    assert engine.solve_shapes == [(18, (7,)), (18, tuple(range(7)))]
    assert result.state_counts.shape == (8, 18)
    np.testing.assert_array_equal(result.state_counts[:7, 12], acquire.reps)
    assert result.state_counts[7, 12] == 0
    assert result.state_counts[7, 0] == acquire.reps
    assert extras["ftf.selected_dimension"] == 18
    assert extras["ftf.pilot_point"] == 7
    assert extras["ftf.pilot_drop"] == 0.0
    assert extras["ftf.pilot_error"] is None
    assert extras["ftf.pilot_verify_price_skipped"] is True
    assert extras["ftf.pilot_candidate_dimension"] == 12
    assert extras["ftf.pilot_candidate_drop"] == 0.0
    assert extras["ftf.pilot_upper_dimension"] == 18
    assert extras["ftf.pilot_timelines"] == 8
    assert extras["ftf.pilot_workers"] == workers
    assert extras["ftf.pilot_remaining_waves"] == waves
    upper_s = extras["ftf.pilot_upper_wall_s"]
    verify_s = extras["ftf.pilot_estimated_verify_s"]
    savings_s = extras["ftf.pilot_estimated_savings_s"]
    assert isinstance(upper_s, float) and upper_s > 0.0
    assert isinstance(verify_s, float) and isinstance(savings_s, float)
    assert verify_s == pytest.approx(upper_s * (2 / 3), abs=0)
    assert savings_s == pytest.approx(upper_s * (waves / 3), abs=0)
    if waves == 1:
        assert savings_s < verify_s
    else:
        assert savings_s == verify_s


def test_equal_dc_replaces_the_split_preparation_before_reusing_full_batch() -> None:
    engine = PilotEngine(extent=15)
    acquire = _request()
    source = FTFSource(
        _settings(16), DeviceManager(), soccfg=profile_soccfg("ftf"), engine=engine
    )
    try:
        state = snapshot()
        session = source.open_acquire(acquire)
        first = expect_statistics(session.sample_round(state, 0))
        second = expect_statistics(session.sample_round(state, 1))
        third = expect_statistics(session.sample_round(state, 2))
    finally:
        source.close()

    assert first.state_counts[7, 12] == 0
    assert second.state_counts[7, 12] == third.state_counts[7, 12] == acquire.reps
    assert engine.solve_shapes == [
        (16, (7,)),
        (16, tuple(range(7))),
        (16, tuple(range(8))),
    ]


@pytest.mark.parametrize(("upper", "points"), [(16, 7), (12, 8)])
def test_small_acquire_uses_upper_without_a_pilot(
    tmp_path: Path, upper: int, points: int
) -> None:
    engine = PilotEngine()
    acquire = _request(points)
    result, extras = _sample(engine, acquire, upper, tmp_path)
    assert engine.solve_shapes == [(upper, tuple(range(points)))]
    np.testing.assert_array_equal(
        result.state_counts[:, min(12, upper - 1)], acquire.reps
    )
    assert result.state_counts.shape == (points, upper)
    assert extras["ftf.selected_dimension"] == upper
    assert extras["ftf.pilot_point"] is None
    assert extras["ftf.pilot_wall_s"] == 0.0
    assert extras["ftf.pilot_drop"] is None
    assert extras["ftf.pilot_error"] is None


@pytest.mark.parametrize("kind", ["herald", "early_cavity"])
def test_readout_continuation_uses_upper_across_whole_acquire(
    kind: Literal["herald", "early_cavity"],
) -> None:
    acquire, readout = dimension_bypass(kind)
    terminal_ends_ns = {
        end
        for plan in acquire.templates
        for end in acquire.time_base.ns(plan.events[-1].start_ticks)
    }
    assert len(terminal_ends_ns) == 8
    engine = EngineDouble(backend="cpu")
    source = FTFSource(
        _settings(16),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=engine,
        readout=readout,
    )
    try:
        session = source.open_acquire(acquire)
        for index, state in enumerate((snapshot(), snapshot(0.8))):
            engine.calls.clear()
            result = expect_statistics(session.sample_round(state, index))
            assert session.dimension is not None
            assert session.dimension.dimension == 16
            assert session.dimension.pilot_point is None
            assert session.dimension.drop is None
            assert result.state_counts.shape[0] == 8
            # Projective row IDs are batch-local. Observe each physical point
            # via the positive-length span reaching its terminal readout.
            evolved_ends_ns = set()
            for model, batches in engine.calls:
                assert model.initial_density.shape == (16, 16)
                for batch in batches:
                    start = 0.0 if batch.start_ns is None else batch.start_ns
                    evolved_ends_ns.update(batch.end_ns[batch.end_ns > start])
            assert terminal_ends_ns <= evolved_ends_ns
    finally:
        source.close()


def test_round_keeps_the_selected_dimension_but_uses_live_dc(tmp_path: Path) -> None:
    engine = PilotEngine()
    numeric = _settings(16)
    numeric = numeric.model_copy(
        update={"chip": numeric.chip.model_copy(update={"temperature_k": 0.3})}
    )
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=engine,
        telemetry=RoutingTelemetry(tmp_path),
    )
    acquire = _request()
    before, after = snapshot(), snapshot(0.8)
    try:
        session = source.open_acquire(acquire)
        first = expect_statistics(session.sample_round(before, 0))
        second = expect_statistics(session.sample_round(before, 1))
        third = expect_statistics(session.sample_round(before, 2))
        changed = expect_statistics(session.sample_round(after, 3))
        reused = expect_statistics(session.sample_round(after, 4))
        assert first.state_counts[7, 12] == 0
        assert second.state_counts[7, 12] == third.state_counts[7, 12] == acquire.reps
        np.testing.assert_array_equal(changed.state_counts[:, 12], acquire.reps)
        np.testing.assert_array_equal(reused.state_counts, changed.state_counts)
        upper = source.model_facts(after).source
        upper_model = upper.model((0, 0, 0))
        initial = upper.initial_density[:13, :13]
        rounds = tuple(
            RoundRequest(acquire, state, index)
            for index, state in enumerate((before, before, before, after, after))
        )
        source.finish_acquire(AcquireOutcome(rounds, 1.0, 0.0, True))
        _execute_acquire(source, replace(acquire, sequence=acquire.sequence + 1), after)
    finally:
        source.close()

    first_round = [(16, (7,)), (13, (7,)), (13, tuple(range(7)))]
    full_round = (13, tuple(range(8)))
    assert engine.solve_shapes == [*first_round, full_round, full_round, *first_round]
    reduced_model, reduced_initial = engine.round_models[4]
    np.testing.assert_array_equal(
        reduced_model.energies_ghz, upper_model.energies_ghz[:13]
    )
    assert not np.array_equal(
        reduced_model.energies_ghz, engine.round_models[2][0].energies_ghz
    )
    assert initial.trace().real < 1.0
    np.testing.assert_array_equal(reduced_initial, initial / initial.trace().real)
    for row in _acquire_rows(tmp_path):
        extras = row["extras"]
        assert extras["ftf.selected_dimension"] == 13
        assert extras["ftf.pilot_point"] == 7
        assert extras["ftf.dimension_reused"] is False


def test_equal_drive_uses_the_far_carrier_endpoint(tmp_path: Path) -> None:
    acquire = _request()
    templates = tuple(
        replace(
            plan,
            events=(
                replace(
                    plan.events[0],
                    gain=np.full(2, 0.25),
                    frequency_mhz=np.array([150_000.0 if index == 0 else 250.0, 250.0]),
                ),
                *plan.events[1:],
            ),
        )
        for index, plan in enumerate(acquire.templates)
    )
    engine = PilotEngine()
    _, extras = _sample(engine, replace(acquire, templates=templates), 16, tmp_path)
    assert engine.solve_shapes[:2] == [(16, (0,)), (13, (0,))]
    assert extras["ftf.pilot_point"] == 0
