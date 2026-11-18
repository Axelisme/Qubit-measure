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
from zcu_tools.simulators.telemetry import RoutingTelemetry

from tests.program._profiles import profile_soccfg
from tests.simulators.ftf.environment._dimension_cases import dimension_bypass
from tests.simulators.ftf.environment._dimension_cases import (
    dimension_request as _request,
)
from tests.simulators.ftf.environment._dimension_cases import (
    dimension_settings as _settings,
)
from tests.simulators.ftf.environment._templates import expect_statistics, snapshot

pytestmark = pytest.mark.sim


class PilotEngine(Engine):
    """Physical density answers make the FTF selection observable at the ADC."""

    def __init__(self, *, extent: int = 12, bad_phase: bool = False) -> None:
        super().__init__(1, backend="cpu")
        self.calls: list[tuple[int, tuple[int, ...]]] = []
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
        self.calls.append((n, tuple(int(p) for p in points)))
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


def _sample(
    engine: Engine, acquire: AcquireRequest, upper: int, logs: Path
) -> tuple[RoundStatistics, dict[str, object]]:
    source = FTFSource(
        _settings(upper),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=engine,
        telemetry=RoutingTelemetry(logs),
    )
    try:
        state = snapshot()
        result = expect_statistics(source.open_acquire(acquire).sample_round(state, 0))
        source.finish_acquire(
            AcquireOutcome((RoundRequest(acquire, state, 0),), 1.0, 0.0, True)
        )
    finally:
        source.close()

    return result, _dimension_extras(logs)


def test_pilot_keeps_a_state_with_small_population_but_large_coherence(
    tmp_path: Path,
) -> None:
    engine = PilotEngine()
    acquire = replace(_request(), reps=32768, loop_dims=(32768, 8))
    result, extras = _sample(engine, acquire, 16, tmp_path)
    assert engine.calls == [(16, (7,)), (13, (7,)), (13, tuple(range(7)))]
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
    assert engine.calls == expected
    np.testing.assert_array_equal(result.state_counts[:7, 12], acquire.reps)
    assert result.state_counts[7, 12] == 0
    assert extras["ftf.selected_dimension"] == 16
    assert extras["ftf.pilot_point"] == 7
    assert extras["ftf.pilot_drop"] == 0.0
    if eligible:
        assert extras["ftf.pilot_error"] == pytest.approx(1.0, abs=2e-6)
    else:
        assert extras["ftf.pilot_error"] is None


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
    assert engine.calls == [(16, (7,)), (16, tuple(range(7))), (16, tuple(range(8)))]


@pytest.mark.parametrize(("upper", "points"), [(16, 7), (12, 8)])
def test_small_acquire_uses_upper_without_a_pilot(
    tmp_path: Path, upper: int, points: int
) -> None:
    engine = PilotEngine()
    acquire = _request(points)
    result, extras = _sample(engine, acquire, upper, tmp_path)
    assert engine.calls == [(upper, tuple(range(points)))]
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
    source = FTFSource(
        _settings(16),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        engine=Engine(1, backend="cpu"),
        readout=readout,
    )
    try:
        session = source.open_acquire(acquire)
        result = expect_statistics(session.sample_round(snapshot(), 0))
        assert session.dimension is not None
        assert session.dimension.dimension == 16
        assert session.dimension.drop is None
        assert result.state_counts.shape[0] == 8
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
        session.sample_round(after, 3)
        assert first.state_counts[7, 12] == 0
        assert second.state_counts[7, 12] == third.state_counts[7, 12] == acquire.reps
        upper = source.model_facts(after).source
        upper_model = upper.model((0, 0, 0))
        initial = upper.initial_density[:13, :13]
        rounds = tuple(
            RoundRequest(acquire, state, index)
            for index, state in enumerate((before, before, before, after))
        )
        source.finish_acquire(AcquireOutcome(rounds, 1.0, 0.0, True))
        next_acquire = source.open_acquire(
            replace(acquire, sequence=acquire.sequence + 1)
        )
        next_acquire.sample_round(after, 0)
    finally:
        source.close()

    first_round = [(16, (7,)), (13, (7,)), (13, tuple(range(7)))]
    full_round = (13, tuple(range(8)))
    assert engine.calls == [*first_round, full_round, full_round, *first_round]
    reduced_model, reduced_initial = engine.round_models[4]
    np.testing.assert_array_equal(
        reduced_model.energies_ghz, upper_model.energies_ghz[:13]
    )
    assert not np.array_equal(
        reduced_model.energies_ghz, engine.round_models[2][0].energies_ghz
    )
    assert initial.trace().real < 1.0
    np.testing.assert_array_equal(reduced_initial, initial / initial.trace().real)
    extras = _dimension_extras(tmp_path)
    assert extras["ftf.selected_dimension"] == 13
    assert extras["ftf.pilot_point"] == 7


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
    assert engine.calls[:2] == [(16, (0,)), (13, (0,))]
    assert extras["ftf.pilot_point"] == 0
