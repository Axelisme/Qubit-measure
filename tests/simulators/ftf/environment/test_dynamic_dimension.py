"""FTF acquire chooses a dimension from a complete pilot density before ADC."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import override

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
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.ftf.environment import FTFSource
from zcu_tools.simulators.telemetry import RoutingTelemetry

from tests.program._profiles import profile_soccfg
from tests.simulators.ftf.environment._templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


def _settings(upper: int) -> FTFSettings:
    base = settings()
    numerics = base.numerics
    truncation = numerics.truncation
    # The 3x2x3 product supports an upper basis larger than the pilot minimum.
    truncation = truncation.model_copy(
        update={
            "retained_levels": upper,
            "f1": truncation.f1.model_copy(update={"levels": 3}),
            "f2": truncation.f2.model_copy(update={"levels": 3}),
        }
    )
    return base.model_copy(
        update={"numerics": numerics.model_copy(update={"truncation": truncation})}
    )


def _request(points: int = 8) -> AcquireRequest:
    # The highest-drive point is not first in C order or alone in its template.
    base = template(tones=1)
    templates = []
    for index in range(0, points, 2):
        rows = slice(0, min(2, points - index))
        drives = (
            replace(base.events[0], gain=np.array([0.1, 0.2 + index / 10])),
            base.events[1],
        )
        templates.append(
            replace(
                base,
                points=base.points[rows] + index,
                events=tuple(
                    replace(
                        drive,
                        start_ticks=drive.start_ticks[rows],
                        duration_ticks=drive.duration_ticks[rows],
                        frequency_mhz=drive.frequency_mhz[rows],
                        phase_deg=drive.phase_deg[rows],
                        gain=drive.gain[rows],
                        envelope_index=drive.envelope_index[rows],
                        active=drive.active[rows],
                    )
                    for drive in drives
                ),
                readouts=tuple(
                    replace(
                        read,
                        trigger_ticks=read.trigger_ticks[rows],
                        frequency_mhz=read.frequency_mhz[rows],
                        phase_deg=read.phase_deg[rows],
                        generator_channels=read.generator_channels[rows],
                    )
                    for read in base.readouts
                ),
                duration_ticks=base.duration_ticks[rows],
            )
        )
    return replace(
        request(templates=tuple(templates)), point_dims=(points,), loop_dims=(3, points)
    )


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
        return EvolutionResult(
            points, np.repeat(rho[None], len(points), axis=0), PathDiagnostics()
        )


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

    (summary,) = logs.glob("routing-*.jsonl")
    return result, json.loads(summary.read_text())["extras"]


def test_pilot_keeps_a_state_with_small_population_but_large_coherence(
    tmp_path: Path,
) -> None:
    engine = PilotEngine()
    acquire = _request()
    result, extras = _sample(engine, acquire, 16, tmp_path)
    assert engine.calls == [(16, (7,)), (13, (7,)), (13, tuple(range(8)))]
    assert result.state_counts.shape == (8, 16)
    np.testing.assert_array_equal(result.state_counts[:, 12], acquire.reps)
    np.testing.assert_array_equal(result.state_counts[:, 13:], 0)
    assert extras["ftf.selected_dimension"] == 13
    assert extras["ftf.pilot_drop"] == 0.0
    assert extras["ftf.pilot_error"] == 0.0
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
    assert extras["ftf.pilot_drop"] == 0.0
    if eligible:
        assert extras["ftf.pilot_error"] == pytest.approx(1.0, abs=2e-6)
    else:
        assert extras["ftf.pilot_error"] is None


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
    assert extras["ftf.pilot_wall_s"] == 0.0
    assert extras["ftf.pilot_drop"] is None
    assert extras["ftf.pilot_error"] is None


def test_round_keeps_the_selected_dimension_but_uses_live_dc() -> None:
    engine = PilotEngine()
    numeric = _settings(16)
    numeric = numeric.model_copy(
        update={"chip": numeric.chip.model_copy(update={"temperature_k": 0.3})}
    )
    source = FTFSource(
        numeric, DeviceManager(), soccfg=profile_soccfg("ftf"), engine=engine
    )
    acquire = _request()
    before, after = snapshot(), snapshot(0.8)
    try:
        session = source.open_acquire(acquire)
        session.sample_round(before, 0)
        session.sample_round(before, 1)
        session.sample_round(after, 2)
        upper = source.model_facts(after).source
        upper_model = upper.model((0, 0, 0))
        initial = upper.initial_density[:13, :13]
        next_acquire = source.open_acquire(
            replace(acquire, sequence=acquire.sequence + 1)
        )
        next_acquire.sample_round(after, 0)
    finally:
        source.close()

    first_round = [(16, (7,)), (13, (7,)), (13, tuple(range(8)))]
    assert engine.calls == [*first_round, (13, tuple(range(8))), *first_round]
    reduced_model, reduced_initial = engine.round_models[3]
    np.testing.assert_array_equal(
        reduced_model.energies_ghz, upper_model.energies_ghz[:13]
    )
    assert not np.array_equal(
        reduced_model.energies_ghz, engine.round_models[2][0].energies_ghz
    )
    assert initial.trace().real < 1.0
    np.testing.assert_array_equal(reduced_initial, initial / initial.trace().real)


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
    _sample(engine, replace(acquire, templates=templates), 16, tmp_path)
    assert engine.calls[:2] == [(16, (0,)), (13, (0,))]
