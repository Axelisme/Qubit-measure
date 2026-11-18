"""Public CPU DOP853 tableau selection and scheduling costs."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, merge_path_diagnostics

from .accuracy_cases import accuracy_case


@pytest.mark.parametrize(
    ("duration", "budget", "dp54"),
    [
        pytest.param(10_000.0, 1e-3, True, id="inclusive-duration-and-budget"),
        pytest.param(10_001.0, 1e-3, False, id="over-duration"),
        pytest.param(100.0, 0.999e-3, False, id="under-budget"),
    ],
)
def test_cpu_dop853_selects_tableau_by_row_duration_and_budget(
    duration: float, budget: float, dp54: bool
) -> None:
    case = accuracy_case("sampled_gaussian_9")
    end = np.full_like(case.batch.end_ns, duration)
    batch = replace(
        case.batch,
        end_ns=end,
        tracks=tuple(
            replace(
                track,
                duration_ns=end,
                envelopes=tuple(
                    replace(env, sample_ns=env.sample_ns * (duration / 100.0))
                    for env in track.envelopes
                ),
            )
            for track in case.batch.tracks
        ),
    )
    with closing(Engine(1, backend="cpu", families=case.families)) as engine:
        paths = engine.evolve(
            case.source, (batch,), budget=budget, telemetry=True
        ).paths

    assert paths.segments_by_kernel == {"dop853": len(batch.points)}
    report = paths.routing["dop853"]
    assert report.dp54_rows == (len(batch.points) if dp54 else 0)
    assert report.dop853_rows == (0 if dp54 else len(batch.points))
    merged = merge_path_diagnostics((paths, paths)).routing["dop853"]
    assert merged.dp54_rows == 2 * report.dp54_rows
    assert merged.dop853_rows == 2 * report.dop853_rows
    reused = paths.for_reuse(telemetry=True).routing["dop853"]
    assert (reused.dp54_rows, reused.dop853_rows) == (0, 0)


def test_dispatch_prices_the_short_cpu_tableau_below_dop853() -> None:
    case = accuracy_case("sampled_gaussian_9")
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=case.families)) as engine,
    ):
        tight = engine.inspect_dispatch(case.source, (case.batch,), budget=0.999e-3)
        loose = engine.inspect_dispatch(case.source, (case.batch,), budget=1e-3)
        result = engine.evolve(case.source, (case.batch,), budget=1e-3, telemetry=True)

    costs = []
    for inspection in (tight, loose):
        prices = [
            candidate.scheduling_cost
            for row in inspection.rows
            for candidate in row.candidates
            if row.family == candidate.family == "dop853"
        ]
        assert len(prices) == len(case.batch.points)
        costs.append(sum(price for price in prices if price is not None))
    assert 0 < costs[1] < costs[0]
    routing = result.paths.routing["dop853"]
    assert routing.dp54_rows == len(case.batch.points)
    assert routing.estimated_cost == pytest.approx(costs[1])
