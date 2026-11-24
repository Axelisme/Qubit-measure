"""FTF allocates the per-segment allowance at its public Engine handoff."""

from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from typing import override

import numpy as np
import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    CancellationQuery,
    RoundRequest,
    TimelineTemplate,
)
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    EvolutionResult,
    ModelSource,
    SampledRWAProbeMemo,
    TimelineBatch,
)
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.ftf.environment import FTFReadoutConfig, FTFSource

from tests.core._simulation import ticks
from tests.program._profiles import profile_soccfg
from tests.simulators._engine import EngineDouble
from tests.simulators.ftf.environment._dimension_cases import (
    dimension_request,
    dimension_settings,
)
from tests.simulators.ftf.environment._templates import (
    herald_template,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


class BudgetEngine(EngineDouble):
    """Record the allowance on each solve and reuse without evolving densities."""

    def __init__(self) -> None:
        super().__init__()
        self.solves: list[tuple[bytes | None, float]] = []
        self.reuses: list[tuple[bytes, float]] = []

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
        probe_memo: SampledRWAProbeMemo | None = None,
    ) -> EvolutionResult:
        self.solves.append((reuse_key, budget))
        return super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
            capture=capture,
            probe_memo=probe_memo,
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
        self.reuses.append((key, budget))
        return super().reuse(source, key, cancel=cancel, budget=budget)


def _execute_and_reuse(
    acquire: AcquireRequest,
    config: FTFSettings,
    *,
    readout: FTFReadoutConfig | None = None,
) -> BudgetEngine:
    engine = BudgetEngine()
    with closing(
        FTFSource(
            config.with_telemetry(enabled=False),
            DeviceManager(),
            soccfg=profile_soccfg("ftf"),
            engine=engine,
            readout=readout,
        )
    ) as source:
        source.execute(RoundRequest(acquire, snapshot(), 0))
        source.execute(RoundRequest(replace(acquire, sequence=8), snapshot(), 0))
    return engine


def _control_train(segments: int) -> TimelineTemplate:
    """Put a contiguous drive train and the final idle on only the second point."""
    base = template(tones=1)
    drive, readout = base.events
    drives = tuple(
        replace(
            drive,
            start_ticks=np.full(2, step * ticks(0.002)),
            active=np.array([False, True]),
        )
        for step in range(segments - 1)
    )
    terminal = np.full(2, segments * ticks(0.002))
    shift = terminal - readout.start_ticks
    return replace(
        base,
        events=(*drives, replace(readout, start_ticks=terminal)),
        readouts=tuple(
            replace(window, trigger_ticks=window.trigger_ticks + shift[:, None])
            for window in base.readouts
        ),
        duration_ticks=base.duration_ticks + shift,
    )


@pytest.mark.parametrize(
    ("segments", "expected"),
    [(0, 0.01), (1, 0.01), (3, 0.0033333333333333335), (10, 0.001), (100, 0.001)],
)
def test_evolve_and_reuse_budget_follow_longest_timeline(
    segments: int, expected: float
) -> None:
    long = _control_train(segments)
    short = replace(_control_train(min(segments, 1)), points=long.points + 2)
    acquire = replace(
        request(templates=(long, short)), point_dims=(4,), loop_dims=(3, 4)
    )
    engine = _execute_and_reuse(acquire, settings())

    [(key, solved)] = engine.solves
    assert key is not None
    assert solved == expected
    assert engine.reuses == [(key, expected)]


def test_mixed_cavity_acquire_prices_the_plain_batch_and_each_span_separately() -> None:
    base = _control_train(5)
    earlier = replace(
        base.events[-1],
        start_ticks=np.zeros(2, dtype=np.int64),
        duration_ticks=np.full(2, ticks(0.002)),
        envelopes=(None,),
    )
    driven = replace(base, events=(earlier, *base.events))
    plain = replace(_control_train(3), points=driven.points + 2)
    acquire = replace(
        request(templates=(driven, plain)), point_dims=(4,), loop_dims=(3, 4)
    )
    engine = _execute_and_reuse(acquire, settings())

    plain_budgets, span_budgets = [], []
    for (_, budget), (_, batches) in zip(engine.solves, engine.calls, strict=True):
        budgets = plain_budgets if batches[0].start_ns is None else span_budgets
        budgets.append(budget)
    assert plain_budgets == [0.0033333333333333335]
    assert span_budgets and set(span_budgets) == {0.01}
    assert engine.reuses == engine.solves


def test_projective_nodes_get_one_span_budget_and_reuse_it() -> None:
    herald, readout = herald_template()
    acquire = request(templates=(herald,))
    engine = _execute_and_reuse(acquire, settings(), readout=readout)

    assert len({key for key, _ in engine.solves}) > 1
    assert {budget for _, budget in engine.solves} == {0.01}
    assert engine.reuses == engine.solves


def test_isolated_pilot_checks_and_remaining_batch_each_get_a_budget() -> None:
    acquire = dimension_request()
    engine = _execute_and_reuse(acquire, dimension_settings(16))

    upper, verify, rest = engine.solves
    assert upper == verify == (None, 0.0033333333333333335)
    assert rest[0] is not None
    assert rest[1] == 0.0033333333333333335
    assert engine.reuses == [rest]
