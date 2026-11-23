"""Synthetic elapsed work and prescribed Engine answers for telemetry owners."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import override

from zcu_tools.simulate.boundary import CancellationQuery
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    EvolutionResult,
    FamilyRouting,
    ModelSource,
    PathDiagnostics,
    SampledRWAProbeMemo,
    TimelineBatch,
)

from tests.simulators._engine import EngineDouble


class WorkClock:
    """Advance only for named collaborators, including their failing calls.

    now is elapsed synthetic seconds. calls counts invocations by function name.
    fail/fail_on select an optional post-work RuntimeError at a one-based call.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.calls: dict[str, int] = {}
        self.fail: str | None = None
        self.fail_on = 1

    def observe[**P, T](self, method: Callable[P, T], seconds: float) -> Callable[P, T]:
        """Delegate real non-evolution work and charge its prescribed seconds."""

        def run(*args: P.args, **kwargs: P.kwargs) -> T:
            name = method.__name__
            self.calls[name] = self.calls.get(name, 0) + 1
            try:
                result = method(*args, **kwargs)
                if self.fail == name and self.calls[name] == self.fail_on:
                    raise RuntimeError("synthetic numeric failure")
                return result
            finally:
                self.now += seconds

        return run


class TelemetryEngine(EngineDouble):
    """Prescribe identity densities and one explicit fixture routing row per call.

    This delegates only to the non-evolving EngineDouble, never Engine.evolve.
    Its input-density answer also preserves projected continuation states.
    """

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
        result = super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
            capture=capture,
            probe_memo=probe_memo,
        )
        return replace(
            result,
            paths=PathDiagnostics(
                routing={"fixture": FamilyRouting(rows=1)},
                segments_by_kernel={"fixture": 1},
                duration_ns_by_kernel={"fixture": 1.0},
            ),
        )
