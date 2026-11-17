"""Non-evolving collaborator for environment composition tests."""

from collections.abc import Collection, Sequence

import numpy as np
from numpy.typing import NDArray
from zcu_tools.core.simulation import CancellationQuery
from zcu_tools.simulate.evolution import (
    Cancelled,
    Engine,
    EvolutionResult,
    ModelSource,
    PathDiagnostics,
    TimelineBatch,
)


def controlled_engine(engine: Engine) -> "EngineDouble":
    """Return the injected collaborator or report a test setup error."""
    if not isinstance(engine, EngineDouble):
        raise AssertionError("environment must use its injected EngineDouble")
    return engine


class EngineDouble(Engine):
    """Return supplied densities, or the input density unchanged, without solving.

    The unchanged-input fixture represents a prescribed identity result, not a
    numerical reference. Tests needing distinct point populations supply result.
    reuse_allowed is an explicit collaborator answer controlled by the test.
    error lets environment tests observe propagation of an Engine failure.
    requested_families records the caller's constructor policy without routing it.
    """

    def __init__(
        self,
        workers: int = 1,
        *,
        result: EvolutionResult | None = None,
        families: Collection[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(workers, families=families, **kwargs)
        self.requested_families = families
        self.result = result
        self.calls: list[tuple[ModelSource, tuple[TimelineBatch, ...]]] = []
        self.reuse_allowed = True
        self.error: Exception | None = None

    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
    ) -> EvolutionResult:
        """Record the public handoff and deliver the fixture's Engine answer."""
        self.calls.append((source, tuple(batches)))
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        points = np.concatenate([batch.points for batch in batches])
        densities: NDArray[np.complex128] = np.concatenate(
            [
                np.repeat(source.initial_density[None], len(batch.points), axis=0)
                if batch.initial_densities is None
                else batch.initial_densities
                for batch in batches
            ]
        )
        return EvolutionResult(points, densities, PathDiagnostics())

    def reuse(
        self,
        source: ModelSource,
        key: bytes,
        *,
        cancel: CancellationQuery | None = None,
        budget: float = 1e-3,
    ) -> bool:
        """Return the test-controlled reuse answer, including cancellation."""
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if self.error is not None:
            raise self.error
        return self.reuse_allowed
