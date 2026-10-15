"""Environment-owned snapshot/execution collaborators and per-call admission."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

import numpy as np

from zcu_tools.core.simulation import (
    AcquireRequest,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
)


@dataclass(frozen=True)
class SimulationProvenance:
    """Public identifier only; never exposes the model or private replay inputs."""

    identifier: str
    backend_version: str
    rng_version: str


class SimulationBinding:
    """One environment's numeric source; snapshots are captured at execution time.

    capture_snapshot runs once immediately before each round's execute. execute
    receives only a RoundRequest and returns all ADCs together. Neither callback
    owns averaging. Calls are synchronous; simultaneous/reentrant acquisition is
    rejected before changing any active call. last_request and rounds retain the
    latest private replay inputs in memory (not automatically serialized).
    """

    def __init__(
        self,
        capture_snapshot: Callable[[], EnvironmentSnapshot],
        execute: Callable[[RoundRequest], RoundSignals],
        *,
        backend_version: str,
        seed: int | None = None,
    ) -> None:
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None")
        self.capture_snapshot = capture_snapshot
        self.execute = execute
        entropy = np.random.SeedSequence(seed).entropy
        if not isinstance(entropy, int):
            raise TypeError("root seed entropy must be an integer")
        self.root_seed = entropy
        self.rng_version = f"PCG64/SeedSequence-v1/numpy-{np.__version__}"
        self.provenance = SimulationProvenance(
            uuid4().hex, backend_version, self.rng_version
        )
        self.last_request: AcquireRequest | None = None
        self.rounds: tuple[RoundRequest, ...] = ()
        self._sequence = 0
        self._admission = Lock()

    def begin(self) -> int:
        """Claim this SoC before request building; failures still consume identity."""
        if not self._admission.acquire(blocking=False):
            raise RuntimeError("simulation acquisition is already active (reentry)")
        sequence = self._sequence
        self._sequence += 1
        self.rounds = ()
        return sequence

    def end(self) -> None:
        """Release admission after the synchronous acquire, even on failure."""
        self._admission.release()

    def sample_round(self, request: AcquireRequest, index: int) -> RoundSignals:
        round_request = RoundRequest(request, self.capture_snapshot(), index)
        self.rounds += (round_request,)
        return self.execute(round_request)
