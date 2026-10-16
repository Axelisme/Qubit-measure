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
    RoundStatistics,
)


@dataclass(frozen=True)
class SimulationProvenance:
    """Binding identity and versions, with only identifier intended for publication.

    Attributes:
        identifier: Opaque UUID4 hex string generated once per binding. This is
            the only field copied to public Run metadata as simulation_id. It
            identifies the environment instance, not model content or a seed.
        backend_version: Environment-author-supplied backend implementation
            version string, retained as private replay metadata. The binding
            imposes no format or version compatibility check.
        rng_version: Binding-generated private RNG/seed-derivation version,
            currently "PCG64/SeedSequence-v2/numpy-<version>". Replay requires
            compatible RNG/library versions as well as the private inputs.

    backend_version and rng_version are not normal GUI/MCP metadata. Publishing
    identifier does not authorize publishing this whole record or replay inputs.
    """

    identifier: str
    backend_version: str
    rng_version: str


class SimulationBinding:
    """One environment's numeric source; snapshots are captured at execution time.

    capture_snapshot runs once immediately before each round's execute. execute
    receives only a RoundRequest and returns all ADCs together. Neither callback
    owns averaging. Calls are synchronous; begin rejects simultaneous/reentrant
    acquisition before changing active state. The acquisition adapter pairs each
    successful begin with end in a finally block. This binding owns no drivers
    and does not close callback resources.

    Attributes:
        capture_snapshot: Environment-owned, no-argument callable returning the
            current EnvironmentSnapshot. It runs in the acquiring thread.
        execute: Environment-owned callable taking a RoundRequest and returning
            RoundStatistics (integrated) or RoundSignals (decimated) for all ADCs. It runs in the same acquiring thread.
        root_seed: Nonnegative integer supplied through seed or generated once
            by NumPy SeedSequence when seed is None. Private replay entropy.
        rng_version: Private RNG/derivation version string, currently
            "PCG64/SeedSequence-v2/numpy-<version>".
        provenance: SimulationProvenance for this binding. Only its identifier
            is intended for public Run metadata.
        last_request: Most recently installed AcquireRequest, or None before any
            request is installed. A failure before installation may leave the
            preceding request here. It is not cleared by begin or end.
        rounds: Tuple of RoundRequest inputs captured since the latest successful
            begin, in execution order. Empty before any round snapshot succeeds.
            A backend failure leaves its captured input here, so this is not a
            list of successfully completed rounds.

    last_request, rounds and seed/version fields remain private replay data in
    memory, not automatic serialized history or normal GUI/MCP metadata.
    """

    def __init__(
        self,
        capture_snapshot: Callable[[], EnvironmentSnapshot],
        execute: Callable[[RoundRequest], RoundStatistics | RoundSignals],
        *,
        backend_version: str,
        seed: int | None = None,
    ) -> None:
        """Bind synchronous environment callbacks and choose the root seed once.

        capture_snapshot takes no arguments and returns current numeric device
        inputs and model identity. execute takes those inputs in a RoundRequest
        and returns all ADC statistics/signals under the core.simulation
        layout/borrow contract. The environment retains ownership of both callbacks and their
        resources; neither callback should perform QICK averaging or rounding.
        backend_version is an author-defined private implementation version
        string. seed is a nonnegative built-in int, or None for fresh entropy.

        Raises ValueError for a boolean, non-int or negative seed, and TypeError
        if NumPy returns non-integer entropy. Callback execution is deferred to
        sample_round; this constructor does not validate their returned records.
        """
        if seed is not None and (
            isinstance(seed, bool) or type(seed) is not int or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None")
        self.capture_snapshot = capture_snapshot
        self.execute = execute
        entropy = np.random.SeedSequence(seed).entropy
        if not isinstance(entropy, int):
            raise TypeError("root seed entropy must be an integer")
        self.root_seed = entropy
        self.rng_version = f"PCG64/SeedSequence-v2/numpy-{np.__version__}"
        self.provenance = SimulationProvenance(
            uuid4().hex, backend_version, self.rng_version
        )
        self.last_request: AcquireRequest | None = None
        self.rounds: tuple[RoundRequest, ...] = ()
        self._sequence = 0
        self._admission = Lock()

    def begin(self) -> int:
        """Claim admission and return the next zero-based acquire sequence.

        Call before request construction. Success consumes the sequence even if
        acquisition later fails or stops, and resets rounds to an empty tuple.
        last_request stays unchanged until the SoC installs a new request.
        Raises RuntimeError on concurrent/reentrant admission without changing
        the active call or consuming a sequence. After success, the caller must
        call end exactly once, including on failure.
        """
        if not self._admission.acquire(blocking=False):
            raise RuntimeError("simulation acquisition is already active (reentry)")
        sequence = self._sequence
        self._sequence += 1
        self.rounds = ()
        return sequence

    def end(self) -> None:
        """Release a successful begin after synchronous acquisition finishes.

        The caller must invoke this once in its finally block, after round work
        ends. No replay inputs or callback resources are cleared. Returns None;
        releasing an unlocked admission lock raises RuntimeError.
        """
        self._admission.release()

    def sample_round(
        self, request: AcquireRequest, index: int
    ) -> RoundStatistics | RoundSignals:
        """Capture inputs and synchronously return one joint ADC realization.

        request is the installed, fixed AcquireRequest for the admitted call.
        index is its zero-based software round index, supplied by the SoC. Call
        between a successful begin and end; this method does not check admission,
        request identity or index range and does not increment the index itself.

        Calls capture_snapshot once, appends the resulting RoundRequest to rounds,
        then calls execute once. Validates the mode/result type and returns the
        borrowed result; the adapter validates channel layouts and copies arrays.
        Integrated RoundSignals remain accepted during backend migration.
        Callback exceptions propagate. Snapshot failure appends nothing; execute
        failure leaves the captured input in rounds. A mode/result mismatch
        raises ValueError. It does not normalize, average, round signals or
        update last_request.
        """
        round_request = RoundRequest(request, self.capture_snapshot(), index)
        self.rounds += (round_request,)
        result = self.execute(round_request)
        if request.mode == "integrated" and isinstance(result, RoundStatistics):
            return result
        # Legacy backend path; contract cleanup removes integrated RoundSignals.
        if isinstance(result, RoundSignals):
            return result
        raise ValueError(f"backend result does not match {request.mode} mode")
