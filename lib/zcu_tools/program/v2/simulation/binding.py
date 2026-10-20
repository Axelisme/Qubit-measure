"""Environment-owned snapshot/execution collaborators and per-call admission."""

import logging
import math
import time
import warnings
from collections.abc import Callable
from contextlib import suppress
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

from ._envelope_pool import EnvelopePool
from .result import IntegratedRound


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


@dataclass(frozen=True)
class SimulationCallbacks:
    """Optional acquire lifecycle observers, owned by the environment.

    wait_ready: no-argument startup wait after admission, before request creation;
        errors release admission and propagate. on_end: completed-acquire observer
        receiving all captured rounds and elapsed admitted wall seconds, including
        failed rounds/pacing. Its exceptions warn once, without masking execution
        errors. Both default None and must not reenter acquisition.
    """

    wait_ready: Callable[[], object] | None = None
    on_end: Callable[[tuple[RoundRequest, ...], float], None] | None = None


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
            IntegratedRound (integrated) or RoundSignals (decimated) for all ADCs. It runs in the same acquiring thread.
        root_seed: Nonnegative integer supplied through seed or generated once
            by NumPy SeedSequence when seed is None. Private replay entropy.
        rng_version: Private RNG/derivation version string, currently
            "PCG64/SeedSequence-v2/numpy-<version>".
        provenance: SimulationProvenance for this binding. Only its identifier
            is intended for public Run metadata.
        envelope_pool: Byte-bounded immutable DAC envelope pool shared across
            this binding's acquisitions. It retains no mutable program buffers.
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
        execute: Callable[[RoundRequest], IntegratedRound | RoundSignals],
        *,
        backend_version: str,
        seed: int | None = None,
        callbacks: SimulationCallbacks | None = None,
        min_round_duration_s: float = 0.0,
    ) -> None:
        """Bind synchronous environment callbacks and choose the root seed once.

        capture_snapshot takes no arguments and returns current numeric device
        inputs and model identity. execute takes those inputs in a RoundRequest
        and returns all ADC statistics/signals under the core.simulation
        layout/borrow contract. The environment retains ownership of both callbacks and their
        resources; neither callback performs QICK averaging. Raw sources own rounding.
        backend_version is an author-defined private implementation version
        string. seed is a nonnegative built-in int, or None for fresh entropy.
        callbacks.wait_ready optionally waits for environment startup after admission and
        before consuming sequence or constructing requests. Its result is ignored;
        errors release admission and propagate. It must not reenter acquisition.

        min_round_duration_s is finite, nonnegative wall time in seconds. Each
        successful sample_round includes snapshot/execution time and waits only
        for the remainder. Zero (default) disables pacing and clock reads.
        Failures propagate without waiting. Waiting is synchronous, not cancellable.

        callbacks.on_end optionally observes one completed admitted acquire, including
        failed/partial calls. It receives all captured rounds (possibly empty)
        and elapsed monotonic seconds since begin, including pacing. It runs
        before releasing admission and must not reenter. Exceptions emit at most
        one warning and never mask an acquisition error, even under -W error.
        callbacks=None supplies no lifecycle hooks. on_end=None avoids
        acquire-level clock reads and completion notifications.

        Raises ValueError for an invalid duration or boolean/non-int/negative seed, and TypeError
        if NumPy returns non-integer entropy. Callback execution is deferred to
        sample_round; this constructor does not validate their returned records.
        """
        if seed is not None and (
            isinstance(seed, bool) or type(seed) is not int or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None")
        if not math.isfinite(min_round_duration_s) or min_round_duration_s < 0:
            raise ValueError("min_round_duration_s must be finite and nonnegative")
        hooks = SimulationCallbacks() if callbacks is None else callbacks
        self._on_end = hooks.on_end
        self._observer_warned = False
        self._acquire_started = 0.0
        self._min_round_duration_s = min_round_duration_s
        self.envelope_pool = EnvelopePool()
        self._wait_ready = hooks.wait_ready
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
        try:
            if self._wait_ready is not None:
                self._wait_ready()
        except BaseException:
            self._admission.release()
            raise
        sequence = self._sequence
        self._sequence += 1
        self.rounds = ()
        if self._on_end is not None:
            self._acquire_started = time.monotonic()
        return sequence

    def end(self) -> None:
        """Release a successful begin after synchronous acquisition finishes.

        The caller must invoke this once in its finally block, after round work
        ends. No replay inputs or callback resources are cleared. Returns None;
        releasing an unlocked admission lock raises RuntimeError.
        The optional on_end observer runs once, including after failed rounds;
        observer exceptions cannot mask acquisition failures or leak admission.
        """
        try:
            if self._on_end is not None:
                self._on_end(self.rounds, time.monotonic() - self._acquire_started)
        except Exception as error:
            if not self._observer_warned:
                self._observer_warned = True
                logging.getLogger(__name__).exception(
                    "simulation acquire observer failed"
                )
                with suppress(Warning):
                    warnings.warn(
                        f"simulation acquire observer failed: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            self._admission.release()

    def sample_round(
        self, request: AcquireRequest, index: int
    ) -> IntegratedRound | RoundSignals:
        """Capture inputs and synchronously return one joint ADC realization.

        request is the installed, fixed AcquireRequest for the admitted call.
        index is its zero-based software round index, supplied by the SoC. Call
        between a successful begin and end; this method does not check admission,
        request identity or index range and does not increment the index itself.

        Calls capture_snapshot once, appends the resulting RoundRequest to rounds,
        then calls execute once. Validates the mode/result type and returns the
        borrowed result; the adapter validates channel layouts and copies arrays.
        Callback exceptions propagate. Snapshot failure appends nothing; execute
        failure leaves the captured input in rounds. A mode/result mismatch
        raises ValueError. It does not normalize, average, round signals or
        update last_request. Successful calls wait until min_round_duration_s
        has elapsed since entry, including snapshot and execute. Slow rounds do
        not wait; zero disables pacing. Failures never add a delay.
        """
        started = time.monotonic() if self._min_round_duration_s > 0 else 0.0
        round_request = RoundRequest(request, self.capture_snapshot(), index)
        self.rounds += (round_request,)
        result = self.execute(round_request)
        if not (
            (request.mode == "integrated" and isinstance(result, IntegratedRound))
            or (request.mode == "decimated" and isinstance(result, RoundSignals))
        ):
            raise ValueError(f"backend result does not match {request.mode} mode")
        if self._min_round_duration_s > 0:
            remaining = self._min_round_duration_s - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        return result
