"""Environment-owned snapshot/execution collaborators and per-call admission."""

import logging
import time
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

import numpy as np

from zcu_tools.program.shot_time import request_round_time_s
from zcu_tools.simulate.boundary import (
    AcquireRequest,
    CancellationQuery,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
)

from ._envelope_pool import EnvelopePool
from .clock import SimulationTiming
from .result import IntegratedRound


@dataclass(frozen=True)
class SimulationProvenance:
    """Binding identity and versions, with only identifier intended for publication.

    Attributes:
        identifier: Opaque UUID4 hex string generated once per binding. This is
            the only field copied to public Run metadata as simulation_id. It
            identifies the environment instance, not model content or a seed.
        backend_version: Environment-author-supplied backend implementation
            version string for private diagnostics. The binding imposes no
            format or version compatibility check.
        rng_version: Binding-generated private RNG/seed-derivation version,
            currently "PCG64/SeedSequence-v2/numpy-<version>".

    backend_version and rng_version are not normal GUI/MCP metadata. Publishing
    identifier does not authorize publishing this whole record or numeric inputs.
    """

    identifier: str
    backend_version: str
    rng_version: str


@dataclass(frozen=True)
class AcquireOutcome:
    """Observed result of one admitted acquire, after its body and cleanup.

    Attributes:
        rounds: Captured round inputs in execution order, including the input
            of a failed round. This is not a list of completed rounds.
        wall_s: Monotonic seconds after sequence admission through body and
            mixin cleanup, before the observer, including pacing.
        compile_wall_s: Sum of compiling scope seconds, including failed builds;
            zero when no compiling scope ran.
        complete: True when the body and cleanup returned without an escaping
            BaseException, including a normally returned partial Stop result.
            This does not mean all requested rounds completed.
    """

    rounds: tuple[RoundRequest, ...]
    wall_s: float
    compile_wall_s: float
    complete: bool


@dataclass(frozen=True)
class SimulationCallbacks:
    """Optional acquire lifecycle observers, owned by the environment.

    wait_ready: no-argument startup wait after admission, before request creation;
        errors release admission and propagate. on_end: completed-acquire observer
        receiving an AcquireOutcome, including failed rounds and pacing.
        Its ordinary Exceptions warn once without masking execution errors;
        BaseExceptions propagate with the acquire error retained as context.
    on_begin: no-argument preparation after wait_ready, before request construction;
        errors release admission and propagate. All default None
        and must not reenter acquisition.
    """

    wait_ready: Callable[[], object] | None = None
    on_end: Callable[[AcquireOutcome], None] | None = None
    on_begin: Callable[[], None] | None = None


class SimulationBinding:
    """One environment's numeric source; snapshots are captured at execution time.

    capture_snapshot runs once immediately before each round's execute. execute
    receives only a RoundRequest and returns all ADCs together. Neither callback
    owns averaging. Calls are synchronous; acquire rejects simultaneous/reentrant
    acquisition before changing active state. The acquisition adapter places
    request construction, execution and cleanup inside the acquire scope.
    This binding owns no drivers and does not close callback resources.

    Attributes:
        capture_snapshot: Environment-owned, no-argument callable returning the
            current EnvironmentSnapshot. It runs in the acquiring thread.
        execute: Environment-owned callable taking a RoundRequest and returning
            IntegratedRound (integrated) or RoundSignals (decimated) for all ADCs. It runs in the same acquiring thread.
        root_seed: Nonnegative integer supplied through seed or generated once
            by NumPy SeedSequence when seed is None. Private sampling entropy.
        rng_version: Private RNG/derivation version string, currently
            "PCG64/SeedSequence-v2/numpy-<version>".
        provenance: SimulationProvenance for this binding. Only its identifier
            is intended for public Run metadata.
        envelope_pool: Byte-bounded immutable DAC envelope pool shared across
            this binding's acquisitions. It retains no mutable program buffers.
        last_request: Most recently installed AcquireRequest, or None before any
            request is installed. A failure before installation may leave the
            preceding request here. It is not cleared by acquire.
        rounds: Tuple of RoundRequest inputs captured since the latest admitted
            acquire, in execution order. Empty before any round snapshot succeeds.
            A backend failure leaves its captured input here, so this is not a
            list of successfully completed rounds.

    last_request, rounds and seed/version fields remain private numeric data in
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
        timing: SimulationTiming | None = None,
    ) -> None:
        """Bind synchronous environment callbacks and choose the root seed once.

        capture_snapshot takes no arguments and returns current numeric device
        inputs and model identity. execute takes those inputs in a RoundRequest
        and returns all ADC statistics/signals under the simulate.boundary
        layout/borrow contract. The environment retains ownership of both callbacks and their
        resources; neither callback performs QICK averaging. Raw sources own rounding.
        backend_version is an author-defined private implementation version
        string. seed is a nonnegative built-in int, or None for fresh entropy.
        callbacks.wait_ready optionally waits for environment startup after admission and
        before consuming sequence or constructing requests. Its result is ignored;
        errors release admission and propagate. It must not reenter acquisition.

        timing=None uses SimulationTiming's defaults. timing.time_scale is a
        finite nonnegative physical-time multiplier. Each round
        targets time_scale * request_round_time_s(request). The program shot-time
        owner sums integer body/relax ticks and reps, then TimeBase converts the
        aggregate to seconds. One follows physical time; 0.1 runs
        ten times faster. Zero (default) disables pacing. Without clock it also
        skips round clock reads. Snapshot
        and execution consume this budget; failures never wait. With cancel,
        synchronous waits check is_set at intervals of at most 10 ms and return
        early so the acquisition adapter applies its existing Stop policy.

        timing.clock optionally shares an environment-owned SimulationClock. Successful
        rounds credit the unscaled physical duration minus actual elapsed seconds,
        including pacing. Cancelled/failed rounds add no offset. None leaves
        simulated time untracked. callbacks.on_begin optionally freezes environment
        inputs after readiness and before request construction. Errors propagate
        and release admission. It does not consume an acquire sequence on failure.

        callbacks.on_end optionally observes one completed admitted acquire, including
        failed/partial calls. It receives an AcquireOutcome after the body and
        cleanup, before releasing admission, and must not reenter. Ordinary
        Exceptions emit at most one warning and never mask an acquisition error,
        even under -W error. BaseExceptions propagate, retaining any acquire
        error as context. callbacks=None supplies no lifecycle hooks.
        on_end=None avoids acquire/compiling clocks and completion notifications.

        Raises ValueError for an invalid scale or boolean/non-int/negative seed, and TypeError
        if NumPy returns non-integer entropy. Callback execution is deferred to
        sample_round; this constructor does not validate their returned records.
        """
        if seed is not None and (
            isinstance(seed, bool) or type(seed) is not int or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None")
        hooks = SimulationCallbacks() if callbacks is None else callbacks
        self._on_end = hooks.on_end
        self._observer_warned = False
        self._acquire_started = 0.0
        self._compile_wall_s = 0.0
        policy = SimulationTiming() if timing is None else timing
        self._time_scale = policy.time_scale
        self._clock = policy.clock
        self.envelope_pool = EnvelopePool()
        self._wait_ready = hooks.wait_ready
        self._on_begin = hooks.on_begin
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

    @contextmanager
    def acquire(self) -> Generator[int]:
        """Admit one synchronous acquire and yield its zero-based int sequence.

        Place request construction, execution and cleanup inside this scope.
        Readiness/on_begin failure releases admission without consuming sequence
        or notifying on_end. Admission resets rounds and compile time; a later
        failure still consumes sequence. last_request is unchanged until install.
        Concurrent/reentrant entry raises RuntimeError without consuming sequence.
        The observer receives the body's normal-return/escaping-error outcome
        after cleanup. Ordinary observer Exceptions warn once; BaseExceptions
        propagate. Admission releases in either case.
        """
        if not self._admission.acquire(blocking=False):
            raise RuntimeError("simulation acquisition is already active (reentry)")
        try:
            if self._wait_ready is not None:
                self._wait_ready()
            if self._on_begin is not None:
                self._on_begin()
            sequence = self._sequence
            self._sequence += 1
            self.rounds = ()
            self._compile_wall_s = 0.0
            if self._on_end is not None:
                self._acquire_started = time.monotonic()
            complete = False
            try:
                yield sequence
                complete = True
            finally:
                self._notify_outcome(complete=complete)
        finally:
            self._admission.release()

    @contextmanager
    def compiling(self) -> Generator[None]:
        """Measure builder wall seconds inside an acquire scope, including failure.

        Use only around request construction, not installation or execution.
        Each scope adds elapsed monotonic seconds to that acquire's outcome.
        Outside acquire, measured time is not delivered to an observer. This
        method does not check scope ownership. Without on_end it reads no clock.
        """
        if self._on_end is None:
            yield
            return
        started = time.monotonic()
        try:
            yield
        finally:
            self._compile_wall_s += time.monotonic() - started

    def _notify_outcome(self, *, complete: bool) -> None:
        if self._on_end is None:
            return
        outcome = AcquireOutcome(
            self.rounds,
            time.monotonic() - self._acquire_started,
            self._compile_wall_s,
            complete,
        )
        try:
            self._on_end(outcome)
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

    def sample_round(
        self,
        request: AcquireRequest,
        index: int,
        *,
        cancel: CancellationQuery | None = None,
        pacing_cancel: CancellationQuery | None = None,
    ) -> IntegratedRound | RoundSignals:
        """Capture inputs and synchronously return one joint ADC realization.

        request is the installed, fixed AcquireRequest for the admitted call.
        index is its zero-based software round index, supplied by the SoC. Call
        inside an admitted acquire scope; this method does not check admission,
        request identity or index range and does not increment the index itself.
        cancel is forwarded unchanged as read-only round control; None disables it.
        pacing_cancel optionally controls waiting only, without reaching the backend.
        When omitted, waiting uses cancel. This lets decimated callers interrupt
        pacing without adding backend cancellation or changing completed traces.
        A backend Cancelled exception propagates without a result or pacing.

        Calls capture_snapshot once, appends the resulting RoundRequest to rounds,
        then calls execute once. Validates the mode/result type and returns the
        borrowed result; the adapter validates channel layouts and copies arrays.
        Callback exceptions propagate. Snapshot failure appends nothing; execute
        failure leaves the captured input in rounds.
        A mode/result mismatch raises ValueError.
        It does not normalize, average, round signals or
        update last_request. Successful calls wait for their scaled physical target,
        including snapshot and execute time. Slow rounds do not wait; zero
        disables pacing. Failures never add a delay. Cancellation ends waiting
        within one 10 ms sleep interval and returns the borrowed result; the
        adapter applies its existing completed-round policy.
        """
        timed = self._time_scale > 0 or self._clock is not None
        now = time.monotonic if self._clock is None else self._clock.now
        started = now() if timed else 0.0
        round_request = RoundRequest(
            request, self.capture_snapshot(), index, cancel=cancel
        )
        self.rounds += (round_request,)
        result = self.execute(round_request)
        if not (
            (request.mode == "integrated" and isinstance(result, IntegratedRound))
            or (request.mode == "decimated" and isinstance(result, RoundSignals))
        ):
            raise ValueError(f"backend result does not match {request.mode} mode")
        if self._time_scale > 0:
            target = self._time_scale * request_round_time_s(request)
            remaining = target - (now() - started)
            wait_cancel = cancel if pacing_cancel is None else pacing_cancel
            if wait_cancel is None:
                if remaining > 0:
                    time.sleep(remaining)
            else:
                while remaining > 0 and not wait_cancel.is_set():
                    time.sleep(min(remaining, 0.01))
                    remaining = target - (now() - started)
        if self._clock is not None and not any(
            query is not None and query.is_set() for query in (cancel, pacing_cancel)
        ):
            self._clock.complete_round(request_round_time_s(request), now() - started)
        return result
