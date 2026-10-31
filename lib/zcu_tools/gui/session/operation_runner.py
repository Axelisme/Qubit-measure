"""OperationRunner — kind-agnostic operation lifecycle mechanism (ADR-0066).

``OperationRunner`` owns async operations and synchronous resource leases.
Async clients include run, analyze, device setup and connection. Synchronous
clients use open_lease for a retained lifetime or inline for a lexical body.
It owns exactly one responsibility: orchestrate the standard lifecycle step
sequence::

    ensure_can_start (if exclusion) → create handle → register exclusion
    → mint progress factory (if wants_progress) → submit to bg
    → on terminal: discard progress → settle handle → release exclusion

Every *policy* decision (what to write to State, which signals to emit, how to
interpret a cancelled return) lives in the ``OperationSpec`` callbacks supplied
by the operation's service — the runner never touches State.

The ``settle`` function injected into ``on_terminal`` is call-once (guarded
internally): a policy calling it twice is logged and then no-ops on the second
call. Policy correctness is still required — the guard is a safety net, not a
licence for sloppy policy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from zcu_tools.gui.event_bus import BaseEventBus, EventOrigin
from zcu_tools.gui.session.operation_handles import (
    NO_RESULT,
    CancelHook,
    OperationHandles,
    OperationOutcome,
)
from zcu_tools.gui.session.ports import (
    BackgroundError,
    BackgroundExecutor,
    ExclusionGate,
    ProgressHub,
)

logger = logging.getLogger(__name__)

# Sentinel re-exported so operation policies can import the lifecycle vocabulary here.
__all__ = [
    "NO_RESULT",
    "BgResult",
    "ExclusionRequest",
    "OperationInfrastructure",
    "OperationLease",
    "OperationRunner",
    "OperationSpec",
    "SettleFn",
]


@dataclass(frozen=True)
class OperationInfrastructure:
    """App-composed operation owners shared by session construction and devices.

    Fields:
        gate: Injected exclusion port used for device admission and busy queries.
        background: Executor for off-owner work and owner-thread terminal delivery.
        runner: Lifecycle mechanism composed with these same gate, background and
            handles instances. Per-operation policy still selects required facets.
        handles: Handle/cancellation owner used by the runner and device commands.

    This record only forwards existing owners; it creates no infrastructure and
    does not require other operation clients to acquire a hardware lease.
    """

    gate: ExclusionGate
    background: BackgroundExecutor
    runner: OperationRunner
    handles: OperationHandles


@dataclass(frozen=True)
class ExclusionRequest:
    """What hardware exclusion lease this operation needs.

    ``kind`` is the OperationKind wire string; ``owner_id`` is the human-
    readable owner label (tab_id / device_name) written to the gate; optional
    ``resource_id`` scopes the conflict check to a specific resource (device ops
    use the device name so different devices can operate concurrently).
    """

    kind: str
    owner_id: str
    note: str
    resource_id: str | None = None

    def __post_init__(self) -> None:
        if not self.note.strip():
            raise ValueError("exclusion note must not be blank")


@dataclass(frozen=True)
class BgResult:
    """What the background callback delivered.

    ``ok=True`` → ``result`` is the worker's return value (``error`` is None).
    ``ok=False`` → ``error`` is the original worker Exception or KeyboardInterrupt
    (``result`` is ``NO_RESULT``), delivered on the owner thread.
    """

    ok: bool
    result: Any
    error: BackgroundError | None


# Injected by runner into each on_terminal call; policy must call it exactly once
# at the domain-correct moment (after all State writes, before signals/events).
SettleFn = Callable[[OperationOutcome], None]


@dataclass
class OperationLease:
    """Retained operation resources, independent of an ambient origin scope.

    token identifies the pending handle. origin is its captured event origin,
    including operation_id; enter bus.origin(lease.origin) only while executing
    each domain callback, never across event-loop turns. Call settle(outcome)
    after domain commits to discard progress, settle the handle and release the
    gate. progress_factory is the requested per-operation progress callable, or
    None when wants_progress is False. This record owns no worker and performs
    no hardware rollback. Obtain leases from open_lease or inline; callers do
    not construct them or change their fields. _settle and _bus are injected
    runner mechanisms, not domain policy.
    """

    token: int
    origin: EventOrigin
    _settle: SettleFn
    _bus: BaseEventBus
    progress_factory: Callable[..., object] | None = None
    _settled: bool = field(default=False, init=False, repr=False)

    @property
    def settled(self) -> bool:
        """Whether settlement was attempted, even if a cleanup step failed."""
        return self._settled

    def settle(self, outcome: OperationOutcome) -> None:
        """Finish with the domain's outcome under the captured operation origin.

        Cleanup failures are logged independently and do not escape. Duplicate
        calls are logged no-ops. Cancel requests alone do not settle resources;
        the domain chooses finished, failed or cancelled after its own cleanup.
        """
        self._settled = True
        with self._bus.origin(self.origin):
            self._settle(outcome)


@dataclass(frozen=True)
class OperationSpec:
    """Complete policy description for one operation.

    The runner reads this to drive the lifecycle; the calling service builds it
    and keeps any closure state (stop_event, live_container, tab_id, …) in its
    own scope — the runner never inspects those.

    Fields:
    - ``exclusion``: hardware exclusion parameters, or ``None`` for analyze ops
      (no exclusion facet, ADR-0066).
    - ``owner_id``: progress-container owner label (tab_id / device_name). Only
      consumed when ``wants_progress`` is True.
    - ``wants_progress``: whether to mint a per-op progress factory and pass it
      to ``work``. Device-setup and run need this; analyze ops do not.
    - ``cancel_hook``: forwarded to ``OperationHandles.create``; ``None`` for
      non-cancellable ops (FIT-analyze, connect).
    - ``work``: the off-main thunk, required by begin and forbidden for
      open_lease/inline (leave None). Receives the minted progress factory (or
      ``None`` if ``wants_progress`` is False) — all other ambient scopes
      (such as schedule stop scope) are baked into the closure by the policy.
    - ``run_in_pool``: selects pool vs dedicated thread in the bg executor;
      unused by open_lease/inline, which never submit background work.
    - ``on_terminal``: required by begin, forbidden for open_lease/inline
      (leave None). Called on the main thread when the bg work completes or
      errors. Receives (BgResult, SettleFn); the policy performs domain side-
      effects (State writes, signals) and calls settle exactly once.
    - ``on_opened``: optional owner-thread hook receiving the newly opened
      token before lease registration, progress notification or submit. With an
      exclusion request, only commit owner state here; do not notify listeners
      or invoke reentrant UI callbacks before the lease is registered. Without
      exclusion, an analysis owner may also publish its current activity.
      A hook failure settles failed, cleans opened resources, and propagates.
    """

    exclusion: ExclusionRequest | None
    owner_id: str
    wants_progress: bool
    cancel_hook: CancelHook | None
    work: Callable[[Any], Any] | None = None
    run_in_pool: bool = False
    on_terminal: Callable[[BgResult, SettleFn], None] | None = None
    # Called before progress notifications or submit, so domain owners can
    # latch admission and retain the token even when startup later fails.
    on_opened: Callable[[int], None] | None = None


class OperationRunner:
    """Kind-agnostic operation lifecycle mechanism (ADR-0066). Only mechanism;
    every op's domain policy is injected via OperationSpec. Only recognises ports.

    gate/progress may be None for operations that do not request those facets.
    Missing required facets fail before any handle is created. All entry points,
    lease settlement and background callbacks execute on the domain owner thread.
    """

    def __init__(
        self,
        gate: ExclusionGate | None,
        handles: OperationHandles,
        progress: ProgressHub | None,
        bg: BackgroundExecutor,
        bus: BaseEventBus,
    ) -> None:
        self._gate = gate
        self._handles = handles
        self._progress = progress
        self._bg = bg
        self._bus = bus

    @contextmanager
    def inline(self, spec: OperationSpec) -> Generator[OperationLease]:
        """Run a lexical owner-thread body under its operation event origin.

        Open the same resources as open_lease and yield its lease. No background
        work is submitted. Normal exit settles finished; a body exception settles
        failed(str(exc)) and propagates unchanged, including BaseException.
        Cleanup failures are isolated and logged, never replacing the body error.
        Restore the outer event origin on either exit. Domain writes belong
        inside the body; cancellation remains the domain's outcome decision.
        A body that calls lease.settle(outcome) chooses the terminal outcome;
        exiting does not attempt settlement again, even if the body later raises.
        """
        lease = self.open_lease(spec)
        with self._bus.origin(lease.origin):
            try:
                yield lease
            except BaseException as exc:
                if not lease.settled:
                    lease.settle(OperationOutcome("failed", str(exc)))
                raise
            else:
                if not lease.settled:
                    lease.settle(OperationOutcome("finished"))

    def open_lease(self, spec: OperationSpec) -> OperationLease:
        """Open resources on the owner thread without submitting work.

        Check exclusion, create a handle from the current event origin, call
        on_opened(token), register exclusion, and create requested progress.
        Return a retained lease without retaining an ambient event-origin scope.
        The caller
        must settle it after domain commits, including on failure or cancel.
        Admission errors propagate before creating a token. Startup failures
        settle failed, release opened resources and propagate unchanged.
        work/on_terminal must both be None, otherwise raise ValueError before
        admission; the caller supplies synchronous work and terminal policy.
        """
        if spec.work is not None or spec.on_terminal is not None:
            raise ValueError("synchronous lease forbids work and on_terminal")
        if spec.wants_progress and self._progress is None:
            raise RuntimeError("operation requires a progress service")
        if spec.exclusion is not None and self._gate is None:
            raise RuntimeError("operation requires an exclusion gate")
        token, origin = self._create_operation(spec)
        settle = self._make_settle(spec, token)
        operation_origin = self._handles.event_origin(token)
        try:
            factory = self._register_resources(spec, token, origin)
        except BaseException as exc:
            with self._bus.origin(operation_origin):
                settle(OperationOutcome("failed", str(exc)))
            raise
        return OperationLease(token, operation_origin, settle, self._bus, factory)

    def begin(self, spec: OperationSpec) -> int:
        """Synchronously open the operation and submit the work.

        Steps (in order):
        1. ``ensure_can_start`` (if exclusion) — fast-fail on conflict.
        2. ``handles.create(cancel_hook)`` — mint the token.
        3. on_opened(token) (if configured), then gate.register (if exclusion).
        4. ``progress.make_factory`` (if wants_progress) — mint the pbar factory.
        5. Build the call-once settle closure and submit to bg.

        Returns the operation token. work and on_terminal must both be provided,
        otherwise raise ValueError before admission. Missing gate/progress facets
        and conflicts raise before a handle is minted. Hook, registration,
        factory or submit failure settles failed, unwinds opened resources
        and re-raises.

        The ``ensure_can_start`` raise propagates directly: no token was minted,
        no resources allocated, no cleanup needed.
        """
        work, on_terminal = spec.work, spec.on_terminal
        if work is None or on_terminal is None:
            raise ValueError("begin requires work and on_terminal")
        if spec.exclusion is not None and self._gate is None:
            raise RuntimeError("operation requires an exclusion gate")
        if spec.wants_progress and self._progress is None:
            raise RuntimeError("operation requires a progress service")
        token, origin = self._create_operation(spec)
        settle = self._make_settle(spec, token)
        try:
            factory = self._register_resources(spec, token, origin)
            self._bg.submit(
                lambda: work(factory),
                run_in_pool=spec.run_in_pool,
                on_done=lambda r: self._deliver_terminal(
                    token,
                    on_terminal,
                    settle,
                    BgResult(ok=True, result=r, error=None),
                ),
                on_error=lambda e: self._deliver_terminal(
                    token,
                    on_terminal,
                    settle,
                    BgResult(ok=False, result=NO_RESULT, error=e),
                ),
            )
        except Exception:
            # Startup failed synchronously: unwind every resource opened above.
            settle(OperationOutcome("failed", "operation failed to start"))
            raise

        return token

    def _create_operation(self, spec: OperationSpec) -> tuple[int, EventOrigin]:
        """Check admission, capture origin and create a handle before cleanup applies."""
        if spec.exclusion is not None and self._gate is not None:
            self._gate.ensure_can_start(
                spec.exclusion.kind, resource_id=spec.exclusion.resource_id
            )
        origin = self._bus.current_origin
        token = self._handles.create(cancel_hook=spec.cancel_hook, origin=origin)
        return token, origin

    def _register_resources(
        self, spec: OperationSpec, token: int, origin: EventOrigin
    ) -> Callable[..., object] | None:
        """Notify the owner, register exclusion and allocate requested progress.

        The entrypoint owns the cleanup boundary and its exception policy.
        Registration uses the origin captured before on_opened can change it.
        """
        if spec.on_opened is not None:
            spec.on_opened(token)
        if spec.exclusion is not None and self._gate is not None:
            self._gate.register(
                token,
                spec.exclusion.kind,
                owner_id=spec.exclusion.owner_id,
                origin_kind=origin.kind,
                note=spec.exclusion.note,
                resource_id=spec.exclusion.resource_id,
            )
        return (
            self._progress.make_factory(token, spec.owner_id)
            if spec.wants_progress and self._progress is not None
            else None
        )

    def continue_operation[R](
        self,
        token: int,
        *,
        work: Callable[[], R],
        run_in_pool: bool,
        on_terminal: Callable[[BgResult, SettleFn], None],
        settle: SettleFn,
    ) -> int:
        """Submit another producer for an idle, unsettled operation on its owner.

        token is the handle returned by begin. The policy must wait for the
        previous producer's owner delivery and retain its original settle
        closure without calling it. work captures the retained run resources;
        it receives no new progress factory. run_in_pool selects the executor's
        shared pool or dedicated thread, as in begin. on_terminal receives the
        returned BgResult and this same settle closure under the handle's
        original event origin. It may retain ownership again or settle once.

        Return token unchanged. Do not create a handle, check/register a new
        lease, replace cancellation, or allocate/discard progress. Unknown or
        evicted tokens raise KeyError; already settled tokens raise RuntimeError
        before submission. A synchronous submit error propagates unchanged and
        leaves ownership intact so the domain can retry or terminate. Worker
        errors reach on_terminal. An escaping terminal callback is logged and
        settles failed using the supplied closure. No hardware rollback occurs.

        The domain owns producer exclusivity and verifies that settle belongs
        to token; this mechanism does not add a producer registry or policy.
        """
        if self._handles.known_outcome(token) is not None:
            raise RuntimeError(f"operation token is already settled: {token}")

        # Submission failure leaves the retained resources for the domain owner.
        self._bg.submit(
            work,
            run_in_pool=run_in_pool,
            on_done=lambda r: self._deliver_terminal(
                token,
                on_terminal,
                settle,
                BgResult(ok=True, result=r, error=None),
            ),
            on_error=lambda e: self._deliver_terminal(
                token,
                on_terminal,
                settle,
                BgResult(ok=False, result=NO_RESULT, error=e),
            ),
        )
        return token

    def _deliver_terminal(
        self,
        token: int,
        on_terminal: Callable[[BgResult, SettleFn], None],
        settle: SettleFn,
        bg: BgResult,
    ) -> None:
        """Run the policy terminal callback without letting it escape Qt delivery."""
        with self._bus.origin(self._handles.event_origin(token)):
            try:
                on_terminal(bg, settle)
            except Exception as exc:
                logger.exception("operation terminal callback failed")
                settle(OperationOutcome("failed", str(exc)))

    def _make_settle(self, spec: OperationSpec, token: int) -> SettleFn:
        """Build the call-once settle closure for this operation.

        Calling order mirrors the old ``_release_lease`` / ``_release`` pattern:
        1. discard progress (if applicable)
        2. settle the handle
        3. release the exclusion lease (if applicable)

        A second call is logged and ignored (call-once guard). Policy correctness
        is still expected — the guard is a safety net, not an invitation.
        """
        done = False

        def settle(outcome: OperationOutcome) -> None:
            nonlocal done
            if done:
                logger.error(
                    "operation settle called more than once: token=%d owner=%s",
                    token,
                    spec.owner_id,
                )
                return  # call-once: duplicate calls are no-ops
            done = True
            if spec.wants_progress and self._progress is not None:
                try:
                    self._progress.discard_operation(token)
                except Exception:
                    logger.exception(
                        "operation progress discard failed: token=%d", token
                    )
            try:
                self._handles.settle(token, outcome)
            except Exception:
                logger.exception("operation handle settle failed: token=%d", token)
            if spec.exclusion is not None and self._gate is not None:
                try:
                    self._gate.release(token)
                except Exception:
                    logger.exception("operation gate release failed: token=%d", token)

        return settle
