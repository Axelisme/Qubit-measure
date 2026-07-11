"""OperationRunner — kind-agnostic operation lifecycle mechanism (ADR-0026 §1).

``OperationRunner`` is the *mechanism* shared by every async operation (run /
FIT-analyze / post-analyze / device-setup / device-connect / device-disconnect).
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.operation_handles import (
    NO_RESULT,
    CancelHook,
    OperationHandles,
    OperationOutcome,
)
from zcu_tools.gui.session.ports import BackgroundExecutor, ExclusionGate, ProgressHub

logger = logging.getLogger(__name__)

# Sentinel re-exported so operation policies can import the lifecycle vocabulary here.
__all__ = [
    "NO_RESULT",
    "BgResult",
    "ExclusionRequest",
    "OperationRunner",
    "OperationSpec",
    "SettleFn",
]


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
    ``ok=False`` → ``error`` is the raised exception (``result`` is ``NO_RESULT``).
    """

    ok: bool
    result: Any
    error: Exception | None


# Injected by runner into each on_terminal call; policy must call it exactly once
# at the domain-correct moment (after all State writes, before signals/events).
SettleFn = Callable[[OperationOutcome], None]


@dataclass(frozen=True)
class OperationSpec:
    """Complete policy description for one operation.

    The runner reads this to drive the lifecycle; the calling service builds it
    and keeps any closure state (stop_event, live_container, tab_id, …) in its
    own scope — the runner never inspects those.

    Fields:
    - ``exclusion``: hardware exclusion parameters, or ``None`` for analyze ops
      (no exclusion facet, ADR-0019).
    - ``owner_id``: progress-container owner label (tab_id / device_name). Only
      consumed when ``wants_progress`` is True.
    - ``wants_progress``: whether to mint a per-op progress factory and pass it
      to ``work``. Device-setup and run need this; analyze ops do not.
    - ``cancel_hook``: forwarded to ``OperationHandles.create``; ``None`` for
      non-cancellable ops (FIT-analyze, connect).
    - ``work``: the off-main thunk. Receives the minted progress factory (or
      ``None`` if ``wants_progress`` is False) — all other ambient scopes
      (figure_ambient, schedule stop scope, …) are baked into the closure by the policy.
    - ``run_in_pool``: selects pool vs dedicated thread in the bg executor.
    - ``on_terminal``: called on the main thread when the bg work completes or
      errors. Receives (BgResult, SettleFn); the policy performs domain side-
      effects (State writes, signals) and calls settle exactly once.
    """

    exclusion: ExclusionRequest | None
    owner_id: str
    wants_progress: bool
    cancel_hook: CancelHook | None
    work: Callable[[Any], Any]
    run_in_pool: bool
    on_terminal: Callable[[BgResult, SettleFn], None]


class OperationRunner:
    """Kind-agnostic operation lifecycle mechanism (ADR-0026 §1). Only mechanism;
    every op's domain policy is injected via OperationSpec. Only recognises ports."""

    def __init__(
        self,
        gate: ExclusionGate,
        handles: OperationHandles,
        progress: ProgressHub,
        bg: BackgroundExecutor,
        bus: BaseEventBus,
    ) -> None:
        self._gate = gate
        self._handles = handles
        self._progress = progress
        self._bg = bg
        self._bus = bus

    def begin(self, spec: OperationSpec) -> int:
        """Synchronously open the operation and submit the work.

        Steps (in order):
        1. ``ensure_can_start`` (if exclusion) — fast-fail on conflict.
        2. ``handles.create(cancel_hook)`` — mint the token.
        3. ``gate.register`` (if exclusion) — register the lease.
        4. ``progress.make_factory`` (if wants_progress) — mint the pbar factory.
        5. Build the call-once settle closure and submit to bg.

        Returns the operation token. Raises on conflict (step 1 — nothing minted)
        or on submit failure (step 5 — settle(failed) unwinds all opened resources
        before re-raising).

        The ``ensure_can_start`` raise propagates directly: no token was minted,
        no resources allocated, no cleanup needed.
        """
        if spec.exclusion is not None:
            self._gate.ensure_can_start(
                spec.exclusion.kind, resource_id=spec.exclusion.resource_id
            )

        origin = self._bus.current_origin
        token = self._handles.create(cancel_hook=spec.cancel_hook, origin=origin)

        if spec.exclusion is not None:
            self._gate.register(
                token,
                spec.exclusion.kind,
                owner_id=spec.exclusion.owner_id,
                origin_kind=origin.kind,
                note=spec.exclusion.note,
                resource_id=spec.exclusion.resource_id,
            )

        factory = (
            self._progress.make_factory(token, spec.owner_id)
            if spec.wants_progress
            else None
        )

        settle = self._make_settle(spec, token)

        try:
            self._bg.submit(
                lambda: spec.work(factory),
                run_in_pool=spec.run_in_pool,
                on_done=lambda r: self._deliver_terminal(
                    token,
                    spec,
                    settle,
                    BgResult(ok=True, result=r, error=None),
                ),
                on_error=lambda e: self._deliver_terminal(
                    token,
                    spec,
                    settle,
                    BgResult(ok=False, result=NO_RESULT, error=e),
                ),
            )
        except Exception:
            # submit failed synchronously: unwind every resource opened above.
            settle(OperationOutcome("failed", "operation failed to start"))
            raise

        return token

    def _deliver_terminal(
        self,
        token: int,
        spec: OperationSpec,
        settle: SettleFn,
        bg: BgResult,
    ) -> None:
        """Run the policy terminal callback without letting it escape Qt delivery."""
        with self._bus.origin(self._handles.event_origin(token)):
            try:
                spec.on_terminal(bg, settle)
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
            if spec.wants_progress:
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
            if spec.exclusion is not None:
                try:
                    self._gate.release(token)
                except Exception:
                    logger.exception("operation gate release failed: token=%d", token)

        return settle
