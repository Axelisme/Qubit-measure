"""App-owned database search lifecycle shared by GUI and remote callers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Literal

from zcu_tools.analysis.fluxdep.search import DatabaseSearchResult, SearchCancelled
from zcu_tools.gui.app.fluxdep.event_bus import EventBus, SearchChangedPayload
from zcu_tools.gui.app.fluxdep.services.fit import FitService, PbarFactory
from zcu_tools.gui.app.fluxdep.state import (
    FIT_VERSION_KEY,
    PROJECT_VERSION_KEY,
    SELECTION_VERSION_KEY,
    SPECTRUM_SET_VERSION_KEY,
    FluxDepState,
    spectrum_version_key,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.operation_handles import (
    AwaitResult,
    OperationHandles,
    OperationOutcome,
)
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    OperationRunner,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.ports import BackgroundExecutor, OwnerScheduler
from zcu_tools.gui.session.progress_control import ProgressControlFacet
from zcu_tools.gui.session.services.progress import ProgressService

logger = logging.getLogger(__name__)
SEARCH_OWNER_ID = "fluxdep-search"


@dataclass(frozen=True)
class SearchActivity:
    """Latest operation: token identifies a retained handle; status is pending
    or a terminal finished/failed/cancelled value. error is a failure reason or
    None. This projection never carries diagnostic arrays.
    """

    token: int
    status: Literal["pending", "finished", "failed", "cancelled"]
    error: str | None = None


@dataclass(frozen=True)
class FluxDepSearchRuntime:
    """Matched background and progress adapters for app search composition.

    background runs detached work and delivers callbacks on State's owner.
    progress uses a transport delivering on that same owner. The composing
    app retains/drains the executor and its transport before destroying views.
    This pair carries no State, hardware gate or numeric terminal policy.
    """

    background: BackgroundExecutor
    progress: ProgressService


class FluxDepSearchOwner:
    """Single-flight search command surface. Mutation and current/result queries
    require the State-owner thread. outcome/await_outcome are thread-safe handle
    queries; await_outcome rejects the owner to avoid blocking delivery.
    """

    def __init__(
        self,
        state: FluxDepState,
        bus: EventBus,
        owner: OwnerScheduler,
        fit: FitService,
        publish_result: Callable[[DatabaseSearchResult], None],
        *,
        runtime: FluxDepSearchRuntime | None,
    ) -> None:
        """Compose numeric capture/compute and owner-only result publication.

        owner must use State's constructing thread. fit captures/computes inputs.
        runtime supplies matched background/progress adapters on that owner.
        None permits queries but start fails fast. publish_result commits numeric
        State and publishes FitChanged before
        the handle settles. No hardware exclusion is configured.
        """
        self._state = state
        self._bus = bus
        self._owner = owner
        self._fit = fit
        self._publish_result = publish_result
        self._handles = OperationHandles()
        self._runner = (
            OperationRunner(
                None, self._handles, runtime.progress, runtime.background, bus
            )
            if runtime is not None
            else None
        )
        self._progress_control = (
            ProgressControlFacet(runtime.progress) if runtime is not None else None
        )
        self._current: SearchActivity | None = None
        self._result: DatabaseSearchResult | None = None
        self._busy = False
        self._closing = False

    @property
    def current(self) -> SearchActivity | None:
        """Latest activity on the owner; None before any handle was opened."""
        self._assert_owner()
        return self._current

    @property
    def active_token(self) -> int | None:
        """Pending token on the owner, or None when no search is pending."""
        self._assert_owner()
        activity = self._current
        return (
            activity.token
            if activity is not None and activity.status == "pending"
            else None
        )

    @property
    def result(self) -> DatabaseSearchResult | None:
        """Latest successfully committed numeric result, on the owner.

        A later failed/cancelled operation does not erase this result.
        """
        self._assert_owner()
        return self._result

    @property
    def progress_control(self) -> ProgressControlFacet | None:
        """Configured owner-keyed progress surface; None without injection."""
        self._assert_owner()
        return self._progress_control

    def start(self) -> int:
        """Capture inputs/dependencies and submit one search; return its token.

        Raises on foreign thread, closing, busy, missing executor/progress or
        invalid inputs before opening a handle. Busy/closing are
        FailedPreconditionError (search_busy/search_closing); capture errors
        propagate as documented by FitService.capture_search. Missing runtime
        and foreign-thread misuse remain RuntimeError. Submit failure re-raises after
        cleanup and records a failed activity for the opened handle.
        """
        self._assert_owner()
        if self._closing:
            raise FailedPreconditionError(
                "search owner is closing", reason_code="search_closing"
            )
        if self._busy:
            raise FailedPreconditionError(
                "search is already pending", reason_code="search_busy"
            )
        runner = self._runner
        if runner is None:
            raise RuntimeError("search background executor is unavailable")
        if self._progress_control is None:
            raise RuntimeError("search progress service is unavailable")
        inputs = self._fit.capture_search()
        keys = (
            PROJECT_VERSION_KEY,
            FIT_VERSION_KEY,
            SELECTION_VERSION_KEY,
            SPECTRUM_SET_VERSION_KEY,
            *(spectrum_version_key(name) for name in self._state.spectrums),
        )
        dependencies = tuple((key, self._state.version.get(key)) for key in keys)
        stop = Event()

        def work(factory: PbarFactory | None) -> DatabaseSearchResult:
            return self._fit.compute_search(
                inputs,
                pbar_factory=factory,
                cancel_requested=stop.is_set,
            )

        self._busy = True
        try:
            return runner.begin(
                OperationSpec(
                    exclusion=None,
                    owner_id=SEARCH_OWNER_ID,
                    wants_progress=True,
                    cancel_hook=stop.set,
                    work=work,
                    run_in_pool=True,
                    on_terminal=lambda bg, settle: self._terminal(
                        bg, settle, dependencies
                    ),
                    on_opened=self._opened,
                )
            )
        except Exception:
            # Runner has settled/cleaned any opened token before re-raising.
            self._busy = False
            activity = self._current
            if activity is not None and activity.status == "pending":
                outcome = self._handles.known_outcome(activity.token)
                if outcome is not None:
                    with self._bus.origin(self._handles.event_origin(activity.token)):
                        self._finish_activity(activity.token, outcome)
            raise

    def cancel(self, token: int) -> None:
        """Request cooperative stop on the owner; terminal cancel is a no-op.

        Unknown/evicted tokens raise InvalidInputError (unknown_operation).
        Foreign threads raise RuntimeError. A request is not a terminal result.
        """
        self._assert_owner()
        if self.outcome(token) is None:
            self._handles.cancel(token)

    def outcome(self, token: int) -> OperationOutcome | None:
        """Thread-safe terminal query for a retained token; None means pending.

        Unknown/evicted tokens raise InvalidInputError (unknown_operation).
        """
        try:
            return self._handles.known_outcome(token)
        except KeyError as exc:
            raise InvalidInputError(
                f"unknown search operation {token}", reason_code="unknown_operation"
            ) from exc

    def await_outcome(self, token: int, timeout: float) -> AwaitResult:
        """Wait off-owner for one known token. Timeout does not cancel.

        Owner calls raise RuntimeError; unknown/evicted tokens raise
        InvalidInputError (unknown_operation), preserving the lookup cause.
        timeout is seconds, with the shared handle wait semantics.
        """
        if self._owner.is_owner_thread():
            raise RuntimeError("cannot await search on the owner thread")
        try:
            return self._handles.await_known_outcome(token, timeout)
        except KeyError as exc:
            raise InvalidInputError(
                f"unknown search operation {token}", reason_code="unknown_operation"
            ) from exc

    def begin_close(self) -> None:
        """Permanently refuse new starts and request pending cancellation, on owner.

        Worker failure remains failed; closing success is discarded as cancelled.
        Calling again is harmless. The app must drain its executor before disposal.
        """
        self._assert_owner()
        self._closing = True
        token = self.active_token
        if token is not None:
            self.cancel(token)

    def _terminal(
        self, bg: BgResult, settle: SettleFn, dependencies: tuple[tuple[str, int], ...]
    ) -> None:
        """Commit only a current numeric success, then settle and publish activity."""
        self._assert_owner()
        activity = self._current
        if activity is None or activity.status != "pending":
            raise RuntimeError("search delivery has no pending owner")
        try:
            if isinstance(bg.error, SearchCancelled):
                outcome = OperationOutcome("cancelled")
            elif bg.error is not None:
                logger.error("search worker failed", exc_info=bg.error)
                outcome = OperationOutcome("failed", str(bg.error))
            elif self._closing:
                outcome = OperationOutcome("cancelled")
            elif any(
                self._state.version.get(key) != version for key, version in dependencies
            ):
                outcome = OperationOutcome("failed", "search inputs changed")
            else:
                result: DatabaseSearchResult = bg.result
                self._publish_result(result)
                self._result = result
                outcome = OperationOutcome("finished")
        except Exception as exc:
            # Isolate domain commit failures while still releasing admission.
            logger.exception("search result publication failed")
            outcome = OperationOutcome("failed", str(exc))
        settle(outcome)
        self._finish_activity(activity.token, outcome)

    def _assert_owner(self) -> None:
        self._state.assert_owner_thread()
        if not self._owner.is_owner_thread():
            raise RuntimeError("search command requires the owner thread")

    def _opened(self, token: int) -> None:
        self._current = SearchActivity(token, "pending")
        with self._bus.origin(self._handles.event_origin(token)):
            self._bus.emit(SearchChangedPayload(token=token, status="pending"))

    def _finish_activity(self, token: int, outcome: OperationOutcome) -> None:
        self._current = SearchActivity(token, outcome.status, outcome.error)
        self._busy = False
        self._bus.emit(
            SearchChangedPayload(
                token=token, status=outcome.status, error=outcome.error
            )
        )
