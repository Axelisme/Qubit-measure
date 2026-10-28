"""StagedAnalyzeService — the shared spine of the two off-main analyze stages.

``AnalyzeService`` (primary FIT/INTERACTIVE) and ``PostAnalyzeService`` (second
layer) run the identical operation lifecycle: a per-tab async handle (no exclusion,
ADR-0066), ``set_tab_analyzing`` for the duration, a single failure terminal path,
and a finished path that records into State before settling the handle. Only three
things differ between them — the gate + request assembly that starts the work, what
``record`` does with the worker's result, and which completion stage identifies a
failure. This base owns the lifecycle skeleton; each subclass fills the three
seams.

FIT/post analyze submit workers through OperationRunner (ADR-0066).
Interactive analyze retains an explicit runner lease across user-paced callbacks.
Each callback enters the captured event origin only while it executes.

The base deliberately holds NO operation-kind knowledge beyond "off-main work that
settles a per-tab handle": the INTERACTIVE branch (no worker, user-paced) and the
distinct gates (run_result vs analyze_result) live in the subclasses, not here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.gui.app.measure.events.completion import AnalyzeFailedPayload
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.session.operation_handles import (
    CancelHook,
    OperationHandles,
    OperationOutcome,
)
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    OperationLease,
    OperationRunner,
    OperationSpec,
    SettleFn,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from ..state import State


class StagedAnalyzeService:
    """Lifecycle skeleton shared by the primary- and post-analyze services.

    Subclasses declare their completion stage while the shared EventBus facts carry
    success and failure. The per-tab token map, failure terminal path, and
    start/finish templates are owned here.

    FIT/post analyze delegate worker submission to OperationRunner.
    Interactive analyze retains runner leases until a finish or cancel callback.
    """

    STARTED_FACT: ClassVar[TabInteractionFact]
    SUCCEEDED_FACT: ClassVar[TabInteractionFact]
    FAILED_FACT: ClassVar[TabInteractionFact]
    START_REJECTED_FACT: ClassVar[TabInteractionFact]
    FAILURE_STAGE: ClassVar[Literal["primary", "post"]]

    def __init__(
        self,
        state: State,
        runner: OperationRunner,
        bus: EventBus,
        handles: OperationHandles,
    ) -> None:
        self._state = state
        self._runner = runner
        self._bus = bus
        self._handles = handles
        # Per-tab token map: analyze has NO exclusion gate (ADR-0066), so two
        # different tabs can run concurrently. A single token would let the second
        # start clobber the first, leaking the first's handle. Keyed by tab_id,
        # every terminal path settles exactly the token its own start created.
        self._active_tokens: dict[str, int] = {}
        # Only interactive operations retain resources across owner callbacks.
        self._interactive_leases: dict[str, OperationLease] = {}

    def active_operations(self) -> tuple[tuple[str, int], ...]:
        """Return each tab's live handle for status, regardless of starter."""
        return tuple(sorted(self._active_tokens.items()))

    # -- shared handle bookkeeping -----------------------------------------

    def _release(self, tab_id: str, outcome: OperationOutcome) -> None:
        """Remove interactive ownership and give its domain outcome to the runner."""
        self._active_tokens.pop(tab_id, None)
        lease = self._interactive_leases.pop(tab_id, None)
        if lease is not None:
            lease.settle(outcome)

    def _open_token(self, tab_id: str, cancel_hook: CancelHook | None = None) -> int:
        """Retain an interactive lease without retaining ambient event origin."""
        lease = self._runner.open_lease(
            OperationSpec(
                exclusion=None,
                owner_id=tab_id,
                wants_progress=False,
                cancel_hook=cancel_hook,
            )
        )
        self._interactive_leases[tab_id] = lease
        self._active_tokens[tab_id] = lease.token
        return lease.token

    def _begin(self, tab_id: str) -> None:
        """Mark the tab analyzing and announce the interaction change.

        Shared tail of every start path (FIT worker, INTERACTIVE no-worker, post):
        the tab is busy for the duration so concurrent run/analyze is gated out.
        """
        token = self._active_tokens[tab_id]
        with self._bus.origin(self._handles.event_origin(token)):
            self._state.set_tab_analyzing(tab_id, analyzing=True)
            self._bus.emit(
                TabInteractionChangedPayload(tab_id=tab_id, fact=self.STARTED_FACT)
            )

    def _submit_with_runner(
        self,
        tab_id: str,
        work: Callable[[Any], Any],
        record: Callable[[str, Any], None],
        on_failure: Callable[[], None],
        _start_fail_message: str,
    ) -> int:
        """Submit ``work`` via OperationRunner (FIT/post analyze path).

        No exclusion (analyze never conflicts with hardware — ADR-0066), no
        progress, no cancel hook. begin() registers the token in _active_tokens
        but not _interactive_tabs, so cancel_interactive cannot cancel it.
        Marks the tab analyzing
        AFTER begin() succeeds (post-begin, stage2c_spec.md).

        Returns the operation token.
        """

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            # Store token for _active_tokens consistency before terminal logic.
            # (Token is already in _active_tokens from runner.begin, but we need
            # settle to replace _release in the terminal paths.)
            if bg.ok:
                _finish(bg.result, settle)
            else:
                assert bg.error is not None
                _fail(
                    bg.error,
                    settle,
                    failure_kind="analysis"
                    if isinstance(bg.error, AnalysisFailed)
                    else None,
                )

        def _finish(result: Any, settle: SettleFn) -> None:
            """The shared finished terminal path: record into State, then settle.

            ``record`` performs the subclass-specific State write. On a recording
            failure route through _fail so the handle never stays live. On success:
            clear analyzing, settle (State already observable — invariant 1),
            then emit completion facts.
            """
            try:
                record(tab_id, result)
            except Exception as exc:
                logger.exception("%s finished post-processing failed: %r", tab_id, exc)
                _fail(exc, settle)
                return
            self._active_tokens.pop(tab_id, None)
            self._state.set_tab_analyzing(tab_id, analyzing=False)
            # settle before facts — State visible to awaiter on wake.
            settle(OperationOutcome("finished"))
            self._bus.emit(
                TabInteractionChangedPayload(tab_id=tab_id, fact=self.SUCCEEDED_FACT)
            )

        def _fail(
            error: Exception,
            settle: SettleFn,
            *,
            failure_kind: Literal["analysis"] | None = None,
        ) -> None:
            """The single failure terminal path: clear analyzing, settle failed, emit."""
            logger.warning("staged-analyze failed: tab_id=%r error=%r", tab_id, error)
            try:
                on_failure()
            except Exception:
                logger.exception(
                    "Unpublished analysis plot cleanup failed: tab_id=%r", tab_id
                )
            self._active_tokens.pop(tab_id, None)
            self._state.set_tab_analyzing(tab_id, analyzing=False)
            # settle before facts — State visible to awaiter on wake.
            settle(OperationOutcome("failed", str(error), failure_kind=failure_kind))
            self._bus.emit(
                TabInteractionChangedPayload(tab_id=tab_id, fact=self.FAILED_FACT)
            )
            self._bus.emit(
                AnalyzeFailedPayload(
                    tab_id=tab_id,
                    stage=self.FAILURE_STAGE,
                    error_message=str(error),
                )
            )

        spec = OperationSpec(
            exclusion=None,  # analyze has no exclusion facet (ADR-0066)
            owner_id=tab_id,
            wants_progress=False,
            cancel_hook=None,
            work=work,
            run_in_pool=False,
            on_terminal=on_terminal,
        )

        try:
            token = self._runner.begin(spec)
        except Exception:
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id=tab_id,
                    fact=self.START_REJECTED_FACT,
                )
            )
            raise
        # Keep all live operations visible to status queries. Interactive
        # accessors distinguish their own tokens through _interactive_tabs.
        self._active_tokens[tab_id] = token

        # POST-BEGIN: mark analyzing after submit succeeds (begin-raise = no worker).
        self._begin(tab_id)
        return token
