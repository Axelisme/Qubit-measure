"""_StagedAnalyzeService — the shared spine of the two off-main analyze stages.

``AnalyzeService`` (primary FIT/INTERACTIVE) and ``PostAnalyzeService`` (second
layer) run the identical operation lifecycle: a per-tab async handle (no exclusion,
ADR-0019), ``set_tab_analyzing`` for the duration, a single failure terminal path,
and a finished path that records into State before settling the handle. Only three
things differ between them — the gate + request assembly that starts the work, what
``record`` does with the worker's result, and which Qt signals fire on
finish/fail. This base owns the lifecycle skeleton; each subclass fills the three
seams.

Stage 2c: FIT/post analyze changed from inline bg.submit to OperationRunner
(ADR-0026 §1). Interactive analyze (no worker, user-paced main-thread strategy)
still uses the direct _open_token path — it is explicitly excluded from runner
(stage2c_spec.md).

The base deliberately holds NO operation-kind knowledge beyond "off-main work that
settles a per-tab handle": the INTERACTIVE branch (no worker, user-paced) and the
distinct gates (run_result vs analyze_result) live in the subclasses, not here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.session.operation_handles import (
    CancelHook,
    OperationHandles,
    OperationOutcome,
)
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    OperationRunner,
    OperationSpec,
    SettleFn,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .ports import AnalyzeStatePort


class _StagedAnalyzeService(QObject):
    """Lifecycle skeleton shared by the primary- and post-analyze services.

    Subclasses declare their own ``finished``/``failed`` Qt signals (a signal must
    be bound on the concrete class) and expose them through ``_finished_signal`` /
    ``_failed_signal`` so the base can emit the right one. The per-tab token map,
    the failure terminal path, and the start/finish templates are owned here.

    FIT/post analyze delegate bg submission to OperationRunner (_submit_with_runner).
    Interactive analyze still uses _open_token + _release directly (no runner).
    """

    def __init__(
        self,
        state: AnalyzeStatePort,
        runner: OperationRunner,
        bus: EventBus,
        handles: OperationHandles,
    ) -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._bus = bus
        self._handles = handles
        # Per-tab token map: analyze has NO exclusion gate (ADR-0019), so two
        # different tabs can run concurrently. A single token would let the second
        # start clobber the first, leaking the first's handle. Keyed by tab_id,
        # every terminal path settles exactly the token its own start created.
        self._active_tokens: dict[str, int] = {}

    # -- subclass seam: which concrete Qt signal to emit on finish / fail --
    #
    # Typed Any: at runtime each accessor returns a *bound* signal (it has .emit),
    # but a class-level ``Signal`` annotation is the unbound descriptor type, which
    # the type checker would (wrongly) reject ``.emit`` on.

    @property
    def _finished_signal(self) -> Any:
        raise NotImplementedError

    @property
    def _failed_signal(self) -> Any:
        raise NotImplementedError

    # -- shared handle bookkeeping -----------------------------------------

    def _release(self, tab_id: str, outcome: OperationOutcome) -> None:
        """Settle and remove the per-tab handle (interactive path only).

        FIT/post use the runner-injected settle function; only interactive
        analyze settles via this method (no runner, direct _open_token).
        """
        token = self._active_tokens.pop(tab_id, None)
        if token is not None:
            self._handles.settle(token, outcome)

    def _open_token(self, tab_id: str, cancel_hook: CancelHook | None = None) -> int:
        """Mint and register the per-tab operation handle (pending).

        Used ONLY by the INTERACTIVE analyze path (no worker, user-paced). FIT
        and post-analyze handles are minted by OperationRunner._make_settle.
        ``cancel_hook`` is forwarded to OperationChannel.create; pass None for
        FIT-analyze (not cancellable) or a teardown callable for interactive.
        """
        token = self._handles.create(cancel_hook=cancel_hook)
        self._active_tokens[tab_id] = token
        return token

    def _begin(self, tab_id: str) -> None:
        """Mark the tab analyzing and announce the interaction change.

        Shared tail of every start path (FIT worker, INTERACTIVE no-worker, post):
        the tab is busy for the duration so concurrent run/analyze is gated out.
        """
        self._state.set_tab_analyzing(tab_id, True)
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))

    def _submit_with_runner(
        self,
        tab_id: str,
        work: Callable[[Any], Any],
        record: Callable[[str, Any], None],
        start_fail_message: str,
    ) -> int:
        """Submit ``work`` via OperationRunner (FIT/post analyze path).

        No exclusion (analyze never conflicts with hardware — ADR-0019), no
        progress, no cancel hook. begin() registers the token in _active_tokens
        so the interactive accessor (cancel_interactive, active_interactive_token)
        cannot see it (it is not in _interactive_tabs). Marks the tab analyzing
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
                _fail(bg.error, settle)

        def _finish(result: Any, settle: SettleFn) -> None:
            """The shared finished terminal path: record into State, then settle.

            ``record`` performs the subclass-specific State write. On a recording
            failure route through _fail so the handle never stays live. On success:
            clear analyzing, settle (State already observable — invariant 1),
            then emit signals/events.
            """
            try:
                record(tab_id, result)
            except Exception as exc:
                logger.exception("%s finished post-processing failed: %r", tab_id, exc)
                _fail(exc, settle)
                return
            self._active_tokens.pop(tab_id, None)
            self._state.set_tab_analyzing(tab_id, False)
            # settle before signals — State visible to awaiter on wake.
            settle(OperationOutcome("finished"))
            self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
            self._finished_signal.emit(tab_id, result)

        def _fail(error: Exception, settle: SettleFn) -> None:
            """The single failure terminal path: clear analyzing, settle failed, emit."""
            logger.warning("staged-analyze failed: tab_id=%r error=%r", tab_id, error)
            self._active_tokens.pop(tab_id, None)
            self._state.set_tab_analyzing(tab_id, False)
            # settle before signals — State visible to awaiter on wake.
            settle(OperationOutcome("failed", str(error)))
            self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
            self._failed_signal.emit(tab_id, error)

        spec = OperationSpec(
            exclusion=None,  # analyze has no exclusion facet (ADR-0019)
            owner_id=tab_id,
            wants_progress=False,
            cancel_hook=None,
            work=work,
            run_in_pool=False,
            on_terminal=on_terminal,
        )

        token = self._runner.begin(spec)
        # runner.begin mints and registers the token internally; we record it in
        # _active_tokens so the interactive accessors cannot see it (they check
        # _interactive_tabs, not _active_tokens, but _release uses _active_tokens).
        self._active_tokens[tab_id] = token

        # POST-BEGIN: mark analyzing after submit succeeds (begin-raise = no worker).
        self._begin(tab_id)
        return token
