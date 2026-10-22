"""Worker lifetime and detached progress for async recipe declarations."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
from threading import Condition, Event, Thread, current_thread
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, JsonValue, TypeAdapter

from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.experiment.script_host import CancelScope, ScriptStatus
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.recipe_api import (
    RecipeCancelled,
    RecipeDeclaration,
    RecipeOutcome,
)
from zcu_tools.mcp.measure.recipe_contracts import RecipeNeedsParameters
from zcu_tools.mcp.measure.recipe_details import RecipeRunDetail
from zcu_tools.mcp.measure.recipe_host import RecipeHost, RecipeHostFactory
from zcu_tools.mcp.measure.recipe_runtime import PendingAnalysis
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

logger = logging.getLogger(__name__)
SCRIPT_TERMINAL = frozenset({"completed", "cancelled", "failed", "needs_parameters"})


@dataclass(frozen=True)
class ScriptRecipeError:
    """Original unhandled failure; type is its class name, message is its diagnostic."""

    type: str
    message: str


@dataclass(frozen=True)
class ScriptPendingQuestion:
    """An explicit handoff, with evidence separate from response metadata.

    request_id uniquely identifies this ask. question_type is its authored class
    name. question retains evidence; answer_schema describes only reply fields.
    """

    request_id: str
    question_type: str
    question: Question[Answer]
    answer_schema: dict[str, JsonValue]


@dataclass(frozen=True)
class ScriptRecipeSnapshot:
    """Immutable local progress, without driving author code or sending RPCs.

    execution is session-local; recipe is its declaration name. status follows
    the shared scheduler; phase separates preparation/waiting/paused/terminal.
    pending_question exists only at explicit ask. pending_analysis identifies
    the GUI-owned interactive operation while status is interactive; no second
    runner is started. outcome exists at terminal;
    cancelled outcomes suppress report, retaining all runs and any error.
    cancel_requested records script-stop intent, not completion of GUI cleanup.
    details retains detached per-Run conditions, analyses and explicit proposals.
    missing lists authored parameter paths only at needs_parameters terminal;
    that state has no outcome/error and releases registry admission.
    """

    execution: str
    recipe: str
    status: ScriptStatus | Literal["needs_parameters", "interactive"] = "running"
    phase: Literal["preparing", "waiting", "paused", "terminal"] = "preparing"
    pending_question: ScriptPendingQuestion | None = None
    pending_analysis: PendingAnalysis | None = None
    outcome: RecipeOutcome[BaseModel] | None = None
    error: ScriptRecipeError | None = None
    cancel_requested: bool = False
    missing: tuple[str, ...] = ()
    details: tuple[RecipeRunDetail, ...] = ()


class ScriptRecipeExecution:
    """Own one author worker and the production shared async host.

    Inputs validate before the factory binds a GUI. Controls are acknowledged by
    the scheduler, never by running author code on the caller thread. Teardown
    cooperatively stops and drains operations; finally remains on this worker.
    """

    def __init__(
        self,
        tools: MeasureToolContext,
        definition: RecipeDeclaration,
        arguments: Mapping[str, object],
        *,
        host_factory: RecipeHostFactory,
    ) -> None:
        self._tools = tools
        self._definition = definition
        self._arguments = deepcopy(dict(arguments))
        self._host_factory = host_factory
        self._host: RecipeHost | None = None
        self._condition = Condition()
        self._cancelled = Event()
        self._progress = ScriptRecipeSnapshot(f"recipe-{uuid4()}", definition.name)
        self._worker = Thread(target=self._run, daemon=True)
        try:
            self._worker.start()
        except (RuntimeError, OSError) as error:
            self._finish(None, error)

    def snapshot(self) -> ScriptRecipeSnapshot:
        """Read detached progress; report identity is retained, questions are copied."""
        with self._condition:
            return replace(
                self._progress,
                pending_question=deepcopy(self._progress.pending_question),
                details=self._host.runtime.details() if self._host is not None else (),
            )

    def wait(self, timeout: float) -> ToolReply:
        """Wait up to 300 seconds for an explicit handoff or terminal result.

        timeout must be finite/nonnegative. Expiry only returns current progress;
        it never cancels work, supplies an answer, or reconnects to the GUI.
        """
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        began = time.monotonic()
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._progress.status in SCRIPT_TERMINAL
                    or self._progress.status in ("paused", "interactive")
                ),
                min(timeout, 300.0),
            )
            host = self._host
            return ToolReply(
                {**asdict(self.snapshot()), "elapsed_s": time.monotonic() - began},
                images=host.runtime.images() if host is not None else (),
                is_error=self._progress.status == "failed",
            )

    def answer(self, mapping: Mapping[str, object]) -> ToolReply:
        """Deliver {request_id, answer} after validating the exact pending question.

        Late/duplicate IDs and absent questions return accepted=False, reason
        and pending_request_id, without consuming a question. Invalid fields or
        writeback choices raise and preserve the request. The shared host owns
        validation and delivery. Accepted answers return at the next handoff or
        terminal, without author-thread reentry.
        """
        if set(mapping) != {"request_id", "answer"}:
            raise ValueError("Answers require exactly request_id and answer")
        request_id = TypeAdapter(str).validate_python(mapping["request_id"])
        if not request_id:
            raise ValueError("request_id must be nonempty")
        answer = TypeAdapter(dict[str, object]).validate_python(mapping["answer"])
        with self._condition:
            host = self._host
            terminal = self._progress.status in SCRIPT_TERMINAL
        if host is None or terminal or host.scheduler.pending_question is None:
            return self._rejected_answer(None)

        def deliver() -> ToolReply | None:
            pending = host.scheduler.pending_question
            if pending is None or pending.request_id != request_id:
                return self._rejected_answer(pending.request_id if pending else None)
            host.scheduler.answer(request_id, answer)
            return None

        try:
            rejected = host.scheduler.control(deliver)
        except RuntimeError:
            if host.scheduler.status in SCRIPT_TERMINAL:
                return self._rejected_answer(None)
            raise
        if rejected is not None:
            return rejected
        # The acknowledgement precedes author continuation. Wait must not return
        # the old question while the loop is about to clear its Future.
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._progress.pending_question is None
                    or self._progress.pending_question.request_id != request_id
                )
            )
        return self.wait(300.0)

    @staticmethod
    def _rejected_answer(pending_request_id: str | None) -> ToolReply:
        return ToolReply(
            {
                "accepted": False,
                "reason": "No pending question for this request id",
                "pending_request_id": pending_request_id,
            }
        )

    def cancel(self, *, scope: CancelScope = "script") -> ToolReply:
        """Acknowledge script or current Run stop, without fabricating completion.

        Script cancellation is retained even during initial binding; run scope is
        a no-op without an active operation. Subsequent operations still start,
        inheriting stop only for script scope. Terminal progress is unchanged.
        """
        if scope not in ("script", "run"):
            raise ValueError("scope must be script or run")
        with self._condition:
            if self._progress.status in SCRIPT_TERMINAL:
                if scope == "run":
                    return ToolReply(
                        {
                            **asdict(self.snapshot()),
                            "cancelled": False,
                            "reason": "no run in progress",
                        }
                    )
                return ToolReply(asdict(self.snapshot()))
            if scope == "script":
                self._cancelled.set()
                self._progress = replace(self._progress, cancel_requested=True)
            host = self._host
        if scope == "run":
            cancelled = host.runtime.cancel_run() if host is not None else False
            return ToolReply(
                {
                    **asdict(self.snapshot()),
                    "cancelled": cancelled,
                    "reason": "run cancellation requested"
                    if cancelled
                    else "no run in progress",
                }
            )
        if host is not None:
            host.scheduler.cancel(scope=scope)
        return ToolReply(asdict(self.snapshot()))

    def close(self) -> None:
        """Request cooperative script cancellation; join drains the original worker."""
        self.cancel()

    def join(self) -> None:
        """Join the author worker and its executor drain, avoiding recursive join."""
        if self._worker.ident is not None and self._worker is not current_thread():
            self._worker.join()

    def _changed(self) -> None:
        host = self._host
        if host is None:
            return
        request = host.scheduler.pending_question
        pending = (
            ScriptPendingQuestion(
                request.request_id,
                type(request.question).__name__,
                request.question,
                request.answer_type.model_json_schema(),
            )
            if request is not None
            else None
        )
        analysis = host.runtime.pending_analysis
        with self._condition:
            if self._progress.status not in SCRIPT_TERMINAL:
                self._progress = replace(
                    self._progress,
                    status="paused"
                    if pending
                    else "interactive"
                    if analysis
                    else "running",
                    pending_analysis=analysis,
                    phase="paused" if pending else "waiting",
                    pending_question=pending,
                )
                self._condition.notify_all()

    def _run(self) -> None:
        report: BaseModel | None = None
        error: Exception | None = None
        try:
            inputs = self._definition.inputs.model_validate(self._arguments)
            host = self._host_factory(self._tools, self._changed)
            with self._condition:
                self._host = host
            if self._cancelled.is_set():
                host.scheduler.cancel()
            report = host.scheduler.run(
                self._definition.coroutine(host.session, inputs)
            )
        except RecipeCancelled as failure:
            error = failure
        except RecipeNeedsParameters as failure:
            error = failure
        except Exception as failure:
            logger.exception("Script recipe execution failed")
            error = failure
        self._finish(report, error)

    def _finish(self, report: BaseModel | None, error: Exception | None) -> None:
        host = self._host
        cancelled = (
            isinstance(error, RecipeCancelled)
            or self._cancelled.is_set()
            or (host is not None and host.scheduler.status == "cancelled")
        )
        status = (
            "cancelled" if cancelled else "failed" if error is not None else "completed"
        )
        runs = host.runtime.runs() if host is not None else ()
        if isinstance(error, RecipeNeedsParameters) and not cancelled:
            with self._condition:
                self._progress = replace(
                    self._progress,
                    status="needs_parameters",
                    phase="terminal",
                    pending_question=None,
                    missing=tuple(item.parameter for item in error.missing),
                    outcome=None,
                    error=None,
                )
                self._condition.notify_all()
            return
        with self._condition:
            self._progress = replace(
                self._progress,
                status=status,
                phase="terminal",
                pending_question=None,
                outcome=RecipeOutcome(status, None if cancelled else report, runs),
                error=ScriptRecipeError(type(error).__name__, str(error))
                if error is not None and not isinstance(error, RecipeCancelled)
                else None,
            )
            self._condition.notify_all()
