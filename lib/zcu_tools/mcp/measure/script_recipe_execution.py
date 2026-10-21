"""Worker lifetime and detached progress for declarations using ScriptDriver."""

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

from pydantic import BaseModel, JsonValue

from zcu_tools.core.script_protocol import Answer, Operation, Question, Signal, Wait
from zcu_tools.experiment.script_driver import ScriptDriver, ScriptStatus
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.recipe_api import (
    RecipeDeclaration,
    RecipeOutcome,
)
from zcu_tools.mcp.measure.recipe_host import RecipeHost, RecipeHostFactory
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

logger = logging.getLogger(__name__)
SCRIPT_TERMINAL = frozenset({"completed", "cancelled", "failed"})


@dataclass(frozen=True)
class ScriptRecipeError:
    """The original failure's class name and diagnostic, without invented status."""

    type: str
    message: str


@dataclass(frozen=True)
class ScriptPendingQuestion:
    """Asked evidence and its discoverable reply contract, in separate namespaces.

    question_type is the authored class name. question retains its named fields;
    answer_schema is that Question's declared Answer JSON schema, including field
    descriptions. Authored names never collide with these envelope metadata.
    """

    question_type: str
    question: Question[Answer]
    answer_schema: dict[str, JsonValue]


@dataclass(frozen=True)
class ScriptRecipeSnapshot:
    """Local execution facts; reading never drives code or sends GUI requests.

    execution is session-local, recipe is the injected tool name. status follows
    ScriptDriver, phase distinguishes preparation from yielded work and terminal.
    pending_question is only the explicit unanswered Ask, never hidden analysis.
    outcome exists only at terminal and preserves the authored report object.
    error retains an uncaught host/script failure. cancel_requested records intent.
    """

    execution: str
    recipe: str
    status: ScriptStatus = "running"
    phase: Literal["preparing", "waiting", "paused", "terminal"] = "preparing"
    pending_question: ScriptPendingQuestion | None = None
    outcome: RecipeOutcome[BaseModel] | None = None
    error: ScriptRecipeError | None = None
    cancel_requested: bool = False


class ScriptRecipeExecution:
    """Own one shared driver and worker, admitted by RecipeExecutions.

    Inputs validate before host_factory binds a session. Only this worker drives
    authored code. Public control/answer calls serialize through the condition;
    invalid answers never reserve or resume. cancel drains active work then
    resumes author-owned cleanup; close abandons it for host teardown. join drains
    the worker, including the host-owned pending operation. No timeout
    supplies a default answer or cancels work.
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
        self._driver: ScriptDriver[BaseModel | None] | None = None
        self._condition = Condition()
        self._cancelled = Event()
        self._closed = Event()
        self._active: Operation[object] | None = None
        self._answer: Answer | None = None
        self._progress = ScriptRecipeSnapshot(f"recipe-{uuid4()}", definition.name)
        self._worker = Thread(target=self._run, daemon=True)
        try:
            self._worker.start()
        except (RuntimeError, OSError) as error:
            logger.exception("Script recipe worker could not start")
            self._fail(error)

    def snapshot(self) -> ScriptRecipeSnapshot:
        """Read immutable local progress, retaining the authored report identity."""
        with self._condition:
            return replace(
                self._progress,
                pending_question=deepcopy(self._progress.pending_question),
            )

    def wait(self, timeout: float) -> ToolReply:
        """Wait at most 300 seconds for an unanswered question or terminal outcome.

        timeout must be finite and nonnegative. A timeout returns current progress
        without changing the generator or its operation. Failure marks is_error.
        """
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        began = time.monotonic()
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._progress.status in SCRIPT_TERMINAL
                    or self._progress.status == "paused"
                ),
                min(timeout, 300.0),
            )
            return ToolReply(
                {**asdict(self._progress), "elapsed_s": time.monotonic() - began},
                is_error=self._progress.status == "failed",
            )

    def answer(self, mapping: Mapping[str, object]) -> ToolReply:
        """Validate named wire fields, reserve a pending Ask, then wait for handoff.

        Parsing uses ScriptDriver's single Question/Answer inference owner.
        Invalid answers leave progress paused; duplicate/non-question calls raise
        RuntimeError. Authored continuation runs only on the original worker.
        """
        with self._condition:
            driver = self._driver
            if driver is None or self._progress.status != "paused":
                raise RuntimeError("No pending question to answer")
            self._answer = driver.parse_answer(mapping)
            self._progress = replace(
                self._progress,
                status="running",
                pending_question=None,
                phase="preparing",
            )
            self._condition.notify_all()
        return self.wait(300.0)

    def cancel(self) -> ToolReply:
        """Request cancel once; drain active work before resuming author cleanup.

        Terminal executions are unchanged. The reply records intent, not a
        fabricated terminal result. Host cancellation errors propagate.
        """
        with self._condition:
            if (
                self._progress.status not in SCRIPT_TERMINAL
                and not self._cancelled.is_set()
            ):
                self._cancelled.set()
                self._progress = replace(self._progress, cancel_requested=True)
                try:
                    if self._active is not None and self._host is not None:
                        self._host.cancel(self._active)
                finally:
                    self._condition.notify_all()
            return ToolReply(asdict(self._progress))

    def finish_early(self) -> ToolReply:
        """Reject early finish until a concrete operation host defines that policy."""
        raise ValueError("Script recipe finish_early is not bound")

    def close(self) -> None:
        """Abandon author continuation for teardown; call join to drain it.

        Active work is cancelled and drained before worker-owned driver.close.
        Finally runs without promising a response to any further yield.
        """
        with self._condition:
            self._closed.set()
            self.cancel()

    def join(self) -> None:
        """Drain the worker; never join recursively from authored code."""
        if self._worker.ident is not None and self._worker is not current_thread():
            self._worker.join()

    def _publish(self, driver: ScriptDriver[BaseModel | None]) -> None:
        question = driver.pending_question
        answer = driver.pending_answer_type()
        pending = (
            ScriptPendingQuestion(
                type(question).__name__, question, answer.model_json_schema()
            )
            if question is not None and answer is not None
            else None
        )
        with self._condition:
            status = driver.status
            self._progress = replace(
                self._progress,
                status=status,
                phase="terminal"
                if status in SCRIPT_TERMINAL
                else "paused"
                if status == "paused"
                else "waiting"
                if status == "waiting"
                else "preparing",
                pending_question=pending,
            )
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            inputs = self._definition.inputs.model_validate(self._arguments)
            host = self._host_factory(self._tools)
            driver = ScriptDriver(
                self._definition.generator(host.session, inputs),
                on_signal=host.on_signal,
                deliver_cancelled=host.deliver_cancelled,
            )
            with self._condition:
                self._host = host
                self._driver = driver
            self._drive(driver, host)
            status = driver.status
            if status not in ("completed", "cancelled", "failed"):
                raise RuntimeError("Script driver did not reach terminal")
            runs = host.runs()
            with self._condition:
                self._progress = replace(
                    self._progress,
                    status=status,
                    phase="terminal",
                    pending_question=None,
                    outcome=RecipeOutcome(status, driver.result, runs),
                )
                self._condition.notify_all()
        except Exception as error:
            logger.exception("Script recipe execution failed")
            self._close_failed_driver(error)
            self._fail(error)

    def _close_failed_driver(self, error: Exception) -> None:
        # A host failure happens outside driver.advance; close only after the
        # host's wait has drained, before publishing terminal/admitting new work.
        driver = self._driver
        if driver is None or driver.status in ("completed", "cancelled"):
            return
        try:
            driver.close()
        except Exception as cleanup_error:
            logger.exception("Script cleanup also failed")
            error.add_note(f"Script cleanup failed: {cleanup_error}")

    def _drive(self, driver: ScriptDriver[BaseModel | None], host: RecipeHost) -> None:
        if self._closed.is_set():
            driver.close()
            return
        if self._cancelled.is_set():
            driver.signal(Signal("cancel"))
        driver.advance()
        while driver.status not in SCRIPT_TERMINAL:
            self._publish(driver)
            request = driver.pending_request
            if isinstance(request, Wait):
                with self._condition:
                    self._active = request.op
                    if self._cancelled.is_set():
                        host.cancel(request.op)
                try:
                    host.wait(request.op, self._cancelled)
                    if request.op.delivery is None:
                        raise RuntimeError(
                            "Recipe host wait returned without terminal delivery"
                        )
                finally:
                    with self._condition:
                        self._active = None
            elif driver.status == "paused":
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._answer is not None or self._cancelled.is_set()
                    )
                    answer = self._answer
                    self._answer = None
                if answer is not None and not self._cancelled.is_set():
                    driver.answer(answer)
                    continue
            if self._closed.is_set():
                driver.close()
            else:
                if self._cancelled.is_set():
                    driver.signal(Signal("cancel"))
                driver.advance()

    def _fail(self, error: Exception) -> None:
        host = self._host
        runs = ()
        if host is not None:
            try:
                runs = host.runs()
            except Exception as capture_error:
                logger.exception("Could not capture recipe acquisition history")
                error.add_note(f"Run capture failed: {capture_error}")
        with self._condition:
            self._progress = replace(
                self._progress,
                status="failed",
                phase="terminal",
                pending_question=None,
                outcome=RecipeOutcome("failed", None, runs),
                error=ScriptRecipeError(type(error).__name__, str(error)),
            )
            self._condition.notify_all()
