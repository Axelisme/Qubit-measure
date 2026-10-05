"""Generator lifetime, local snapshots and single-recipe admission.

Native completion and cfg ownership stay behind recipe.py's opaque handles.
The session injects definitions; this owner has no global recipe imports.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from threading import Condition, Event, Lock, Thread, current_thread
from types import GeneratorType
from typing import Literal
from uuid import uuid4

from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.analysis_execution import (
    AnalysisStart,
    AnalysisWriteback,
    CancelError,
    ExecutionSnapshot,
    GuiCancel,
)
from zcu_tools.mcp.measure.raw_save import RawSaveReceipt
from zcu_tools.mcp.measure.recipe import (
    AnalysisStage,
    AnalyzeOperation,
    MissingParameter,
    RecipeDefinition,
    RecipeDelivery,
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeOperation,
    RecipeSession,
    RecipeWritebackPreview,
    RunOperation,
    RunPreview,
    StepStatus,
    WritebackDecision,
    WritebackQuestion,
    WritebackReceipt,
)
from zcu_tools.mcp.measure.recipe_capture import RecipeActual
from zcu_tools.mcp.measure.recipe_inputs import RecipeInputs
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

logger = logging.getLogger(__name__)

RecipeStatus = Literal[
    "running",
    "interactive",
    "awaiting_answer",
    "needs_parameters",
    "finished",
    "failed",
    "cancelled",
]
RecipePhase = Literal[
    "preparing",
    "run",
    "raw_save",
    "analysis",
    "writeback_read",
    "preview",
    "awaiting_answer",
    "terminal",
]
TERMINAL_STATUSES = frozenset({"needs_parameters", "finished", "failed", "cancelled"})


@dataclass(frozen=True)
class RecipeError:
    """Failure capture: phase attempted, machine reason, message and optional wire code."""

    phase: RecipePhase
    reason: str
    message: str
    code: str | None = None


@dataclass(frozen=True)
class RecipeSnapshot:
    """Detached recipe progress; queries never send RPCs or refresh GUI guards.

    execution is a session-local ID; recipe is its injected tool name.
    tab is the last Run's locator, or the prepared/requested locator before Run.
    A requested locator alone does not certify GUI existence/readiness.
    status is running/interactive/awaiting_answer, or a terminal status.
    phase is the active yield, preparing between yields, or terminal.
    cancel_requested is one unconsumed stop intent, not a native outcome.
    finish_early_requested records the active Run's early-stop intent.
    run_op is the last Run handle; op is the current operation/save handle.
    actual is the last Run's fixed conditions, or None before capture.
    missing contains needs_parameters prompts, empty for other outcomes.
    run_outcome/result_state retain original native completion and provenance.
    run_start retains not_started/unknown/running receipt and rejection reason.
    raw_save retains the latest save and confirmed prefix.
    preview is the captured session Run image, or None when unavailable.
    analysis_mode identifies captured stages, not requested future steps.
    analysis/post_analysis retain each stage's latest completion/start capture.
    writeback/post_writeback are those stages' captured proposals, or None.
    analysis_stage is the latest attempted stage; analysis_starts records receipts.
    question_items/question_preview are the active question, otherwise None.
    written contains ordered actual write receipts, including partial failures.
    error is an uncaught generator/completion failure, otherwise None.
    """

    execution: str
    recipe: str
    tab: str | None = None
    status: RecipeStatus = "running"
    phase: RecipePhase = "preparing"
    cancel_requested: bool = False
    finish_early_requested: bool = False
    run_op: int | None = None
    op: int | None = None
    actual: RecipeActual | None = None
    missing: tuple[MissingParameter, ...] = ()
    run_outcome: Mapping[str, object] | None = None
    run_start: AnalysisStart = field(default_factory=AnalysisStart)
    result_state: Mapping[str, object] | None = None
    raw_save: RawSaveReceipt = field(default_factory=RawSaveReceipt)
    analysis_mode: Literal["primary", "primary_post", "none"] = "none"
    preview: RunPreview | None = None
    analysis: ExecutionSnapshot | None = None
    writeback: AnalysisWriteback | None = None
    post_analysis: ExecutionSnapshot | None = None
    post_writeback: AnalysisWriteback | None = None
    analysis_stage: AnalysisStage | None = None
    analysis_starts: Mapping[AnalysisStage, AnalysisStart] = field(
        default_factory=lambda: {"primary": AnalysisStart(), "post": AnalysisStart()}
    )
    question_items: tuple[str, ...] | None = None
    question_preview: RecipeWritebackPreview | None = None
    written: tuple[WritebackReceipt, ...] = ()
    error: RecipeError | None = None


class RecipeExecution:
    """One generator and its sole worker, owned by RecipeExecutions.

    tools is the owning session's fixed tool context. definition is the validated
    registry declaration; arguments is the caller's detached keyword mapping,
    checked against inputs before GUI binding. session_closed is the registry's
    close event.
    Construction starts background execution; a worker-start failure is retained
    as failed progress without sending GUI operations. Use the registry's start
    method for validated definitions and exclusion. inputs owns that definition's
    argument codec. Keyword errors become a failed preparing capture before GUI
    binding; direct construction does not enforce registry exclusion.
    """

    def __init__(
        self,
        tools: MeasureToolContext,
        definition: RecipeDefinition,
        arguments: Mapping[str, object],
        session_closed: Event,
        *,
        inputs: RecipeInputs,
    ) -> None:
        self._tools = tools
        self._definition = definition
        self._arguments = deepcopy(dict(arguments))
        self._inputs = inputs
        self._session_closed = session_closed
        self._closed = Event()
        self._condition = Condition()
        self._progress = RecipeSnapshot(f"recipe-{uuid4()}", definition.name)
        self._session: RecipeSession | None = None
        self._active: RecipeOperation | None = None
        self._answer: WritebackDecision | None = None
        self._pending_cancel = False
        self._last_status: StepStatus = "completed"
        self._analysis_images: dict[AnalysisStage, tuple[PngImage, ...]] = {}
        self._stopped_operation: RunOperation | None = None
        self._run_cancel = GuiCancel("not_needed")
        self._worker = Thread(target=self._run, daemon=True)
        try:
            self._worker.start()
        except (RuntimeError, OSError) as error:
            logger.exception("Recipe worker could not start")
            self._progress = replace(
                self._progress,
                status="failed",
                phase="terminal",
                error=RecipeError(
                    "preparing", "recipe_worker_start_failed", str(error)
                ),
            )

    def snapshot(self) -> RecipeSnapshot:
        """Return detached local facts, including unyielded start failures.

        A pending interactive handoff is reported only while its original owner
        is interactive. A late done never re-delivers a stale interaction.
        """
        with self._condition:
            progress = self._progress
            session = self._session
            analysis: ExecutionSnapshot | None = None
            if session is not None:
                progress = replace(
                    progress,
                    tab=session.tab_locator(),
                    written=session.writeback_receipts(),
                )
                run = session.run_snapshot()
                if run is not None:
                    progress = replace(
                        progress,
                        tab=run.tab,
                        actual=run.actual,
                        run_op=run.op,
                        run_start=AnalysisStart(run.start_status, run.start_reason),
                        run_outcome=run.outcome,
                        result_state=run.result_state,
                        raw_save=run.raw_save,
                        preview=run.preview,
                        op=run.save_op if self._active is None else progress.op,
                    )
                    if (
                        run.raw_save.status in ("saving", "failed", "unknown")
                        and self._active is None
                        and progress.phase == "preparing"
                    ):
                        progress = replace(progress, phase="raw_save")
                    if (
                        run.preview_status != "not_started"
                        and self._active is None
                        and progress.phase == "preparing"
                    ):
                        progress = replace(progress, phase="preview")
                    for stage, capture in run.analyses.items():
                        progress = self._retain_analysis(progress, stage, capture)
                analysis = session.analysis_snapshot()
                if analysis is not None:
                    progress = self._retain_analysis(progress, analysis.stage, analysis)
            capture = (
                self._active.snapshot()
                if isinstance(self._active, AnalyzeOperation)
                else analysis
            )
            if (
                capture is not None
                and progress.status not in TERMINAL_STATUSES
                and (
                    isinstance(self._active, AnalyzeOperation)
                    or (
                        self._active is None
                        and progress.phase == "preparing"
                        and (
                            capture.error is not None
                            or capture.start.status == "unknown"
                        )
                    )
                )
            ):
                phase = (
                    capture.error.phase if capture.error is not None else capture.phase
                )
                progress = replace(
                    progress,
                    status="interactive"
                    if capture.status == "interactive"
                    else "running",
                    op=capture.op,
                    phase="writeback_read"
                    if phase == "writeback_read"
                    else "preview"
                    if phase == "figure_read"
                    else "analysis",
                )
            return deepcopy(progress)

    def wait(self, timeout: float) -> ToolReply:
        """Wait locally for terminal/interaction/question, at most 300 seconds.

        timeout is finite and nonnegative; zero only reads local capture.
        Timeout does not cancel or stop background execution. Return native
        captured data and previews; only failed execution marks is_error.
        """
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        began = time.monotonic()
        with self._condition:
            self._condition.wait_for(
                lambda: self.snapshot().status != "running",
                min(timeout, 300.0),
            )
            progress = self.snapshot()
            analysis_images = dict(self._analysis_images)
            if isinstance(self._active, AnalyzeOperation):
                capture = self._active.snapshot()
                if capture is not None:
                    analysis_images[capture.stage] = self._active.preview_images()
            images = tuple(
                image
                for stage in ("primary", "post")
                for image in analysis_images.get(stage, ())
            )
            if not images and self._session is not None:
                run = self._session.run_snapshot()
                if run is not None:
                    images = run.preview_images
            return ToolReply(
                {**asdict(progress), "elapsed_s": time.monotonic() - began},
                images,
                is_error=progress.status == "failed"
                or (
                    progress.status == "interactive"
                    and any(
                        capture is not None
                        and capture.interaction is not None
                        and "delivery_error" in capture.interaction
                        for capture in (progress.analysis, progress.post_analysis)
                    )
                ),
            )

    def cancel(self) -> ToolReply:
        """Consume one stop intent at the active yield or next Run/analysis.

        Fast cfg/save steps remain available. An admitted native operation keeps
        its true outcome; gui_cancel is a separate control receipt, never proof
        of cancellation. Terminal calls are harmless and do not rerun finally.
        """
        with self._condition:
            if self._progress.status in TERMINAL_STATUSES:
                return self._control_reply(GuiCancel("not_needed"))
            self._pending_cancel = True
            self._progress = replace(
                self._progress,
                cancel_requested=True,
                finish_early_requested=False,
                status="running",
            )
            active = self._active
            self._condition.notify_all()
        result = self._cancel_active(active)
        return self._control_reply(result)

    def finish_early(self) -> ToolReply:
        """Request cooperative stop of an active Run; reject every other yield.

        A native failed outcome still throws. Cancel overrides this intent.
        The recipe receives finished_early after a successful native stop.
        """
        with self._condition:
            active = self._active
            if not isinstance(active, RunOperation) or active.snapshot() is None:
                raise GuiRpcError(
                    "finish_early requires an active Run", reason="not_running"
                )
            self._progress = replace(self._progress, finish_early_requested=True)
            self._condition.notify_all()
        return self._control_reply(self._cancel_active(active))

    def answer(self, decision: WritebackDecision) -> ToolReply:
        """Answer the active captured question without applying writeback.

        decision is accepted or skipped. Unknown values raise ValueError; calls
        without an unanswered question raise GuiRpcError. The worker resumes
        independently; this method waits at most 300 seconds for its next handoff.
        """
        if decision not in ("accepted", "skipped"):
            raise ValueError("decision must be accepted or skipped")
        with self._condition:
            if (
                self._progress.status != "awaiting_answer"
                or self._answer is not None
                or self._pending_cancel
            ):
                raise GuiRpcError(
                    "Recipe has no unanswered question", reason="not_awaiting_answer"
                )
            self._answer = decision
            self._progress = replace(self._progress, status="running")
            self._condition.notify_all()
        return self.wait(300.0)

    def close(self) -> None:
        """Signal stop; the sole worker serializes generator.close/finally.

        Idempotent and nonblocking. Does not cancel hardware or close the GUI.
        Call join to drain finally before deleting preview PNGs. Native late
        results never send into the closed generator.
        """
        with self._condition:
            self._closed.set()
            active = self._active
            self._condition.notify_all()
        if isinstance(active, AnalyzeOperation):
            active.wake()

    def join(self) -> None:
        """Join this worker after close; never close a GUI or delete preview files."""
        if self._worker.ident is not None and current_thread() is not self._worker:
            self._worker.join()

    def _retain_analysis(
        self, progress: RecipeSnapshot, stage: AnalysisStage, capture: ExecutionSnapshot
    ) -> RecipeSnapshot:
        starts = dict(progress.analysis_starts)
        starts[stage] = capture.start
        progress = replace(progress, analysis_stage=stage, analysis_starts=starts)
        if stage == "primary":
            return replace(
                progress,
                analysis=capture,
                writeback=capture.writeback,
                analysis_mode="primary_post" if progress.post_analysis else "primary",
            )
        return replace(
            progress,
            post_analysis=capture,
            post_writeback=capture.writeback,
            analysis_mode="primary_post",
        )

    def _consume_cancel(self) -> bool:
        with self._condition:
            consumed = self._pending_cancel
            self._pending_cancel = False
            if consumed:
                self._progress = replace(self._progress, cancel_requested=False)
            return consumed

    def _cancel_active(self, active: RecipeOperation | None) -> GuiCancel | ToolReply:
        try:
            if isinstance(active, AnalyzeOperation):
                reply = active.cancel()
                if reply is None:
                    return GuiCancel("not_needed")
                return reply
            if isinstance(active, RunOperation):
                capture = active.snapshot()
                if capture is not None and capture.op is not None:
                    with self._condition:
                        if self._stopped_operation is active:
                            return self._run_cancel
                        # Claim one cooperative stop before waiting for the RPC lock.
                        self._stopped_operation = active
                        self._run_cancel = GuiCancel("requested")
                    reply = self._tools.gui.read_internal(
                        "operation.cancel", {}, operation_handle=capture.op
                    )
                    if reply.get("status") == "cancelling":
                        result = GuiCancel("requested")
                    elif reply.get("status") in ("finished", "cancelled"):
                        result = GuiCancel("not_needed")
                    else:
                        raise GuiRpcError(
                            "Invalid cancel reply", reason="incompatible_wire"
                        )
                    with self._condition:
                        self._run_cancel = result
                    return result
            return GuiCancel("not_needed")
        except (
            Exception
        ) as error:  # Isolate control failure without rewriting native outcome.
            logger.exception("Recipe stop request failed")
            reason = error.reason if isinstance(error, GuiRpcError) else None
            code = error.code if isinstance(error, GuiRpcError) else None
            result = GuiCancel(
                "failed", CancelError(reason or "cancel_failed", str(error), code)
            )
            with self._condition:
                self._run_cancel = result
            return result

    def _control_reply(self, result: GuiCancel | ToolReply) -> ToolReply:
        reply = self.wait(0.0)
        control = (
            result.data["gui_cancel"]
            if isinstance(result, ToolReply)
            else asdict(result)
        )
        failed = (
            result.is_error
            if isinstance(result, ToolReply)
            else result.status == "failed"
        )
        return ToolReply(
            {**reply.data, "gui_cancel": control},
            reply.images,
            is_error=reply.is_error or failed,
        )

    def _check_open(self) -> None:
        if self._closed.is_set() or self._session_closed.is_set():
            raise GuiRpcError("MCP session is closed", reason="session_closed")

    def _complete(self, operation: RecipeOperation) -> RecipeDelivery:
        with self._condition:
            self._active = operation
            phase: RecipePhase = (
                "run"
                if isinstance(operation, RunOperation)
                else "analysis"
                if isinstance(operation, AnalyzeOperation)
                else "awaiting_answer"
            )
            self._progress = replace(self._progress, phase=phase, status="running")
            pending = self._pending_cancel
            self._condition.notify_all()
        if isinstance(operation, WritebackQuestion):
            question = operation.snapshot()
            with self._condition:
                self._progress = replace(
                    self._progress,
                    status="awaiting_answer",
                    question_items=question.items,
                    question_preview=question.preview,
                )
                self._condition.notify_all()
                self._condition.wait_for(
                    lambda: (
                        self._answer is not None
                        or self._pending_cancel
                        or self._closed.is_set()
                        or self._session_closed.is_set()
                    )
                )
                self._check_open()
                delivery: RecipeDelivery = (
                    (None, "cancelled")
                    if self._consume_cancel()
                    else (self._answer, "completed")
                )
                self._answer = None
                return delivery
        if pending:
            self._cancel_active(operation)
        if isinstance(operation, RunOperation):
            capture = operation.snapshot()
            with self._condition:
                self._progress = replace(
                    self._progress, op=capture.op if capture is not None else None
                )
            run, status = operation.complete_in_current_worker()
            with self._condition:
                if self._progress.finish_early_requested and not self._pending_cancel:
                    status = "finished_early"
            return run, status
        try:
            return operation.complete_in_current_worker()
        finally:
            # Keep each stage and its confirmed previews even when completion fails.
            capture = operation.snapshot()
            if capture is not None:
                with self._condition:
                    self._analysis_images[capture.stage] = operation.preview_images()

    def _advance(self, generator: RecipeGenerator) -> None:
        operation = next(generator)
        while True:
            self._check_open()
            try:
                delivery = self._complete(operation)
            except Exception as error:  # Throw operation failure at the author's yield.
                logger.exception("Recipe operation failed; delivering to generator")
                self._check_open()
                with self._condition:
                    self._progress = replace(
                        self._progress, phase=self.snapshot().phase
                    )
                    self._active = None
                    self._condition.notify_all()
                operation = generator.throw(error)
                continue
            self._check_open()
            with self._condition:
                self._active = None
                self._last_status = delivery[1]
                self._pending_cancel = False
                self._progress = replace(
                    self._progress,
                    status="running",
                    phase="preparing",
                    op=None,
                    cancel_requested=False,
                    finish_early_requested=False,
                    question_items=None,
                    question_preview=None,
                )
                self._condition.notify_all()
            operation = generator.send(delivery)

    def _run(self) -> None:
        generator: RecipeGenerator | None = None
        terminal: RecipeStatus = "finished"
        try:
            self._check_open()
            arguments = self._inputs.normalize(self._arguments)
            tools = self._tools.bound()
            session = RecipeSession(
                tools,
                closed=self._closed,
                condition=self._condition,
                consume_cancel=self._consume_cancel,
            )
            with self._condition:
                self._tools = tools
                self._session = session
            generator = self._definition.run(session, **arguments)
            if not isinstance(generator, GeneratorType):
                raise TypeError("Recipe must return a generator")
            try:
                self._advance(generator)
            except StopIteration:
                terminal = (
                    "cancelled" if self._last_status == "cancelled" else "finished"
                )
        except RecipeNeedsParameters as error:
            terminal = "needs_parameters"
            with self._condition:
                self._progress = replace(self._progress, missing=error.missing)
        except Exception as error:  # Isolate generator failures with captured facts.
            logger.exception("Recipe execution failed")
            if self._closed.is_set() or self._session_closed.is_set():
                terminal = "cancelled"
            else:
                terminal = "failed"
                reason = error.reason if isinstance(error, GuiRpcError) else None
                code = error.code if isinstance(error, GuiRpcError) else None
                with self._condition:
                    self._progress = replace(
                        self._progress,
                        error=RecipeError(
                            self.snapshot().phase,
                            reason or "recipe_failed",
                            str(error),
                            code,
                        ),
                    )
        finally:
            if generator is not None:
                try:
                    generator.close()
                except (
                    Exception
                ) as error:  # Retain teardown failure and drain its worker.
                    logger.exception("Recipe generator close failed")
                    terminal = "failed"
                    with self._condition:
                        self._progress = replace(
                            self._progress,
                            error=RecipeError(
                                self._progress.phase, "recipe_close_failed", str(error)
                            ),
                        )
            with self._condition:
                self._active = None
                self._progress = replace(
                    self._progress,
                    status=terminal,
                    phase="terminal",
                    question_items=None,
                    question_preview=None,
                )
                self._condition.notify_all()


class RecipeExecutions:
    """Session-local injected definitions and admission for one active generator.

    closed is the owning session's lifetime event. recipes supplies handwritten
    definitions; empty/duplicate names, missing callables and invalid schemas
    raise ValueError during construction, before any GUI operation.
    All executions use the first admitted tools context; replacements are refused.
    """

    def __init__(self, closed: Event, *, recipes: Sequence[RecipeDefinition]) -> None:
        self._closed = closed
        self._lock = Lock()
        self._definitions: dict[str, RecipeDefinition] = {}
        self._inputs: dict[str, RecipeInputs] = {}
        self._executions: dict[str, RecipeExecution] = {}
        self._tools: MeasureToolContext | None = None
        for definition in recipes:
            if (
                not definition.name.strip()
                or not definition.description.strip()
                or not definition.adapter_name.strip()
                or not callable(definition.run)
            ):
                raise ValueError(
                    "Recipe requires name, description, adapter and run callable"
                )
            if definition.name in self._definitions:
                raise ValueError(f"Duplicate recipe name: {definition.name!r}")
            copied = deepcopy(definition)
            self._inputs[copied.name] = RecipeInputs(copied.input_schema)
            self._definitions[copied.name] = copied

    @property
    def definitions(self) -> tuple[RecipeDefinition, ...]:
        """Return detached admitted definitions in supplied order, without registration."""
        return deepcopy(tuple(self._definitions.values()))

    def start(
        self, tools: MeasureToolContext, recipe: str, arguments: Mapping[str, object]
    ) -> RecipeExecution:
        """Validate explicit keywords and start one generator, or fail before GUI work.

        Unknown names/closed registry/replacement sessions raise GuiRpcError.
        Invalid keywords become a failed preparing execution before GUI binding.
        A running, interactive or awaiting_answer recipe blocks admission with its
        ID/status. Terminal records, including argument failures, remain queryable.
        """
        definition = self._definitions.get(recipe)
        if definition is None:
            raise GuiRpcError(f"Unknown recipe: {recipe!r}", reason="unknown_recipe")
        with self._lock:
            if self._closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            if self._tools is not None and tools is not self._tools:
                raise GuiRpcError(
                    "Recipe tools are not the owning fixed context",
                    reason="wrong_binding",
                )
            for existing in self._executions.values():
                snapshot = existing.snapshot()
                if snapshot.status not in TERMINAL_STATUSES:
                    raise GuiRpcError(
                        f"Recipe {snapshot.execution} is {snapshot.status}",
                        reason="recipe_busy",
                    )
            self._tools = tools
            execution = RecipeExecution(
                tools, definition, arguments, self._closed, inputs=self._inputs[recipe]
            )
            self._executions[execution.snapshot().execution] = execution
            return execution

    def get(self, execution: str) -> RecipeExecution:
        """Read a session-local ID, or raise GuiRpcError(reason=unknown_execution)."""
        with self._lock:
            found = self._executions.get(execution)
        if found is None:
            raise GuiRpcError(
                f"Unknown execution: {execution!r}", reason="unknown_execution"
            )
        return found

    def for_op(self, op: int) -> RecipeExecution | None:
        """Find the recipe capturing this session operation, or return None.

        op is a positive opaque session handle, not a native GUI ID. Includes Run,
        current save and captured Primary/Post operations, without GUI requests.
        Terminal owners remain discoverable; their control methods are harmless.
        """
        if isinstance(op, bool) or op <= 0:
            raise ValueError("op must be a positive integer")
        with self._lock:
            executions = tuple(self._executions.values())
        for execution in executions:
            snapshot = execution.snapshot()
            analysis_ops = tuple(
                capture.op
                for capture in (snapshot.analysis, snapshot.post_analysis)
                if capture is not None
            )
            if op in (snapshot.run_op, snapshot.op, *analysis_ops):
                return execution
        return None

    def snapshots(self) -> tuple[RecipeSnapshot, ...]:
        """Read detached local captures in admission order, with no GUI requests."""
        with self._lock:
            executions = tuple(self._executions.values())
        return tuple(execution.snapshot() for execution in executions)

    def stop_admission(self) -> None:
        """Refuse new starts and signal all workers; join drains them separately."""
        self._closed.set()
        with self._lock:
            executions = tuple(self._executions.values())
        for execution in executions:
            execution.close()

    def join(self) -> None:
        """Drain stopped workers; safe repeatedly, never deletes preview files."""
        with self._lock:
            executions = tuple(self._executions.values())
        for execution in executions:
            execution.join()
