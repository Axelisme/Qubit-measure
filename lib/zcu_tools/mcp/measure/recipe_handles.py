"""Recipe operation completion, fixed Run handles and shared binding state.

The authoring owner prepares bindings; these handles retain their original Run,
analysis and writeback captures through cancellation, save and session close.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Generator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Condition, Event

from zcu_tools.mcp.core.images import validated_png
from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.analysis_execution import (
    AnalysisExecution,
    ExecutionSnapshot,
)
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.interaction import handoff_interaction
from zcu_tools.mcp.measure.operation_wait import await_operation
from zcu_tools.mcp.measure.raw_save import RawSaveRequest, SaveReport, save_raw_data
from zcu_tools.mcp.measure.recipe_capture import SweepSources
from zcu_tools.mcp.measure.recipe_contracts import (
    AnalysisStage,
    RecipeAnalysis,
    RecipeParameter,
    RecipeRunSnapshot,
    RecipeWritebackPreview,
    StepStatus,
    WritebackDecision,
    WritebackReceipt,
    recipe_gui_object,
)
from zcu_tools.mcp.measure.recipe_inputs import copy_recipe_parameters
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.writeback import WritebackSelection, select_writeback_items


@dataclass(frozen=True)
class RunStepResult:
    """Run delivery: run retains partial data, or is None for a suppressed start.

    status is completed, finished_early, or cancelled. Cancellation is returned to
    the author, who decides how to clean up and whether to return.
    """

    run: RecipeRun | None
    status: StepStatus


@dataclass(frozen=True)
class AnalysisStepResult:
    """Analysis delivery with completed/cancelled status and captured analysis.

    analysis is None when cancelled; otherwise it holds the completed capture.
    """

    analysis: RecipeAnalysis | None
    status: StepStatus


@dataclass(frozen=True)
class WritebackResult:
    """Answer delivery, without writing or authorizing any data.

    decision is accepted/skipped on completed status, or None on cancelled status.
    """

    decision: WritebackDecision | None
    status: StepStatus


class RunOperation:
    """Opaque, non-iterable Run handle created by RecipeTab.run.

    Use RecipeTab.run_step with yield from, not a direct yield of this handle.
    delivery is None until the driver stores the named RunStepResult. A
    between-step cancel delivers a result with no run, without sending Run.
    Failure or superseded source is thrown back at the yield expression.
    Construction is framework-only: binding is the owner-created Run state, or
    None for one suppressed start. Authors obtain handles from RecipeTab.run.
    """

    def __init__(self, binding: RecipeRunBinding | None) -> None:
        self._binding = binding
        self.delivery: RunStepResult | None = None

    def snapshot(self) -> RecipeRunSnapshot | None:
        """Read this fixed Run capture without RPCs; None means suppressed start."""
        return self._binding.snapshot() if self._binding is not None else None

    def complete_in_current_worker(self) -> tuple[RecipeRun | None, StepStatus]:
        """Await this handle and capture its original Run result in the current worker.

        A suppressed start delivers (None, cancelled). A native finished/cancelled
        outcome delivers (RecipeRun, completed/cancelled); the driver maps its own
        finish_early intent onto this delivery. Cancellation retains a partial handle
        even without available data or after tab close. Native failure, lost binding
        or a different result source raises for the driver to throw at the yield.
        Do not create another worker or replace the original result with a newer one.
        """
        binding = self._binding
        if binding is None:
            return None, "cancelled"
        capture = binding.capture
        if capture.op is None:
            raise ValueError("Run has no admitted operation handle")
        completion = await_operation(
            binding.tab.tools.gui,
            capture.op,
            closed=binding.tab.closed,
            condition=binding.tab.condition,
        )
        binding.publish(replace(binding.capture, outcome=completion.native))
        if completion.status == "failed":
            raise GuiRpcError(
                str(completion.native.get("error", "Run failed")), reason="run_failed"
            )
        try:
            observed = binding.tab.tools.send_gui_rpc(
                "tab.snapshot", {"tab_id": capture.tab}
            )["tabs"]
        except GuiRpcError as error:
            if completion.status != "cancelled" or error.reason != "unknown_tab":
                raise
            observed = []
        if not observed and completion.status == "cancelled":
            return RecipeRun(binding), "cancelled"
        if not isinstance(observed, list) or len(observed) != 1:
            raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
        result = recipe_gui_object(recipe_gui_object(observed[0])["result_state"])
        source = result["source_operation_id"]
        binding.publish(replace(binding.capture, result_state=deepcopy(result)))
        if source is not None:
            if binding.tab.tools.gui.expose_operation(source) != capture.op:
                raise GuiRpcError(
                    "The tab no longer contains this Run's result",
                    reason="result_superseded",
                )
        elif completion.status != "cancelled" or result["available"]:
            raise GuiRpcError(
                "The tab no longer contains this Run's result",
                reason="result_superseded",
            )
        if completion.status == "finished" and not result["available"]:
            raise GuiRpcError(
                "Run did not publish usable data", reason="run_result_unavailable"
            )
        status: StepStatus = (
            "cancelled" if completion.status == "cancelled" else "completed"
        )
        return RecipeRun(binding), status


class AnalyzeOperation:
    """Opaque, non-iterable analysis handle created by RecipeRun.analyze.

    Use RecipeRun.analyze_step with yield from, not a direct yield of this handle.
    delivery is None until the driver stores the named AnalysisStepResult.
    Interactive handoff does not finish the yield; GUI done resumes observation.
    Completed delivery captures the native result, synchronously saves the whole
    stage artifact and reads writeback candidates once without writing them.
    The artifact includes JSON even without named figures. Its Save report retains
    every output status/error; output failures do not raise at the yield.
    Available PNGs are delivered as in-memory images, not session files.
    Native operation, transport, superseded-source and later capture failures
    raise GuiRpcError at the yield. snapshot retains the captured result and whole
    Save report, including successful and failed outputs.
    Construction is framework-only: execution is the registered completion owner,
    or None for one suppressed start. Authors use RecipeRun.analyze.
    """

    def __init__(self, execution: AnalysisExecution | None) -> None:
        self._execution = execution
        self.delivery: AnalysisStepResult | None = None

    def snapshot(self) -> ExecutionSnapshot | None:
        """Read detached local progress; None means a suppressed start, not unknown."""
        return self._execution.snapshot() if self._execution is not None else None

    def preview_images(self) -> tuple[PngImage, ...]:
        """Read captured preview images without a GUI request or operation wait."""
        return self._execution.wait(0.0).images if self._execution is not None else ()

    def wake(self) -> None:
        """Wake completion after the recipe sets its close event; no GUI request."""
        if self._execution is not None:
            self._execution.wake()

    def cancel(self) -> ToolReply | None:
        """Delegate cancellation to this owner; None means no start was sent.

        An admitted analysis keeps its true native outcome and any captured
        prefix. The returned reply reports the owner's separate GUI cancel facts,
        not a delivery status. This does not consume between-yield cancellation.
        """
        return self._execution.cancel() if self._execution is not None else None

    def complete_in_current_worker(self) -> tuple[RecipeAnalysis | None, StepStatus]:
        """Complete on the recipe's worker without starting or joining another one.

        A suppressed or cancelled analysis delivers (None, cancelled). Finished
        analysis delivers (RecipeAnalysis, completed) after result, whole-artifact
        Save report, in-memory image and writeback capture. Per-output save failures
        remain in the report. Failed native operations, superseded sources and
        continuation failures raise GuiRpcError for the driver to throw at the yield.
        Snapshot/image reads retain available capture without new GUI requests.
        Raise ValueError if completion already started or has no receipt.
        """
        if self._execution is None:
            return None, "cancelled"
        snapshot = self._execution.complete_in_current_worker()
        if snapshot.status == "failed":
            error = snapshot.error
            if error is None:
                raise ValueError("Failed analysis has no error capture")
            raise GuiRpcError(error.message, reason=error.reason, code=error.code)
        if snapshot.status == "cancelled":
            return None, "cancelled"
        if snapshot.status != "finished":
            raise ValueError("Analysis completion did not reach a terminal status")
        return RecipeAnalysis(snapshot), "completed"


class WritebackQuestion:
    """Opaque, non-iterable captured question created by propose_writeback.

    Use yield from ask(question) to pause for an answer or cancellation.
    delivery is None until the driver stores the named WritebackResult.
    Answer itself never writes data.
    Construction is framework-only: selection is the validated captured proposal.
    Authors obtain handles from RecipeRun.propose_writeback.
    """

    def __init__(self, selection: WritebackSelection) -> None:
        self._selection = deepcopy(selection)
        self.delivery: WritebackResult | None = None

    def snapshot(self) -> WritebackSelection:
        """Read detached stable names and proposals without a GUI request or write."""
        return deepcopy(self._selection)


@dataclass
class RecipeTabBinding:
    """Same-package state shared by tab preparation and operation handles.

    tools is the execution's fixed, bound tool context; tab is its GUI locator.
    Parameter sources come from the tab's fixed Open/Reset environment.
    observe_run and observe_analysis publish the latest handle, including a
    suppressed start. observe_writeback records each returned receipt, including
    a failed prefix. These callbacks update local state, not the GUI.
    origins maps logical cfg paths to nonblank provenance labels (explicit,
    gui_default, disabled, library:<key> or caller labels), or SweepSources with
    separate start/stop/expts labels. Empty means no recorded sources.
    summary_parameters selects fixed Run facts; an empty tuple captures none.
    closed is the execution close event, initially unset; setting it stops later
    admissions/continuations. condition is the shared progress wake condition.
    consume_cancel atomically consumes one pending between-step cancel and returns
    whether to suppress a start; None means no pending cancel. It may run under
    the RPC lock and must not block or send RPCs. Cfg edits/saves do not consume it.
    """

    tools: MeasureToolContext
    tab: str
    observe_run: Callable[[RunOperation], None]
    observe_analysis: Callable[[AnalyzeOperation], None]
    observe_writeback: Callable[[WritebackReceipt], None]
    origins: dict[str, str | SweepSources] = field(default_factory=dict)
    summary_parameters: tuple[SummaryParameter, ...] = ()
    closed: Event = field(default_factory=Event)
    condition: Condition = field(default_factory=Condition)
    consume_cancel: Callable[[], bool] | None = None


@dataclass
class RecipeRunBinding:
    """Same-package fixed Run state created by tab admission and used by handles.

    tab is the shared preparation/lifetime binding for the original GUI tab.
    capture holds the latest local Run facts, initially with no admitted handle;
    publish replaces it as admission, completion, images and saving progress.
    analyses maps primary/post to each stage's latest AnalyzeOperation, including
    suppressed attempts. Empty means no analysis attempt. Mutations and capture
    replacement use tab.condition; snapshot returns detached observations.
    """

    tab: RecipeTabBinding
    capture: RecipeRunSnapshot
    analyses: dict[AnalysisStage, AnalyzeOperation] = field(default_factory=dict)

    def publish(self, capture: RecipeRunSnapshot) -> None:
        """Replace this Run's local facts and wake all progress waiters.

        capture describes the same original Run, including partial/failed facts.
        No GUI request, validation, merge or copying occurs.
        """
        with self.tab.condition:
            self.capture = capture
            self.tab.condition.notify_all()

    def run_source(self) -> int:
        """Return this Run's session handle when its captured data is available.

        Raise GuiRpcError(run_result_unavailable) without an admitted handle or
        available captured result. This local check sends no GUI request and
        does not certify that a later source-bound request will be admitted.
        """
        capture = self.snapshot()
        if capture.op is None or not (
            capture.result_state is not None and capture.result_state["available"]
        ):
            raise GuiRpcError(
                "Run did not publish usable data", reason="run_result_unavailable"
            )
        return capture.op

    def analysis_source(self, stage: AnalysisStage) -> tuple[int, int | None]:
        """Return source handles for primary or post analysis of this Run.

        stage is primary or post. Primary returns (Run, None); Post requires
        this Run's finished Primary capture and returns (Run, Primary).
        Raise GuiRpcError(run_result_unavailable/primary_result_unavailable)
        for missing sources. No GUI request or admission is performed.
        """
        run_op = self.run_source()
        if stage == "primary":
            return run_op, None
        primary = self.snapshot().analyses.get("primary")
        if primary is None or primary.status != "finished" or primary.op is None:
            raise GuiRpcError(
                "Post analysis requires this Run's completed Primary",
                reason="primary_result_unavailable",
            )
        return run_op, primary.op

    def snapshot(self) -> RecipeRunSnapshot:
        """Detach Run facts and latest non-suppressed primary/post captures.

        Copy local state under the shared condition, then read each analysis
        handle's snapshot. No GUI request, wait or completion runs here.
        """
        with self.tab.condition:
            capture = deepcopy(self.capture)
            operations = dict(self.analyses)
        analyses: dict[AnalysisStage, ExecutionSnapshot] = {}
        for stage, operation in operations.items():
            snapshot = operation.snapshot()
            if snapshot is not None:
                analyses[stage] = snapshot
        return replace(capture, analyses=analyses)


class RecipeStepCancelled(Exception):
    """Same-package admission signal for one consumed between-yield cancel.

    Raise only before native dispatch; Run/analysis admission catches it to
    deliver a suppressed handle. It is not a failed native start or author error.
    """


class RecipeRun:
    """Opaque capture of one tab Run, including a cancelled Run's partial result.

    The framework creates this handle at a RunOperation yield. The recipe decides
    whether to save partial data; cancellation does not make raw data saveable.
    Construction is framework-only: binding is this owner's completed Run state.
    Authors receive the handle from a yielded RunOperation.
    """

    def __init__(self, binding: RecipeRunBinding) -> None:
        self._binding = binding

    def snapshot(self) -> RecipeRunSnapshot:
        """Read this Run's detached conditions/save facts without GUI requests."""
        return self._binding.snapshot()

    def save_raw(self) -> SaveReport:
        """Save this Run once and return its complete synchronous native report.

        Reject unavailable/superseded sources and closed sessions. Native output
        failures stay in the report; transport/admission errors propagate.
        Replace the previous attempt, without a Save handle or successful prefix.
        """
        binding = self._binding
        run_op = binding.run_source()
        capture = binding.snapshot()

        def observe(report: SaveReport | None) -> None:
            binding.publish(replace(binding.capture, raw_save=report))

        def admit() -> None:
            if binding.tab.closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")

        return save_raw_data(
            binding.tab.tools.gui,
            RawSaveRequest(capture.tab, run_op, admit),
            observe=observe,
        )

    def preview(self) -> None:
        """Read this Run's current PNG bytes for inline reply, without files.

        This synchronous step does not consume pending cancel. Unavailable or
        superseded Run data, closed session and GUI/PNG failures raise here.
        It does not save an artifact, create a persistent identity, retry or
        refresh a guard. Later GUI reads need not show the same image.
        """
        binding = self._binding
        run_op = binding.run_source()

        def admit() -> None:
            if binding.tab.closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")

        reply = binding.tab.tools.gui.send_gui_rpc(
            "tab.get_figure",
            {"tab_id": binding.capture.tab, "subtab_id": "run"},
            run_operation_handle=run_op,
            before_send=admit,
        )
        encoded = reply.get("png_b64")
        if not isinstance(encoded, str):
            raise GuiRpcError("Invalid Run preview reply", reason="incompatible_wire")
        image = validated_png(base64.b64decode(encoded, validate=True))
        binding.publish(replace(binding.capture, images=(image,)))

    def analyze_step(
        self,
        stage: AnalysisStage,
        *,
        params: Mapping[str, RecipeParameter] | None = None,
    ) -> Generator[RecipeStep, None, AnalysisStepResult]:
        """Start analysis and await its capture with ``yield from``.

        stage and params follow analyze(). Interactive analysis hands control to
        the agent until done; failures are thrown here. Cancellation returns a
        named result with no analysis, leaving author cleanup in control.
        """
        operation = self.analyze(stage, params=params)
        yield RecipeStep(operation)
        if operation.delivery is None:
            raise ValueError("Analysis helper resumed without driver delivery")
        return operation.delivery

    def analyze(
        self,
        stage: AnalysisStage,
        *,
        params: Mapping[str, RecipeParameter] | None = None,
    ) -> AnalyzeOperation:
        """Immediately start primary/post analysis bound to this Run's source.

        Recipe authors use yield from analyze_step(), not yield analyze().

        params supplies named scalar/one-level-array GUI updates; None keeps defaults.
        Invalid stage or unsupported/non-finite parameters raise ValueError before
        GUI dispatch. Missing Run data or completed Primary raises GuiRpcError.
        Post uses this Run's completed Primary source. Pending between-yield cancel
        suppresses this one start and is consumed once. Rejection fails immediately;
        the yielded handle delivers cancellation or throws completion failures.

        Before completed yield delivery, capture the native result, synchronously
        save the whole stage artifact and read writeback candidates once without
        writing them. JSON is saved even without named figures. Output failures
        remain in the whole Save report, not exceptions at yield. Available PNGs
        are delivered as in-memory images; no session preview files are created.
        Native operation, transport, superseded-source and later capture failures
        raise GuiRpcError at yield. The handle snapshot retains available result
        and Save report capture.
        """
        if stage not in ("primary", "post"):
            raise ValueError("stage must be primary or post")
        updates = copy_recipe_parameters(params if params is not None else {})
        binding = self._binding
        tab = binding.tab
        suppressed = AnalyzeOperation(None)

        def observe(operation: AnalyzeOperation) -> None:
            with tab.condition:
                binding.analyses[stage] = operation
            tab.observe_analysis(operation)

        if tab.closed.is_set():
            raise GuiRpcError("MCP session is closed", reason="session_closed")
        if tab.consume_cancel is not None and tab.consume_cancel():
            observe(suppressed)
            return suppressed
        run_op, primary_op = binding.analysis_source(stage)
        execution = tab.tools.session.executions.start(
            tab.tools.gui, tab.tab, stage, start_worker=False
        )
        execution.bind_continuation(tab.closed, condition=tab.condition)
        operation = AnalyzeOperation(execution)
        observe(operation)

        def admit() -> None:
            if tab.closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            if tab.consume_cancel is not None and tab.consume_cancel():
                raise RecipeStepCancelled
            execution.admit_start()

        try:
            started = tab.tools.gui.send_gui_rpc(
                "tab.analyze" if stage == "primary" else "tab.post_analyze",
                {"tab_id": tab.tab, "updates": updates},
                run_operation_handle=run_op,
                operation_handle=primary_op,
                before_send=admit,
            )
            tab.tools.session.executions.accept_start(
                execution, started, start_worker=False
            )
        except RecipeStepCancelled:
            # This registered but unsubmitted attempt is no longer active.
            execution.fail_start(
                GuiRpcError("Analysis start was suppressed", reason="recipe_cancelled"),
                suppressed=True,
            )
            observe(suppressed)
            return suppressed
        except (
            Exception
        ) as error:  # Retain admission facts before propagating start failure.
            execution.fail_start(error)
            raise

        def admit_handoff() -> None:
            if tab.closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")

        handoff_interaction(tab.tools, execution, before_send=admit_handoff)
        return operation

    def propose_writeback(
        self, items: Sequence[str] | None = None
    ) -> WritebackQuestion:
        """Capture selected candidates from completed analyses without another read.

        items uses stable target_name, None selects all, empty selects none. Reject
        unknown, repeated or cross-stage ambiguous names before publishing the
        question. Use yield from ask(question) to await accepted/skipped or
        cancellation; no timeout or write is implied. The recipe must call
        tab.accept explicitly to write anything.
        """
        analyses = self.snapshot().analyses
        primary = analyses.get("primary")
        post = analyses.get("post")
        preview = RecipeWritebackPreview(
            primary.writeback
            if primary is not None and primary.status == "finished"
            else None,
            post.writeback if post is not None and post.status == "finished" else None,
        )
        return WritebackQuestion(select_writeback_items(preview, items))


RecipeOperation = RunOperation | AnalyzeOperation | WritebackQuestion


@dataclass(frozen=True)
class RecipeStep:
    """Framework envelope yielded by helpers, not constructed by recipe authors.

    operation is the opaque Run, analysis, or captured question to complete.
    Keeping this envelope separate lets the driver reject direct operation yields.
    """

    operation: RecipeOperation


# Include operations so the driver can diagnose direct-yield author mistakes.
RecipeGenerator = Generator[RecipeStep | RecipeOperation, None, None]


def ask(question: WritebackQuestion) -> Generator[RecipeStep, None, WritebackResult]:
    """Wait for accepted/skipped or cancellation via ``yield from ask(question)``.

    question must be a captured WritebackQuestion from propose_writeback(). This
    handoff never writes data. Cancellation returns status cancelled with no
    decision, allowing author cleanup; session close instead closes the generator.
    """
    yield RecipeStep(question)
    if question.delivery is None:
        raise ValueError("Question helper resumed without driver delivery")
    return question.delivery
