"""Author-facing generator recipe contracts; GUI policy stays in the GUI owner.

Recipes receive a RecipeSession from the driver, not a transport or cfg publication.
Run and analysis use typed yield-from helpers; ask marks a writeback handoff.
The driver stores each named result on its operation, or throws a failure back
into the helper call. Session close closes the generator, so ordinary Python
finally blocks still run.

Cfg preparation uses one fixed connection and the GUI's returned publications.
The injected definitions drive these handles through the shared execution worker.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Generator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from threading import Condition, Event
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, TypeGuard

from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.remote.form_description import RecipeFieldObservation
from zcu_tools.mcp.core.images import validated_png
from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.analysis_execution import (
    AnalysisExecution,
    AnalysisWriteback,
    ExecutionSnapshot,
)
from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.interaction import handoff_interaction
from zcu_tools.mcp.measure.operation_wait import await_operation
from zcu_tools.mcp.measure.raw_save import RawSaveRequest, SaveReport, save_raw_data
from zcu_tools.mcp.measure.recipe_capture import (
    ActualField,
    RecipeActual,
    SweepSources,
    capture_actual,
)
from zcu_tools.mcp.measure.recipe_form import (
    RecipeFormField,
    RecipeModuleValues,
    directed_errors,
    field_observation,
    read_form_field,
    read_module_values,
    RecipeFieldPath,
    field_path,
)
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.writeback import WritebackSelection

RecipeScalar = str | int | float | bool | None
RecipeParameter = RecipeScalar | list[RecipeScalar]
# Declared containers are invocation inputs, not scalar edits or analysis updates.
type RecipeInputValue = (
    RecipeScalar | list[RecipeInputValue] | dict[str, RecipeInputValue]
)
StepStatus = Literal["completed", "finished_early", "cancelled"]
WritebackDecision = Literal["accepted", "skipped"]
AnalysisStage = Literal["primary", "post"]


class RecipeExpression(TypedDict):
    """GUI expression input, forwarded unchanged rather than evaluated in MCP.

    expr is the expression text in the GUI's Form expression language.
    The GUI owns parsing, source resolution and diagnostics, including blanks.
    """

    expr: str


@dataclass(frozen=True)
class RecipeFieldEdit:
    """One scalar field edit for RecipeTab.set_fields.

    field is a typed logical path of str names and int original list indexes. value is a direct scalar/list or
    RecipeExpression; None retains the current input, it does not write null.
    source is a non-blank provenance label; retained inputs use gui_default.
    """

    field: RecipeFieldPath
    value: RecipeParameter | RecipeExpression
    source: str = "explicit"


@dataclass(frozen=True)
class RecipeDeviceState:
    """Detached cached device coordinate metadata, not live driver information.

    unit is the snapshot's raw YAML/JSON unit value, including None or tagged
    containers. No connected/settable/type/unit policy or conversion is applied.
    """

    unit: YamlValue


class RecipePropertySchema(TypedDict):
    """Handwritten schema for scalars, declared objects and arrays of either.

    type lists accepted JSON types, including null for nullable inputs. minLength
    constrains strings; items describes scalar or declared object array elements
    (not direct arrays of arrays); minItems/maxItems bound array length.
    properties declares object fields; required lists mandatory
    declared fields. additionalProperties rejects unknown fields when false, allows
    scalar/array extras when true or omitted, or declares a schema for name-keyed
    object entries. Objects need properties or a schema-valued additionalProperties.
    description is operator-facing documentation. No schema generation takes place.
    """

    type: str | list[str]
    minLength: NotRequired[int]
    items: NotRequired[RecipePropertySchema]
    minItems: NotRequired[int]
    maxItems: NotRequired[int]
    description: NotRequired[str]
    properties: NotRequired[dict[str, RecipePropertySchema]]
    required: NotRequired[list[str]]
    additionalProperties: NotRequired[bool | RecipePropertySchema]


class RecipeInputSchema(TypedDict):
    """An object schema whose properties name the recipe's typed keyword inputs.

    additionalProperties rejects undeclared keys when false. required names the
    mandatory keys; omitted keys otherwise use the recipe's Python defaults.
    """

    type: Literal["object"]
    additionalProperties: bool
    properties: dict[str, RecipePropertySchema]
    required: NotRequired[list[str]]


@dataclass(frozen=True)
class MissingParameter:
    """One missing tool parameter and the instruction for supplying it.

    parameter is the public keyword name; reason explains the absent required source
    or input. Both strings must be non-empty. Construction raises ValueError with
    "missing parameters must contain non-empty names and reasons" if either is empty.
    """

    parameter: str
    reason: str

    def __post_init__(self) -> None:
        if not self.parameter or not self.reason:
            raise ValueError(
                "missing parameters must contain non-empty names and reasons"
            )


class RecipeNeedsParameters(Exception):
    """End a recipe with a needs_parameters receipt instead of a GUI failure.

    missing must be a non-empty tuple of MissingParameter values with non-empty
    strings. The same tuple is exposed as missing. Invalid declarations raise
    ValueError; the exception does not close a generator or perform a GUI action.
    """

    def __init__(self, missing: tuple[MissingParameter, ...]) -> None:
        if not missing or any(
            not item.parameter or not item.reason for item in missing
        ):
            raise ValueError(
                "missing parameters must contain non-empty names and reasons"
            )
        self.missing = missing
        super().__init__(
            "; ".join(f"{item.parameter}: {item.reason}" for item in missing)
        )


class WritebackItemReceipt(TypedDict):
    """One GUI-confirmed write, retaining its native before/after values.

    id is the current pane's opaque candidate ID; kind is its GUI-owned writeback
    kind; target is the native destination description. before and after are
    opaque native values and may be None. An unconfirmed write is never listed.
    """

    id: str
    kind: str
    target: object
    before: object
    after: object


class WritebackStageReceipt(TypedDict):
    """Confirmed writes for the named primary or post analysis stage."""

    stage: AnalysisStage
    written: list[WritebackItemReceipt]


class WritebackFailure(TypedDict):
    """First failed writeback step: GUI code/reason may be None; message is text."""

    code: str | None
    reason: str | None
    message: str


class WritebackReceipt(TypedDict):
    """Progress of a Primary-then-Post writeback without rollback or retry.

    tab is the GUI locator. status is finished or failed. completed contains only
    confirmed writes grouped by stage. skipped names stages with no selected
    candidates; not_started names stages not attempted after the first failure.
    Failed replies additionally name failed_stage and error. The partial-writes
    flag says only that a write RPC was attempted in the failed stage, not which
    unconfirmed items were written. Finished replies omit all three failure keys.
    """

    tab: str
    status: Literal["finished", "failed"]
    completed: list[WritebackStageReceipt]
    skipped: list[AnalysisStage]
    not_started: list[AnalysisStage]
    failed_stage: NotRequired[AnalysisStage]
    error: NotRequired[WritebackFailure]
    failed_stage_may_have_partial_writes: NotRequired[bool]


class WritebackError(RuntimeError):
    """Report failed tab.accept while retaining its confirmed progress receipt.

    receipt must have status failed. The same receipt is exposed as receipt.
    ValueError rejects a finished receipt. No retry or rollback is performed.
    """

    def __init__(self, receipt: WritebackReceipt) -> None:
        if receipt["status"] != "failed":
            raise ValueError("WritebackError requires a failed receipt")
        self.receipt = receipt
        error = receipt.get("error")
        super().__init__(error["message"] if error else "writeback failed")


@dataclass(frozen=True)
class RecipeWritebackPreview:
    """Captured proposals for a question; no live query or authorization.

    primary and post are their completed analysis captures, including destination
    context. None means that stage has no captured preview. Selection resolves
    the stable target_name across both stages before asking the question.
    """

    primary: AnalysisWriteback | None = None
    post: AnalysisWriteback | None = None


@dataclass(frozen=True)
class RecipeAnalysis:
    """Completed analysis capture delivered by an AnalyzeOperation yield.

    snapshot contains the stage's native result, actual parameters, invalid
    fields, artifacts, preview and writeback facts. No method re-reads candidates.
    """

    snapshot: ExecutionSnapshot


@dataclass(frozen=True)
class RecipeRunSnapshot:
    """Detached Run facts for framework delivery; reading them sends no RPCs.

    tab is the GUI locator. actual is the conditions captured before native start.
    op is the opaque session Run handle, or None before its receipt. start_status
    is not_started, unknown while dispatch is uncertain, or running after receipt.
    start_reason captures a rejected start's reason, or None. outcome is the native
    Run terminal reply, or None before completion. result_state is its captured
    native data availability/provenance, or None before capture or after tab close.
    raw_save is the latest synchronous native Save report, or None before its
    reply. analyses maps primary/post to each stage's latest local completion,
    including failures and partial artifacts; a suppressed start has no entry.
    images contains validated PNG bytes for the current reply, without files
    or persistent identity. A later live GUI image may differ.
    """

    tab: str
    actual: RecipeActual
    op: int | None = None
    start_status: Literal["not_started", "unknown", "running"] = "not_started"
    start_reason: str | None = None
    outcome: Mapping[str, object] | None = None
    result_state: Mapping[str, object] | None = None
    raw_save: SaveReport | None = None
    analyses: Mapping[AnalysisStage, ExecutionSnapshot] = field(default_factory=dict)
    images: tuple[PngImage, ...] = ()


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

    def __init__(self, binding: _RunBinding | None) -> None:
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
        result = _cfg_object(_cfg_object(observed[0])["result_state"])
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


class _CfgEdit(TypedDict):
    """One wire edit: path is a public field path; value is GUI input data."""

    path: list[str | int]
    value: object


@dataclass
class _TabBinding:
    """Cfg preparation state hidden behind an author tab handle.

    tools is the execution's fixed connection. tab is the GUI locator.
    Parameter sources are read from the tab's fixed Open/Reset environment.
    origins records sources by public/derived field name for the later Run
    capture, including individual sweep labels. summary_parameters selects the fixed Run facts to capture.
    """

    tools: MeasureToolContext
    tab: str
    observe_run: Callable[[RunOperation], None]
    observe_analysis: Callable[[AnalyzeOperation], None]
    observe_writeback: Callable[[WritebackReceipt], None]
    origins: dict[RecipeFieldPath, str | SweepSources] = field(default_factory=dict)
    summary_parameters: tuple[SummaryParameter, ...] = ()
    closed: Event = field(default_factory=Event)
    condition: Condition = field(default_factory=Condition)
    consume_cancel: Callable[[], bool] | None = None


@dataclass
class _RunBinding:
    """One fixed GUI Run and its local capture; never read by another owner."""

    tab: _TabBinding
    capture: RecipeRunSnapshot
    analyses: dict[AnalysisStage, AnalyzeOperation] = field(default_factory=dict)

    def publish(self, capture: RecipeRunSnapshot) -> None:
        with self.tab.condition:
            self.capture = capture
            self.tab.condition.notify_all()

    def run_source(self) -> int:
        """Require this Run's captured usable data and return its native handle."""
        capture = self.snapshot()
        if capture.op is None or not (
            capture.result_state is not None and capture.result_state["available"]
        ):
            raise GuiRpcError(
                "Run did not publish usable data", reason="run_result_unavailable"
            )
        return capture.op

    def analysis_source(self, stage: AnalysisStage) -> tuple[int, int | None]:
        """Require this Run's data and return its Run/optional Primary handles."""
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
        with self.tab.condition:
            capture = deepcopy(self.capture)
            operations = dict(self.analyses)
        analyses: dict[AnalysisStage, ExecutionSnapshot] = {}
        for stage, operation in operations.items():
            snapshot = operation.snapshot()
            if snapshot is not None:
                analyses[stage] = snapshot
        return replace(capture, analyses=analyses)


class _StepCancelled(Exception):
    """A consumed between-yield cancel, not a failed native start."""


def _cfg_object(value: object) -> dict[str, object]:
    """Decode an object at the GUI wire boundary without accepting bad keys."""
    if not isinstance(value, dict):
        raise GuiRpcError("Expected a GUI object", reason="incompatible_wire")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise GuiRpcError("Expected GUI string keys", reason="incompatible_wire")
        result[key] = item
    return result


def _wire_value(value: object) -> YamlValue:
    """Copy the JSON/YAML representation, retaining opaque tagged objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_wire_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _wire_value(item) for key, item in _cfg_object(value).items()}
    raise GuiRpcError("Invalid GUI value", reason="incompatible_wire")


def _is_expression(value: object) -> bool:
    """Recognize only the existing one-key expression wire carrier."""
    return (
        isinstance(value, dict)
        and set(value) == {"expr"}
        and isinstance(value["expr"], str)
    )


def _check_edit_value(value: object) -> None:
    if (
        value is None
        or isinstance(value, (str, bool, int, float))
        or _is_expression(value)
    ):
        return
    if isinstance(value, list) and all(
        item is None or isinstance(item, (str, bool, int, float)) for item in value
    ):
        return
    raise ValueError("value must be a scalar, scalar list or RecipeExpression")


def _source_label(source: object) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty provenance label")
    return source


def _check_points(expts: object) -> None:
    if expts is not None and (
        not isinstance(expts, int) or isinstance(expts, bool) or expts < 2
    ):
        raise ValueError("expts must be an integer of at least 2")


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _alias_overrides(value: object) -> dict[str, str] | None:
    """Own optional named overrides; reject malformed or empty names before RPC."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("aliases must map non-empty names to components")
    overrides: dict[str, str] = {}
    for alias, component in value.items():
        if (
            not isinstance(alias, str)
            or not alias.strip()
            or not isinstance(component, str)
            or not component.strip()
        ):
            raise ValueError("aliases must map non-empty names to components")
        overrides[alias] = component
    return overrides or None


class RecipeSession:
    """One execution's fixed GUI binding, created and passed by the driver.

    tools is the owning session's tool context. Construction captures its current
    GUI connection (or performs the initial attach); subsequent helpers never
    reconnect, retry stale requests or synthesize GUI defaults. Fast edits remain
    available after cancellation; the recipe decides its policy.
    """

    def __init__(
        self,
        tools: MeasureToolContext,
        *,
        summary_parameters: tuple[SummaryParameter, ...] = (),
        closed: Event | None = None,
        condition: Condition | None = None,
        consume_cancel: Callable[[], bool] | None = None,
    ) -> None:
        """Bind tools and optional execution lifetime/admission collaborators.

        summary_parameters is the recipe definition's declared cfg fields and
        units for Run capture. Empty tuple records no actual conditions.
        closed/condition are the driver's close event and wake condition; omission
        creates local ones. consume_cancel atomically consumes one pending cancel
        before a Run/analysis start and returns whether to suppress it. This
        callback runs under the RPC lock: it must not block or send RPCs. None
        means no pending cancel. Cfg helpers and raw saves do not consume it.
        """
        self._summary_parameters = summary_parameters
        self._tools = tools.bound()
        self._closed = closed if closed is not None else Event()
        self._condition = condition if condition is not None else Condition()
        self._consume_cancel = consume_cancel
        self._tab: str | None = None
        self._run: RunOperation | None = None
        self._analysis: AnalyzeOperation | None = None
        self._latest_operation: RunOperation | AnalyzeOperation | None = None
        self._written: list[WritebackReceipt] = []

    def _observe_run(self, operation: RunOperation) -> None:
        with self._condition:
            self._run = operation
            self._latest_operation = operation
            self._condition.notify_all()

    def _observe_analysis(self, operation: AnalyzeOperation) -> None:
        with self._condition:
            self._analysis = operation
            self._latest_operation = operation
            self._condition.notify_all()

    def _observe_writeback(self, receipt: WritebackReceipt) -> None:
        with self._condition:
            self._written.append(deepcopy(receipt))
            self._condition.notify_all()

    def pending_operation(self) -> RunOperation | AnalyzeOperation | None:
        """Read the latest unfinished Run/analysis handle, even before its yield.

        Include an admitted start awaiting its receipt and interactive handoff.
        Return None before admission, after completion, or for a suppressed start.
        This performs no GUI request or completion. The driver uses the same
        opaque handle for cancel/finish_early during synchronous author calls.
        """
        with self._condition:
            operation = self._latest_operation
        if isinstance(operation, RunOperation):
            capture = operation.snapshot()
            if (
                capture is not None
                and capture.start_status in ("unknown", "running")
                and capture.outcome is None
            ):
                return operation
        elif isinstance(operation, AnalyzeOperation):
            capture = operation.snapshot()
            if capture is not None and capture.status in ("running", "interactive"):
                return operation
        return None

    def writeback_receipts(self) -> tuple[WritebackReceipt, ...]:
        """Read detached actual writes in call order, including failed prefixes.

        Empty means tab.accept has not returned a receipt. Answers and proposals
        are not writes. Reading sends no GUI request and does not refresh guards.
        """
        with self._condition:
            return deepcopy(tuple(self._written))

    def tab_locator(self) -> str | None:
        """Read the latest prepared/requested tab locator without GUI requests.

        A successful creation receipt or explicit reuse records its locator before
        cfg reads/reset/validation can fail. None means neither has occurred.
        A requested reuse locator does not certify existence, adapter or readiness.
        The driver uses this capture when no Run attempt exists yet.
        """
        with self._condition:
            return self._tab

    def analysis_snapshot(self) -> ExecutionSnapshot | None:
        """Read the latest analysis attempt, including a not-yet-yielded failure.

        No GUI request or completion runs here. None means no analysis attempt or
        one suppressed start; unknown means dispatch occurred without a receipt.
        Completed results, artifacts and failures stay with the original owner.
        """
        with self._condition:
            operation = self._analysis
        return operation.snapshot() if operation is not None else None

    def run_snapshot(self) -> RecipeRunSnapshot | None:
        """Read the latest Run attempt for this execution without GUI requests.

        Return detached conditions/start/completion/save facts, including an
        uncertain or rejected start whose handle was not yielded. None means no
        Run was attempted or the latest start was suppressed by pending cancel.
        This framework observation does not complete Run or refresh guards.
        """
        with self._condition:
            operation = self._run
        return operation.snapshot() if operation is not None else None

    def open_tab(
        self,
        adapter: str,
        *,
        aliases: Mapping[str, str] | None = None,
        reuse: str | None = None,
    ) -> RecipeTab:
        """Open adapter's tab, or validate/reset the named reusable tab.

        adapter is a non-empty public GUI adapter ID. aliases maps caller-owned
        alias names to concrete components and is copied into a single Reset.
        None and {} use global aliases at Reset. Every recipe preparation resets
        before edits, including a new tab. Parameter/library reads, edits and Run
        use that fixed capture until the next explicit Reset. The override is not
        written to global aliases or saved input. reuse=None creates a fresh tab;
        otherwise reuse is a non-empty GUI locator. Busy, mismatched adapters and
        invalid locators fail without creating a fallback. GUI/connection
        errors propagate; empty/non-string names raise ValueError before any GUI request. Identifier errors propagate from the GUI context owner.
        """
        if not adapter or not adapter.strip():
            raise ValueError("adapter must be a non-empty GUI adapter ID")
        overrides = _alias_overrides(aliases)
        if reuse is not None and (not reuse or not reuse.strip()):
            raise ValueError("reuse must be a non-empty GUI tab locator")
        if reuse is None:
            payload: dict[str, object] = {"adapter_name": adapter}
            created = self._tools.send_gui_rpc("tab.new", payload)
            tab = created["tab_id"]
            if not isinstance(tab, str) or not tab:
                raise GuiRpcError(
                    "Invalid tab creation reply", reason="incompatible_wire"
                )
            with self._condition:
                self._tab = tab

        else:
            tab = reuse
            with self._condition:
                self._tab = tab
            observed = self._tools.send_gui_rpc("tab.snapshot", {"tab_id": tab})["tabs"]
            if not isinstance(observed, list) or len(observed) != 1:
                raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
            snapshot = _cfg_object(observed[0])
            if snapshot["tab_id"] != tab:
                raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
            if snapshot["adapter_name"] != adapter:
                raise GuiRpcError(
                    "Tab has a different experiment", reason="wrong_experiment"
                )
            interaction = _cfg_object(snapshot["interaction"])
            if interaction["is_running"] or interaction["is_analyzing"]:
                raise GuiRpcError("Tab is busy", reason="tab_busy")
        reset: dict[str, object] = {"tab_id": tab}
        if overrides:
            reset["aliases"] = overrides
        self._tools.send_gui_rpc("tab.reset_cfg", reset)
        return RecipeTab(
            _TabBinding(
                self._tools,
                tab,
                observe_run=self._observe_run,
                observe_analysis=self._observe_analysis,
                observe_writeback=self._observe_writeback,
                closed=self._closed,
                condition=self._condition,
                consume_cancel=self._consume_cancel,
                summary_parameters=self._summary_parameters,
            )
        )


class RecipeTab:
    """Opaque tab handle returned by RecipeSession.open_tab.

    Fields are nonempty tuples of str names and int original list indexes.
    Numeric strings remain literal field names; names never select positions.
    Methods read current field observations and record parameter sources.
    GUI errors propagate without re-read, retry or rollback. Authors obtain
    this handle from open_tab rather than constructing the private binding.
    """

    def __init__(self, binding: _TabBinding) -> None:
        self._binding = binding

    def read_value(self, path: str) -> YamlValue:
        """Read one non-blank logical leaf from this tab's fixed Reset capture.

        path is a parameter path such as $custom.threshold or Q1.ext.seed, not an
        arithmetic formula. Missing and explicit null both return None. Each call
        uses tab.parameter_read without overrides and returns a detached value.
        It does not change cfg, provenance or Run state, or refresh guards. Source,
        alias and fixed-connection failures propagate without fallback or retry.
        ValueError rejects blank paths; malformed wire raises incompatible_wire.
        """
        if not path or not path.strip():
            raise ValueError("path must be a non-empty logical leaf path")
        return _wire_value(self._read_parameter(path))

    def read_library_root(self, reference: str | None) -> str | None:
        """Ask the source owner for reference's canonical library module root.

        None returns None without an RPC. Other references must be non-blank;
        custom/missing modules return None, never a client-guessed root. Queries
        use the fixed Reset capture, not context.snapshot or live aliases. Cfg,
        provenance and Run state are unchanged. Source/connection errors propagate
        without fallback/retry; malformed root replies raise incompatible_wire.
        """
        if reference is not None and (not reference or not reference.strip()):
            raise ValueError("reference must be a non-empty module reference")
        return self._library_reference(reference)

    def read_field(self, field: RecipeFieldPath) -> RecipeFormField:
        """Read a logical field with one fresh tab.read_recipe_field RPC.

        field is a nonempty typed logical path; only generated wrappers are
        hidden. Range count is n. Metadata, authored input, ordinary value,
        value_status, selector/source identity, readonly and local errors come
        from one owner turn. Ref reads project fixed captured Specs without
        materialization, source refresh or write authority. Unknown/inactive
        paths and source failures retain owner errors; malformed local paths
        raise ValueError, malformed replies raise incompatible_wire. Returned
        views are detached. Cfg validity still belongs to get_cfg/Run admission.
        """
        return self._node(field)

    def read_device(self, name: str) -> RecipeDeviceState:
        """Read name's cached device.snapshot once on the fixed GUI connection.

        name is a non-blank registered device name. This does not connect, setup,
        poll hardware, change cfg/provenance, or interpret/convert units. Missing
        unit or malformed wire raises GuiRpcError(incompatible_wire). Owner errors
        (including unknown devices) and connection failures propagate, without
        reconnect or retry. ValueError rejects blank names before dispatch.
        """
        if not name or not name.strip():
            raise ValueError("device name must be non-empty")
        reply = self._binding.tools.send_gui_rpc("device.snapshot", {"name": name})
        snapshot = _cfg_object(reply.get("snapshot"))
        if "unit" not in snapshot:
            raise GuiRpcError("Missing device unit", reason="incompatible_wire")
        return RecipeDeviceState(unit=_wire_value(snapshot["unit"]))

    def _read_parameter(self, path: str) -> object:
        """Read one logical path, retaining missing/null values."""
        reply = self._binding.tools.send_gui_rpc(
            "tab.parameter_read",
            {
                "tab_id": self._binding.tab,
                "paths": [path],
                "module_refs": [],
            },
        )
        return _cfg_object(reply.get("values")).get(path)

    def _library_reference(self, reference: str | None) -> str | None:
        """Resolve a canonical module root using the fixed Reset sources."""
        if reference is None:
            return None
        reply = self._binding.tools.send_gui_rpc(
            "tab.parameter_read",
            {
                "tab_id": self._binding.tab,
                "paths": [],
                "module_refs": [reference],
            },
        )
        modules = _cfg_object(reply.get("modules"))
        if reference not in modules:
            raise GuiRpcError("Missing module lookup reply", reason="incompatible_wire")
        root = modules[reference]
        if root is not None and not isinstance(root, str):
            raise GuiRpcError("Invalid module lookup reply", reason="incompatible_wire")
        return root

    def _node(self, name: RecipeFieldPath) -> RecipeFormField:
        path = tuple(field_path(name))
        return read_form_field(self._describe(path), path)

    def _describe(self, path: RecipeFieldPath) -> RecipeFieldObservation:
        return field_observation(
            self._binding.tools.send_gui_rpc(
                "tab.read_recipe_field",
                {"tab_id": self._binding.tab, "path": list(path)},
            )
        )

    def _edit(self, edits: list[_CfgEdit]) -> Mapping[str, object] | None:
        if edits:
            return self._binding.tools.send_gui_rpc(
                "tab.edit_recipe_fields",
                {
                    "tab_id": self._binding.tab,
                    "edits": edits,
                },
            )

        return None

    def use_library(
        self, field: RecipeFieldPath, name: str | None, *, required: str | None = None
    ) -> MissingParameter | None:
        """Select an actual library key using one logical owner command.

        field is a logical generated selector. name is a nonblank actual key;
        owner checks captured family/choices/locks/ref ancestors. None retains
        input and performs one fresh read. required names a nonblank caller
        parameter; absent/invalid retained keys return MissingParameter without
        mutation or success provenance. Explicit success records explicit,
        retained success gui_default. No Option/input inspection, materialization,
        refresh or retry occurs; owner and connection failures propagate.
        """
        field_path(field)
        if required is not None and not required.strip():
            raise ValueError("required must be a nonempty parameter name")
        if name is not None:
            if not name.strip():
                raise ValueError("library name must be non-empty")
            selected = self._select_module(field, "use_library", key=name)
        else:
            selected = self.read_field(field)
        if selected.selector is None:
            raise ValueError(f"Not a module reference field: {field}")
        valid = (
            selected.mode == "key"
            and selected.key is not None
            and selected.value_status == "available"
            and not selected.errors
        )
        if name is not None and (not valid or selected.key != name):
            raise GuiRpcError(f"Invalid {field} reference", reason="invalid_cfg")
        if name is None and required is not None and not valid:
            return MissingParameter(required, f"Provide a valid {field} library key")
        self._binding.origins[field_path(field)] = "explicit" if name is not None else "gui_default"
        return None

    def _select_module(
        self,
        field: RecipeFieldPath,
        action: Literal["use_library", "use_inline", "disable"],
        *,
        key: str | None = None,
    ) -> RecipeFormField:
        path = tuple(field_path(field))
        params: dict[str, object] = {
            "tab_id": self._binding.tab,
            "path": list(path),
            "action": action,
        }
        if key is not None:
            params["key"] = key
        reply = self._binding.tools.send_gui_rpc("tab.select_recipe_module", params)
        return read_form_field(field_observation(_cfg_object(reply.get("field"))), path)

    def disable_library(self, field: RecipeFieldPath) -> None:
        """Disable a nullable logical selector; owner checks locks and ancestry.

        Sends one selector command, never set(null). On success records disabled
        provenance. Admission/source/control failures propagate without retry.
        """
        selected = self._select_module(field, "disable")
        if (
            selected.selector is None
            or selected.selector["mode"] != "disabled"
            or selected.errors
        ):
            raise GuiRpcError(f"Cannot disable {field}", reason="invalid_cfg")
        self._binding.origins[field_path(field)] = "disabled"

    def read_module(self, field: RecipeFieldPath) -> RecipeModuleValues:
        """Read one core module's value, mode and source from fixed Reset sources.

        field is a nonempty tuple of str core module field names, not an inline
        leaf. Int indexes are unsupported by this materialization owner and raise
        ValueError before dispatch. Numeric strings remain literal names. Values
        are detached module JSON, without full-cfg validation, native to_cfg or
        extras lowering. Other invalid fields do not block this read. The owner
        rejects this module's invalid values with NOT_VALID, unknown/nonmodule
        paths with invalid_params, and revoked resources with RESOURCE_GONE.
        Busy and notification reads are allowed. No input change, publication,
        source refresh or retry occurs. GUI/source/connection failures propagate.
        """
        field_path(field)
        if any(not isinstance(segment, str) for segment in field):
            raise ValueError("read_module supports str fields only, not list indexes")
        return read_module_values(
            self._binding.tools.send_gui_rpc(
                "tab.read_module", {"tab_id": self._binding.tab, "field": list(field_path(field))}
            )
        )

    def use_inline(self, field: RecipeFieldPath) -> None:
        """Explicitly copy a logical selector from its fixed captured Spec.

        Sends tab.select_recipe_module/use_inline even for existing inline, so
        holder/busy/lifetime/notification guards still apply to no-ops. Outer ref
        ancestors must be copied explicitly first. Missing/disabled/unresolved
        input never creates defaults. A successful copy preserves child keys;
        later failed writes do not roll it back. Owner/source/connection failures
        propagate without refresh, fallback or retry.
        """
        self._select_module(field, "use_inline")

    def retain_module_source(
        self, field: RecipeFieldPath, *, summary_field: RecipeFieldPath, value_path: RecipeFieldPath
    ) -> None:
        """Retain provenance for one logical scalar without materializing a module.

        field is a logical selector; value_path is its nonempty relative logical
        scalar path. summary_field must identify that same leaf. Validates its
        current ordinary value/status/errors, then records library:<actual-key>
        from owner source or field for authored inline values. No input changes
        or write authority result. Fresh Run reads capture detached facts.
        """
        path = tuple(field_path(field))
        value_path = field_path(value_path)
        if tuple(field_path(summary_field)) != (*path, *value_path):
            raise ValueError("summary_field must match the logical module value path")
        node = self.read_field(summary_field)
        if (
            node.selector is not None
            or node.fields
            or node.value_status != "available"
            or node.errors
        ):
            raise ValueError("Module value path must select an available scalar")
        self._binding.origins[field_path(summary_field)] = (
            f"library:{node.source['key']}" if node.source is not None else repr(field)
        )

    def retain_source(self, field: RecipeFieldPath, source: str) -> None:
        """Record a caller-chosen provenance label for an unchanged cfg field.

        field is an existing typed scalar or sweep path. source is a nonblank
        string, such as a library label; this method interprets no domain policy.
        It changes only this tab's source metadata for the next Run. Subsequent
        set/set_sweep calls can replace the label, and captured Runs remain fixed.
        One fresh logical read validates the field category. No Form input
        change or validation of the resolved value occurs.
        Unknown/non-scalar/non-sweep fields and empty labels raise ValueError before metadata changes. set(None) still records gui_default.
        """
        label = _source_label(source)
        node = self._node(field)
        if node.selector is not None or (
            node.fields and not {"start", "stop", "n"} <= set(node.fields)
        ):
            raise ValueError(f"Not a scalar or sweep cfg field: {field}")
        self._binding.origins[field_path(field)] = label

    def set_fields(self, edits: Sequence[RecipeFieldEdit]) -> None:
        """Send supplied logical scalar edits in one tab.edit_recipe_fields RPC.

        edits is an ordered sequence of typed scalar paths, direct/expression
        inputs and non-blank provenance labels. None values retain current inputs
        and record gui_default. Empty/default-only batches send no edit. Validate
        local path syntax, carrier and label before any mutation; local
        errors raise ValueError. Expressions go unchanged to the GUI owner.
        The resource returns changed and complete logical Form errors. Check directed
        errors before recording sources; invalid replies raise invalid_cfg
        but leave the owner's committed draft intact.
        GUI/connection failures propagate without refresh, retry or rollback;
        the resource owns the atomic write.
        """
        planned = tuple(edits)
        wire: list[_CfgEdit] = []
        for edit in planned:
            if not edit.source or not edit.source.strip():
                raise ValueError("source must be a non-empty provenance label")
            field_path(edit.field)
            _check_edit_value(edit.value)
            if edit.value is not None:
                wire.append(
                    {"path": list(field_path(edit.field)), "value": deepcopy(edit.value)}
                )
        reply = self._edit(wire)
        for edit in planned:
            if (
                reply is not None
                and edit.value is not None
                and directed_errors(reply, tuple(field_path(edit.field)))
            ):
                raise GuiRpcError(
                    f"Invalid {edit.field} value; owner retained the draft",
                    reason="invalid_cfg",
                )
        for edit in planned:
            self._binding.origins[field_path(edit.field)] = (
                edit.source if edit.value is not None else "gui_default"
            )

    def set(
        self,
        field: RecipeFieldPath,
        value: RecipeParameter | RecipeExpression,
        *,
        source: str = "explicit",
    ) -> None:
        """Set one typed scalar path to a direct input or GUI expression carrier.

        None keeps its current input and records gui_default. source is a non-blank
        provenance label for supplied values. Unknown/non-scalar fields and invalid
        carriers/labels are checked locally. The GUI owns path/admission errors,
        parses expressions
        and validates values; returned errors raise GuiRpcError(invalid_cfg) without
        retry or rollback. Earlier returned field views remain detached.
        """
        self.set_fields((RecipeFieldEdit(field, value, source),))

    def _write_sweep(self, field: RecipeFieldPath, values: dict[str, object]) -> None:
        field_path(field)
        if not values:
            return
        edits: list[_CfgEdit]
        if {"start", "stop", "expts"} <= values.keys():
            edits = [
                {
                    "path": list(field_path(field)),
                    "value": {
                        "kind": "linrange",
                        "start": values["start"],
                        "stop": values["stop"],
                        "n": values["expts"],
                    },
                }
            ]
        else:
            edits = [
                {
                    "path": [*field_path(field), "n" if key == "expts" else key],
                    "value": value,
                }
                for key, value in values.items()
            ]
        reply = self._edit(edits)
        if reply is not None and directed_errors(reply, tuple(field_path(field))):
            raise GuiRpcError(
                f"Invalid {field} sweep; draft retained", reason="invalid_cfg"
            )

    def set_sweep(
        self,
        field: RecipeFieldPath,
        *,
        start: float | RecipeExpression | None = None,
        stop: float | RecipeExpression | None = None,
        expts: int | None = None,
        sources: SweepSources | None = None,
    ) -> None:
        """Set sweep coordinates in native units; complete values select Range + Expts.

        Supplying start, stop and expts replaces the whole sweep atomically,
        independent of active coordinate mode or an invalid previous draft.
        Partial values edit only active Range leaves, retaining omitted ones;
        inactive paths reject rather than guessing a conversion.

        Supplied numeric endpoints must be finite and not boolean; RecipeExpression
        endpoints pass unchanged to the GUI for parsing/resolution/validation.
        expts is an integer
        of at least 2; None retains the GUI count. sources optionally labels
        start/stop/expts separately in the Run capture; all three non-empty labels
        are required. Omitted values always retain gui_default provenance.
        Without sources, capture uses the existing explicit/gui_default labels.
        Invalid explicit inputs, sources and unknown fields raise ValueError.
        GUI field errors raise GuiRpcError.
        """
        if sources is not None and (
            set(sources) != {"start", "stop", "expts"}
            or any(
                not isinstance(label, str) or not label.strip()
                for label in sources.values()
            )
        ):
            raise ValueError(
                "sources must label start, stop and expts with non-empty strings"
            )
        _check_points(expts)
        for endpoint in (start, stop):
            if (
                endpoint is not None
                and not _is_expression(endpoint)
                and not _finite_number(endpoint)
            ):
                raise ValueError("sweep endpoints must be finite and not boolean")
        values: dict[str, object] = {}
        for key, value in (("start", start), ("stop", stop), ("expts", expts)):
            if value is not None:
                values[key] = value
        self._write_sweep(field, values)
        captured_sources: SweepSources = (
            deepcopy(sources)
            if sources is not None
            else {"start": "explicit", "stop": "explicit", "expts": "explicit"}
        )
        for key in ("start", "stop", "expts"):
            if key not in values:
                captured_sources[key] = "gui_default"
            model_field = "n" if key == "expts" else key
            self._binding.origins[(*field_path(field), model_field)] = captured_sources[key]
        self._binding.origins[field_path(field)] = captured_sources

    def run_step(self) -> Generator[RecipeStep, None, RunStepResult]:
        """Send Run and await its named result using ``yield from tab.run_step()``.

        Uses the same cfg, source and cancellation contract as run(). Completion
        failures are thrown here. Cancelled results let the author clean up and
        return; finish_early retains the partial Run for saving and analysis.
        """
        operation = self.run()
        yield RecipeStep(operation)
        if operation.delivery is None:
            raise ValueError("Run helper resumed without driver delivery")
        return operation.delivery

    def run(self) -> RunOperation:
        """Immediately send Run and return an opaque handle for framework use.

        Recipe authors use yield from run_step(), not yield run().

        The recipe cfg publication must be valid. Reset captured aliases and
        parameter sources before edits; Run sends no source overrides.
        Capture actual conditions and result provenance. A pending between-yield
        cancel suppresses this one start and is consumed once. GUI rejection fails
        immediately; completion failures are thrown by the driver at the yield.
        """
        suppressed = RunOperation(None)
        self._binding.observe_run(suppressed)
        if self._binding.closed.is_set():
            raise GuiRpcError("MCP session is closed", reason="session_closed")
        consume_cancel = self._binding.consume_cancel
        if consume_cancel is not None and consume_cancel():
            return suppressed
        current = self._binding.tools.send_gui_rpc(
            "tab.get_cfg", {"tab_id": self._binding.tab}
        )
        if current["errors"]:
            raise GuiRpcError("Recipe cfg is not Valid", reason="invalid_cfg")
        actual = capture_actual(self._actual_fields())
        binding = _RunBinding(
            self._binding, RecipeRunSnapshot(self._binding.tab, actual)
        )
        operation = RunOperation(binding)
        self._binding.observe_run(operation)
        tools = self._binding.tools
        tab = self._binding.tab
        tools.send_gui_rpc("tab.snapshot", {"tab_id": tab})
        tools.send_gui_rpc("soc.info", {"include_cfg": True})
        for device in tools.send_gui_rpc("device.list", {})["devices"]:
            tools.send_gui_rpc("device.snapshot", {"name": device["name"]})

        def admit() -> None:
            if self._binding.closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            consume_cancel = self._binding.consume_cancel
            if consume_cancel is not None and consume_cancel():
                raise _StepCancelled
            binding.publish(replace(binding.capture, start_status="unknown"))

        payload: dict[str, object] = {"tab_id": tab}
        try:
            started = tools.gui.send_gui_rpc(
                "tab.run_start", payload, before_send=admit
            )
        except _StepCancelled:
            self._binding.observe_run(suppressed)
            return suppressed
        except GuiRpcError as error:
            if error.request_rejected:
                binding.publish(
                    replace(
                        binding.capture,
                        start_status="not_started",
                        start_reason=error.reason or error.code,
                    )
                )
            raise
        binding.publish(
            replace(binding.capture, op=started["handle"], start_status="running")
        )
        return operation

    def _capture_cfg_field(self, name: RecipeFieldPath, source: str | SweepSources) -> ActualField:
        node = self._node(name)
        if node.value_status != "available" or node.errors:
            raise GuiRpcError(f"Unavailable {name} summary value", reason="invalid_cfg")
        value = node.value
        if node.mode == "key":
            value = node.key
        elif isinstance(value, dict) and {"start", "stop", "n"} <= value.keys():
            value = {
                "start": value["start"],
                "stop": value["stop"],
                "expts": value["n"],
            }
        return {"value": value, "source": source}

    def _actual_fields(self) -> dict[str, ActualField]:
        fields: dict[str, ActualField] = {}
        for parameter in self._binding.summary_parameters:
            if parameter.field not in self._binding.origins:
                continue
            source = self._binding.origins[parameter.field]
            fact = self._capture_cfg_field(parameter.field, source)
            if parameter.unit is not None:
                fact["unit"] = parameter.unit
            fields[parameter.name] = fact
        return fields

    def accept(self, items: Sequence[str] | None = None) -> WritebackReceipt:
        """Write current draft items by stable target_name, Primary then Post.

        None selects all, an empty sequence selects none. Reject unknown, duplicate
        or cross-stage ambiguous names before any write. Ignore GUI checkboxes and
        resolve current IDs, not proposal snapshot IDs. Return confirmed progress;
        raise WritebackError with its receipt on the first failed stage. No refresh
        of guards, rollback, retry or matching-to-question-version is performed.
        """
        # The owner imports the receipt contracts here, so resolve it at execution.
        from zcu_tools.mcp.measure.writeback import write_current_draft

        receipt = write_current_draft(self._binding.tools, self._binding.tab, items)
        self._binding.observe_writeback(receipt)
        if receipt["status"] == "failed":
            raise WritebackError(receipt)
        return receipt


class RecipeRun:
    """Opaque capture of one tab Run, including a cancelled Run's partial result.

    The framework creates this handle at a RunOperation yield. The recipe decides
    whether to save partial data; cancellation does not make raw data saveable.
    Construction is framework-only: binding is this owner's completed Run state.
    Authors receive the handle from a yielded RunOperation.
    """

    def __init__(self, binding: _RunBinding) -> None:
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
        from zcu_tools.mcp.measure.recipe_inputs import copy_recipe_parameters

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
                raise _StepCancelled
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
        except _StepCancelled:
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
        from zcu_tools.mcp.measure.writeback import select_writeback_items

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


@dataclass(frozen=True)
class RecipeDefinition:
    """One explicitly injected recipe and its operator-facing declarations.

    name is its unique non-empty MCP tool name; description explains its workflow.
    input_schema is its handwritten schema and must match run's typed keyword
    parameters. run receives RecipeSession first and returns RecipeGenerator.
    adapter_name is the public GUI adapter ID for guide lookup. summary_parameters
    selects captured actual conditions; summary_estimates selects native scalar
    estimates. Empty summary tuples are valid. The registry validates definitions
    and schemas before any execution; this value does not register on import.
    """

    name: str
    description: str
    input_schema: RecipeInputSchema
    run: Callable[..., RecipeGenerator]
    adapter_name: str = field(kw_only=True)
    summary_parameters: tuple[SummaryParameter, ...] = field(kw_only=True)
    summary_estimates: tuple[SummaryEstimate, ...] = field(kw_only=True)
