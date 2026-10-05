"""Author-facing generator recipe contracts; GUI policy stays in the GUI owner.

Recipes receive a RecipeSession from the driver, not a transport or cfg publication.
Run, analysis and writeback questions are yielded once. The driver delivers their
status or throws a failure back into that yield expression. Session close closes
the generator, so ordinary Python finally blocks still run.

Cfg preparation uses one fixed connection and the GUI's returned publications.
The injected definitions drive these handles through the shared execution worker.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable, Generator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from threading import Condition, Event
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, TypeGuard

from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.analysis_execution import (
    AnalysisExecution,
    AnalysisWriteback,
    ExecutionSnapshot,
)
from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.images import validated_png
from zcu_tools.mcp.measure.interaction import handoff_interaction
from zcu_tools.mcp.measure.operation_wait import await_operation
from zcu_tools.mcp.measure.raw_save import RawSaveReceipt, RawSaveRequest, save_raw_data
from zcu_tools.mcp.measure.recipe_capture import (
    ActualField,
    RecipeActual,
    SweepSources,
    capture_actual,
)
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.writeback import WritebackSelection

RecipeScalar = str | int | float | bool | None
RecipeParameter = RecipeScalar | list[RecipeScalar]
StepStatus = Literal["completed", "finished_early", "cancelled"]
WritebackDecision = Literal["accepted", "skipped"]
AnalysisStage = Literal["primary", "post"]


class RecipePropertySchema(TypedDict):
    """Handwritten JSON Schema for an existing scalar or array recipe parameter.

    type lists the accepted JSON types, including null when optional. minLength
    constrains strings; items describes array elements; minItems/maxItems bound
    the array length. description is operator-facing parameter documentation.
    No annotations-to-schema generation takes place.
    """

    type: str | list[str]
    minLength: NotRequired[int]
    items: NotRequired[RecipePropertySchema]
    minItems: NotRequired[int]
    maxItems: NotRequired[int]
    description: NotRequired[str]


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

    parameter is the public keyword name; reason explains the absent calibration
    or input. Both strings must be non-empty.
    """

    parameter: str
    reason: str


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
class RunPreview:
    """Run preview path owned by this MCP session; kind is always run_preview."""

    path: str
    kind: Literal["run_preview"] = "run_preview"


@dataclass(frozen=True)
class RecipeRunSnapshot:
    """Detached Run facts for framework delivery; reading them sends no RPCs.

    tab is the GUI locator. actual is the conditions captured before native start.
    op is the opaque session Run handle, or None before its receipt. start_status
    is not_started, unknown while dispatch is uncertain, or running after receipt.
    start_reason captures a rejected start's reason, or None. outcome is the native
    Run terminal reply, or None before completion. result_state is its captured
    native data availability/provenance, or None before capture or after tab close.
    save_op is the latest admitted raw-save handle, or None before its receipt.
    raw_save retains the latest save receipt and any confirmed prefix. analyses
    maps primary/post to each stage's latest local start/completion attempt,
    including failures and partial artifacts; a suppressed start has no entry.
    preview_status is not_started, reading (also an uncertain/failed read), or
    captured. preview and preview_images retain the last confirmed session path
    and validated PNG tuple; a later failure never erases that confirmed prefix.
    """

    tab: str
    actual: RecipeActual
    op: int | None = None
    start_status: Literal["not_started", "unknown", "running"] = "not_started"
    start_reason: str | None = None
    outcome: Mapping[str, object] | None = None
    result_state: Mapping[str, object] | None = None
    raw_save: RawSaveReceipt = field(default_factory=RawSaveReceipt)
    save_op: int | None = None
    analyses: Mapping[AnalysisStage, ExecutionSnapshot] = field(default_factory=dict)
    preview_status: Literal["not_started", "reading", "captured"] = "not_started"
    preview: RunPreview | None = None
    preview_images: tuple[PngImage, ...] = ()


class RunOperation:
    """Opaque, non-iterable Run handle created by RecipeTab.run.

    Yield once to receive (RecipeRun, completed/finished_early/cancelled). A
    between-yield cancel instead delivers (None, cancelled) without sending Run.
    Failure or superseded source is thrown back at the yield expression.
    Construction is framework-only: binding is the owner-created Run state, or
    None for one suppressed start. Authors obtain handles from RecipeTab.run.
    """

    def __init__(self, binding: _RunBinding | None) -> None:
        self._binding = binding

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

    Yield once to receive (RecipeAnalysis, completed) or (None, cancelled).
    Interactive handoff does not finish the yield; GUI done resumes observation.
    Failure or superseded source is thrown back at the yield expression.
    Construction is framework-only: execution is the registered completion owner,
    or None for one suppressed start. Authors use RecipeRun.analyze.
    """

    def __init__(self, execution: AnalysisExecution | None) -> None:
        self._execution = execution

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
        analysis delivers (RecipeAnalysis, completed). Failed native operations,
        superseded sources and continuation failures raise GuiRpcError for the
        driver to throw at the yield. Snapshot/image reads retain confirmed prefix
        facts. Raise ValueError if completion already started or has no receipt.
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

    Yield once to pause until answer delivers (accepted/skipped, completed), or
    cancellation delivers (None, cancelled). Answer itself never writes data.
    Construction is framework-only: selection is the validated captured proposal.
    Authors obtain handles from RecipeRun.propose_writeback.
    """

    def __init__(self, selection: WritebackSelection) -> None:
        self._selection = deepcopy(selection)

    def snapshot(self) -> WritebackSelection:
        """Read detached stable names and proposals without a GUI request or write."""
        return deepcopy(self._selection)


class _CfgEdit(TypedDict):
    """One wire edit: path is a public field path; value is GUI input data."""

    path: list[str]
    value: object


@dataclass
class _TabBinding:
    """Cfg preparation state hidden behind an author tab handle.

    tools is the execution's fixed connection. tab is the GUI locator.
    publication is the last returned cfg observation, never refreshed on failure.
    calibrations contains opaque md values from the declared context observation.
    libraries contains the observed module-library names. origins records sources
    by public/derived field name for the later Run capture, including individual
    sweep labels. flux_unit is the asserted device unit.
    """

    tools: MeasureToolContext
    tab: str
    publication: dict[str, object]
    calibrations: dict[str, object]
    libraries: frozenset[str]
    observe_run: Callable[[RunOperation], None]
    observe_analysis: Callable[[AnalyzeOperation], None]
    observe_writeback: Callable[[WritebackReceipt], None]
    origins: dict[str, str | SweepSources] = field(default_factory=dict)
    flux_unit: str | None = None
    frequency_sweep: str | None = None
    flux_sweeps: set[str] = field(default_factory=set)
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


def _field_path(name: str) -> list[str]:
    if not name or any(not part or part != part.strip() for part in name.split(".")):
        raise ValueError("field must be a non-empty dotted cfg field name")
    return name.split(".")


def _calibration_key(calibration: Literal["resonator", "qubit"]) -> str:
    if calibration == "resonator":
        return "r_f"
    if calibration == "qubit":
        return "q_f"
    raise ValueError("calibration must be resonator or qubit")


def _check_points(expts: object) -> None:
    if expts is not None and (
        not isinstance(expts, int) or isinstance(expts, bool) or expts < 2
    ):
        raise ValueError("expts must be an integer of at least 2")


def _finite_frequency(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


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
        closed: Event | None = None,
        condition: Condition | None = None,
        consume_cancel: Callable[[], bool] | None = None,
    ) -> None:
        """Bind tools and optional execution lifetime/admission collaborators.

        closed/condition are the driver's close event and wake condition; omission
        creates local ones. consume_cancel atomically consumes one pending cancel
        before a Run/analysis start and returns whether to suppress it. This
        callback runs under the RPC lock: it must not block or send RPCs. None
        means no pending cancel. Cfg helpers and raw saves do not consume it.
        """
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

    def open_tab(self, adapter: str, *, reuse: str | None = None) -> RecipeTab:
        """Open adapter's tab, or validate/reset the named reusable tab.

        adapter is a non-empty public GUI adapter ID. reuse=None creates a fresh
        tab; otherwise reuse is a non-empty GUI locator. Busy, mismatched adapters
        and invalid locators fail without creating a fallback. Observe context
        sources once and capture the cfg ref for edits and Run. GUI/connection
        errors propagate; invalid names raise ValueError before any GUI request.
        """
        if not adapter or not adapter.strip():
            raise ValueError("adapter must be a non-empty GUI adapter ID")
        if reuse is not None and (not reuse or not reuse.strip()):
            raise ValueError("reuse must be a non-empty GUI tab locator")
        sources = self._tools.send_gui_rpc("context.snapshot", {})
        calibrations = _cfg_object(sources["md"])
        libraries = _cfg_object(_cfg_object(sources["ml"])["modules"])
        if reuse is None:
            created = self._tools.send_gui_rpc("tab.new", {"adapter_name": adapter})
            tab = created["tab_id"]
            if not isinstance(tab, str) or not tab:
                raise GuiRpcError(
                    "Invalid tab creation reply", reason="incompatible_wire"
                )
            with self._condition:
                self._tab = tab
            publication = self._tools.send_gui_rpc("tab.get_cfg", {"tab_id": tab})
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
            if any(
                interaction[key]
                for key in ("is_running", "is_analyzing", "is_saving_data")
            ):
                raise GuiRpcError("Tab is busy", reason="tab_busy")
            observed_cfg = self._tools.send_gui_rpc("tab.get_cfg", {"tab_id": tab})
            publication = self._tools.send_gui_rpc(
                "tab.reset_cfg", {"tab_id": tab, "expected": observed_cfg["cfg_ref"]}
            )
        return RecipeTab(
            _TabBinding(
                self._tools,
                tab,
                _cfg_object(publication),
                calibrations,
                frozenset(libraries),
                observe_run=self._observe_run,
                observe_analysis=self._observe_analysis,
                observe_writeback=self._observe_writeback,
                closed=self._closed,
                condition=self._condition,
                consume_cancel=self._consume_cancel,
            )
        )


class RecipeTab:
    """Opaque tab handle returned by RecipeSession.open_tab.

    field names are dot-separated public cfg fields (for example rounds or
    modules.readout), not wire path lists. Methods preserve the observed cfg ref
    and record parameter sources. GUI errors propagate without re-read, retry or
    rollback; required calibration gaps raise RecipeNeedsParameters. Authors obtain
    this handle from open_tab rather than constructing the private binding.
    """

    def __init__(self, binding: _TabBinding) -> None:
        self._binding = binding

    def _node(self, name: str) -> dict[str, object]:
        node = _cfg_object(self._binding.publication["tree"])
        for part in _field_path(name):
            children = _cfg_object(node.get("children", {}))
            if part not in children:
                raise ValueError(f"Unknown cfg field: {name}")
            node = _cfg_object(children[part])
        return node

    def _edit(self, edits: list[_CfgEdit]) -> None:
        if edits:
            reply = self._binding.tools.send_gui_rpc(
                "tab.edit_cfg",
                {
                    "tab_id": self._binding.tab,
                    "expected": self._binding.publication["cfg_ref"],
                    "edits": edits,
                },
            )
            self._binding.publication = _cfg_object(reply)

    def _reference(self, field: str) -> dict[str, object]:
        node = self._node(field)
        if node["kind"] != "reference":
            raise ValueError(f"Not a module reference field: {field}")
        return node

    def use_library(
        self, field: str, name: str | None, *, required: str | None = None
    ) -> None:
        """Select name for a module field; None retains its current GUI value.

        required names the missing tool parameter if no usable reference exists.
        Unknown or invalid explicit library references fail rather than fall back.
        """
        node = self._reference(field)
        if name is not None:
            if not name or not name.strip():
                raise ValueError("library name must be non-empty")
            self._edit([{"path": _field_path(field), "value": {"__ref": name}}])
            node = self._reference(field)
            if node.get("error") or not node.get("valid") or node.get("ref") != name:
                raise GuiRpcError(f"Invalid {field} reference", reason="invalid_cfg")
        elif required is not None and (
            not isinstance(node.get("ref"), str)
            or node.get("ref") not in self._binding.libraries
            or not node.get("valid")
            or node.get("error")
        ):
            raise RecipeNeedsParameters(
                (
                    MissingParameter(
                        required, f"Provide a valid {field} library reference"
                    ),
                )
            )
        self._binding.origins[field] = "explicit" if name is not None else "gui_default"

    def disable_library(self, field: str) -> None:
        """Disable an optional module field explicitly; reject unknown fields."""
        self._reference(field)
        self._edit([{"path": _field_path(field), "value": {"__ref": None}}])
        node = self._reference(field)
        if node.get("ref") is not None or node.get("error"):
            raise GuiRpcError(f"Cannot disable {field}", reason="invalid_cfg")
        self._binding.origins[field] = "disabled"

    def set(
        self, field: str, value: RecipeParameter | None, *, source: str = "explicit"
    ) -> None:
        """Set a scalar cfg field; None keeps its GUI default and records that source.

        Reject unknown/non-scalar fields locally. The GUI validates supplied values;
        a returned field error raises GuiRpcError without retrying the edit.
        source is a non-empty provenance label for supplied values (for example
        their tool parameter name); None values always retain gui_default.
        """
        if not source or not source.strip():
            raise ValueError("source must be a non-empty provenance label")
        node = self._node(field)
        if node["kind"] != "scalar":
            raise ValueError(f"Not a scalar cfg field: {field}")
        if value is not None:
            self._edit([{"path": _field_path(field), "value": value}])
            node = self._node(field)
            state = _cfg_object(node["input"])
            if (
                not node.get("valid")
                or state.get("error")
                or state.get("validation_error")
            ):
                raise GuiRpcError(f"Invalid {field} value", reason="invalid_cfg")
        self._binding.origins[field] = source if value is not None else "gui_default"

    def _frequency_fields(self, field: str) -> tuple[list[str], str | None]:
        """Resolve scalar/readout-root fields and the enclosing library reference."""
        node = self._node(field)
        if node["kind"] == "reference":
            children = _cfg_object(node["children"])
            if "ro_freq" in children:
                paths = [f"{field}.ro_freq"]
            elif "pulse_cfg" in children and "ro_cfg" in children:
                paths = [f"{field}.pulse_cfg.freq", f"{field}.ro_cfg.ro_freq"]
            else:
                raise GuiRpcError("Unsupported readout shape", reason="invalid_cfg")
            reference = node.get("ref")
            return paths, reference if isinstance(reference, str) else None
        if node["kind"] != "scalar":
            raise ValueError(f"Not a frequency field: {field}")
        ancestors = _field_path(field)[:-1]
        while ancestors:
            parent = self._node(".".join(ancestors))
            if parent["kind"] == "reference":
                reference = parent.get("ref")
                return [field], reference if isinstance(reference, str) else None
            ancestors.pop()
        return [field], None

    def _usable_frequency(self, field: str) -> bool:
        node = self._node(field)
        state = _cfg_object(node["input"])
        return bool(
            node.get("valid")
            and not state.get("error")
            and not state.get("validation_error")
            and _finite_frequency(state.get("resolved"))
        )

    def set_frequency(
        self,
        field: str,
        frequency_mhz: float | None = None,
        *,
        calibration: Literal["resonator", "qubit"],
        prefer_library: bool = True,
        required: str,
        source: str = "explicit",
    ) -> None:
        """Set a frequency scalar or readout module root in MHz.

        Choose explicit finite frequency_mhz, then usable current library when
        prefer_library, then the named calibration. If none exists, report required
        as a missing tool parameter. Handle pulse and single-frequency readout cfg
        in this helper; callers do not inspect the GUI's nested representation.
        source is the non-empty provenance label for explicit frequency_mhz;
        calibrated and library inputs retain their original source labels.
        """
        if not source or not source.strip():
            raise ValueError("source must be a non-empty provenance label")
        key = _calibration_key(calibration)
        if frequency_mhz is not None and not _finite_frequency(frequency_mhz):
            raise ValueError("frequency_mhz must be finite and not boolean")
        paths, reference = self._frequency_fields(field)
        calibrated = self._binding.calibrations.get(key)
        edits: list[_CfgEdit] = []
        origins: dict[str, str] = {}
        for path in paths:
            if frequency_mhz is not None:
                edits.append({"path": _field_path(path), "value": frequency_mhz})
                origins[path] = source
            elif (
                prefer_library
                and reference in self._binding.libraries
                and self._usable_frequency(path)
            ):
                origins[path] = f"library:{reference}"
            elif _finite_frequency(calibrated):
                edits.append({"path": _field_path(path), "value": {"__expr": key}})
                origins[path] = key
            else:
                raise RecipeNeedsParameters(
                    (
                        MissingParameter(
                            required,
                            f"Provide an explicit frequency, valid library, or calibrated {key}",
                        ),
                    )
                )
        self._edit(edits)
        for path in paths:
            if not self._usable_frequency(path):
                raise GuiRpcError(f"Invalid {path} frequency", reason="invalid_cfg")
        self._binding.origins.update(origins)

    def _sweep_inputs(self, field: str) -> dict[str, object]:
        node = self._node(field)
        if node["kind"] != "sweep":
            raise ValueError(f"Not a sweep cfg field: {field}")
        return _cfg_object(node["inputs"])

    def _write_sweep(self, field: str, values: dict[str, object]) -> None:
        self._sweep_inputs(field)
        if not values:
            return
        self._edit([{"path": _field_path(field), "value": values}])
        inputs = self._sweep_inputs(field)
        if not self._node(field).get("valid") or any(
            _cfg_object(inputs[key]).get("error")
            or _cfg_object(inputs[key]).get("validation_error")
            for key in values
        ):
            raise GuiRpcError(f"Invalid {field} sweep", reason="invalid_cfg")

    def _linewidth_span(
        self, field: str, center_key: str, width_key: str
    ) -> str | None:
        """Preserve the GUI's width expression instead of choosing a new span."""
        width = self._binding.calibrations.get(width_key)
        inputs = self._sweep_inputs(field)
        edges = [_cfg_object(inputs[key]) for key in ("start", "stop")]
        if not _finite_frequency(width) or width <= 0:
            return None
        expressions: list[str] = []
        for edge in edges:
            raw = edge["raw"]
            if (
                edge["mode"] != "expression"
                or not isinstance(raw, str)
                or not re.search(rf"\b{width_key}\b", raw)
            ):
                return None
            expressions.append(re.sub(rf"\b{center_key}\b", "0", raw))
        return f"({expressions[1]}) - ({expressions[0]})"

    def set_frequency_sweep(
        self,
        field: str,
        *,
        calibration: Literal["resonator", "qubit"],
        center_mhz: float | None = None,
        span_mhz: float | None = None,
        expts: int | None = None,
    ) -> None:
        """Set MHz sweep endpoints/points, preserving omitted GUI-derived inputs.

        Missing center uses the named frequency calibration. Missing span retains
        its calibrated linewidth expression; a supplied span must be positive.
        expts is an integer of at least 2; None retains the GUI count. Missing
        calibration/linewidth raises RecipeNeedsParameters. Invalid explicit
        inputs raise ValueError; invalid resolved ranges raise GuiRpcError.
        """
        key = _calibration_key(calibration)
        width_key = "rf_w" if calibration == "resonator" else "qf_w"
        _check_points(expts)
        if center_mhz is not None and not _finite_frequency(center_mhz):
            raise ValueError("center_mhz must be finite and not boolean")
        if span_mhz is not None and (not _finite_frequency(span_mhz) or span_mhz <= 0):
            raise ValueError("span_mhz must be finite and positive")
        self._sweep_inputs(field)
        missing: list[MissingParameter] = []
        center: str | float = key if center_mhz is None else center_mhz
        if center_mhz is None and not _finite_frequency(
            self._binding.calibrations.get(key)
        ):
            missing.append(
                MissingParameter("center_mhz", f"No finite {key} calibration")
            )
        span: str | float | None = span_mhz
        if span_mhz is None:
            span = self._linewidth_span(field, key, width_key)
            if span is None:
                missing.append(
                    MissingParameter("span_mhz", "No GUI linewidth-derived range")
                )
        if missing:
            raise RecipeNeedsParameters(tuple(missing))
        values: dict[str, object] = {
            "start": {"__expr": f"({center}) - ({span}) / 2"},
            "stop": {"__expr": f"({center}) + ({span}) / 2"},
        }
        if expts is not None:
            values["expts"] = expts
        self._write_sweep(field, values)
        inputs = self._sweep_inputs(field)
        start = _cfg_object(inputs["start"])["resolved"]
        stop = _cfg_object(inputs["stop"])["resolved"]
        if (
            not _finite_frequency(start)
            or not _finite_frequency(stop)
            or stop <= start
            or not _finite_frequency(stop - start)
        ):
            raise GuiRpcError("Invalid resolved frequency range", reason="invalid_cfg")
        self._binding.frequency_sweep = field
        self._binding.origins[field] = (
            "gui_calibration" if center_mhz is None and span_mhz is None else "explicit"
        )
        self._binding.origins["center_mhz"] = key if center_mhz is None else "explicit"
        self._binding.origins["span_mhz"] = (
            "gui_linewidth" if span_mhz is None else "explicit"
        )
        self._binding.origins[f"{field}.expts"] = (
            "gui_default" if expts is None else "explicit"
        )

    def set_flux_sweep(
        self,
        field: str,
        *,
        start: float | None = None,
        stop: float | None = None,
        expts: int | None = None,
    ) -> None:
        """Set flux sweep coordinates without converting their device unit.

        start/stop must both be finite, distinct or both omitted. Omitted endpoints
        retain valid flx_half/flx_int expressions; absent calibration raises
        RecipeNeedsParameters. expts is an integer of at least 2; None retains
        the GUI count. Invalid explicit inputs raise ValueError; invalid resolved
        ranges raise GuiRpcError.
        """
        _check_points(expts)
        if (start is None) != (stop is None) or (
            start is not None
            and stop is not None
            and (
                not _finite_frequency(start)
                or not _finite_frequency(stop)
                or start == stop
            )
        ):
            raise ValueError(
                "flux endpoints must both be finite and distinct, or both omitted"
            )
        inputs = self._sweep_inputs(field)
        if start is None:
            half = self._binding.calibrations.get("flx_half")
            integer = self._binding.calibrations.get("flx_int")
            if (
                not _finite_frequency(half)
                or not _finite_frequency(integer)
                or half == integer
            ):
                raise RecipeNeedsParameters(
                    (
                        MissingParameter(
                            "flux_range", "No distinct calibrated flux endpoints"
                        ),
                    )
                )
            for key in ("start", "stop"):
                state = _cfg_object(inputs[key])
                raw = state["raw"]
                if (
                    state["mode"] != "expression"
                    or not isinstance(raw, str)
                    or set(re.findall(r"\bflx_(?:half|int)\b", raw))
                    != {"flx_half", "flx_int"}
                ):
                    raise RecipeNeedsParameters(
                        (
                            MissingParameter(
                                "flux_range", "No GUI calibration-derived range"
                            ),
                        )
                    )
        self.set_sweep(field, start=start, stop=stop, expts=expts)
        inputs = self._sweep_inputs(field)
        resolved_start = _cfg_object(inputs["start"])["resolved"]
        resolved_stop = _cfg_object(inputs["stop"])["resolved"]
        if (
            not _finite_frequency(resolved_start)
            or not _finite_frequency(resolved_stop)
            or resolved_start == resolved_stop
        ):
            raise GuiRpcError("Invalid resolved flux range", reason="invalid_cfg")
        self._binding.flux_sweeps.add(field)
        self._binding.origins[field] = (
            "gui_calibration" if start is None else "explicit"
        )

    def _flux_unit(
        self, snapshot: dict[str, object], requested: str | None, *, explicit: bool
    ) -> str:
        unit = snapshot["unit"]
        info_value = snapshot.get("info")
        info = _cfg_object(info_value) if info_value is not None else {}
        native = (
            unit == "none"
            and snapshot.get("type_name") == "FakeDevice"
            and info.get("type") == "FakeDevice"
            and requested == "native"
        )
        if native:
            return "native"
        if (
            snapshot.get("type_name") == "FakeDevice"
            or info.get("type") == "FakeDevice"
        ):
            raise GuiRpcError(
                "Fake flux requires confirmed native coordinates",
                reason="invalid_device",
            )
        if requested is not None and (requested != unit or requested == "native"):
            raise GuiRpcError(
                "Flux unit does not match the device coordinate",
                reason="invalid_device",
            )
        if not isinstance(unit, str) or not unit.strip() or unit in ("none", "native"):
            if explicit:
                raise GuiRpcError(
                    "Flux device has no physical unit", reason="invalid_device"
                )
            raise RecipeNeedsParameters(
                (MissingParameter("flux_device", "Flux device has no physical unit"),)
            )
        return unit

    def use_flux_device(
        self, name: str | None = None, *, unit: str | None = None
    ) -> None:
        """Select explicit name or device.flux.name; assert unit without conversion.

        Missing default source raises RecipeNeedsParameters. Unknown devices and
        unit mismatches fail as invalid_device. native is allowed only for a
        confirmed FakeDevice with original unit none and explicit unit="native".
        """
        if name is not None and (not name or not name.strip()):
            raise ValueError("flux device name must be non-empty")
        if unit is not None and (not unit or not unit.strip()):
            raise ValueError("flux unit must be non-empty")
        self._node("dev.flux_dev")
        device: object = name
        if name is None:
            values = self._binding.tools.send_gui_rpc("value.list", {})["values"]
            if any(_cfg_object(value)["key"] == "device.flux.name" for value in values):
                device = self._binding.tools.send_gui_rpc(
                    "value.read", {"key": "device.flux.name"}
                )["value"]
        if not isinstance(device, str) or not device.strip():
            raise RecipeNeedsParameters(
                (MissingParameter("flux_device", "No registered flux device source"),)
            )
        observed = self._binding.tools.send_gui_rpc("device.snapshot", {"name": device})
        resolved_unit = self._flux_unit(
            _cfg_object(observed["snapshot"]), unit, explicit=name is not None
        )
        self.set("dev.flux_dev", device)
        self._binding.origins["dev.flux_dev"] = (
            "device.flux.name" if name is None else "explicit"
        )
        self._binding.flux_unit = resolved_unit

    def set_sweep(
        self,
        field: str,
        *,
        start: float | None = None,
        stop: float | None = None,
        expts: int | None = None,
        sources: SweepSources | None = None,
    ) -> None:
        """Edit only supplied sweep fields in their native unit; retain omitted ones.

        Supplied endpoints must be finite and not boolean. expts is an integer
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
            if endpoint is not None and not _finite_frequency(endpoint):
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
            self._binding.origins[f"{field}.{key}"] = captured_sources[key]
        if sources is not None:
            self._binding.origins[field] = captured_sources
        else:
            self._binding.origins[field] = "explicit" if values else "gui_default"

    def run(self) -> RunOperation:
        """Immediately send Run with captured cfg ref and return its yield handle.

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
        if self._binding.publication["status"] != "Valid":
            raise GuiRpcError("Recipe cfg is not Valid", reason="invalid_cfg")
        actual = capture_actual(self._binding.publication, self._actual_fields())
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

        try:
            started = tools.gui.send_gui_rpc(
                "tab.run_start",
                {"tab_id": tab, "expected": actual["cfg_ref"]},
                before_send=admit,
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

    def _capture_cfg_field(self, name: str, source: str | SweepSources) -> ActualField:
        """Capture one scalar, reference, sweep or sweep member from the publication."""
        parts = _field_path(name)
        parent = self._node(".".join(parts[:-1])) if len(parts) > 1 else None
        if parent is not None and parent["kind"] == "sweep":
            state = _cfg_object(_cfg_object(parent["inputs"])[parts[-1]])
            return {"value": state["resolved"], "input": state, "source": source}
        node = self._node(name)
        if node["kind"] == "scalar":
            state = _cfg_object(node["input"])
            if name == "dev.flux_dev":
                # Device identity carries unit/source, not a numeric cfg input.
                return {"value": state["resolved"], "source": source}
            return {"value": state["resolved"], "input": state, "source": source}
        if node["kind"] == "reference":
            return {"value": node["ref"], "source": source}
        if node["kind"] == "sweep":
            inputs = _cfg_object(node["inputs"])
            return {
                "value": {
                    key: _cfg_object(inputs[key])["resolved"]
                    for key in ("start", "stop", "expts")
                },
                "input": inputs,
                "source": source,
            }
        raise ValueError(f"Not a capturable cfg field: {name}")

    def _actual_fields(self) -> dict[str, ActualField]:
        fields = {
            name: self._capture_cfg_field(name, source)
            for name, source in self._binding.origins.items()
            if name not in ("center_mhz", "span_mhz")
        }
        frequency_sweep = self._binding.frequency_sweep
        if frequency_sweep is not None:
            inputs = self._sweep_inputs(frequency_sweep)
            start = _cfg_object(inputs["start"])["resolved"]
            stop = _cfg_object(inputs["stop"])["resolved"]
            if not _finite_frequency(start) or not _finite_frequency(stop):
                raise GuiRpcError(
                    "Invalid resolved frequency range", reason="invalid_cfg"
                )
            span = stop - start
            if not _finite_frequency(span) or span <= 0:
                raise GuiRpcError(
                    "Invalid resolved frequency span", reason="invalid_cfg"
                )
            fields["center_mhz"] = {
                "value": start + span / 2,
                "source": self._binding.origins["center_mhz"],
            }
            fields["span_mhz"] = {
                "value": span,
                "source": self._binding.origins["span_mhz"],
            }
        if self._binding.flux_unit is not None:
            for name in (*self._binding.flux_sweeps, "dev.flux_dev"):
                if name in fields:
                    fields[name]["unit"] = self._binding.flux_unit
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

    def save_raw(self) -> RawSaveReceipt:
        """Save this Run's raw data synchronously and return its actual receipt.

        Reject superseded or unavailable source data; await an admitted save's
        true outcome even when cancel arrives. Failures retain confirmed prefix
        facts and propagate. A reserved path is never reported as a saved path.
        """
        binding = self._binding
        run_op = binding.run_source()
        capture = binding.snapshot()

        def observe(op: int | None, receipt: RawSaveReceipt) -> None:
            binding.publish(replace(binding.capture, raw_save=receipt, save_op=op))

        def admit() -> None:
            if binding.tab.closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")

        return save_raw_data(
            binding.tab.tools.gui,
            RawSaveRequest(capture.tab, run_op, admit),
            closed=binding.tab.closed,
            condition=binding.tab.condition,
            previous=capture.raw_save,
            observe=observe,
        )

    def preview(self) -> None:
        """Capture this Run's validated PNG in session storage, without analysis.

        This synchronous fast step does not consume pending cancel. Missing Run
        data, superseded source, closed session and GUI/PNG/storage failures raise
        at this call. Failed repeated reads retain the prior confirmed path/image.
        snapshot exposes local delivery facts; preview files live until session
        close and are not persistent saved artifacts. No retry or guard refresh.
        """
        binding = self._binding
        run_op = binding.run_source()
        binding.publish(replace(binding.capture, preview_status="reading"))

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
        path = binding.tab.tools.session.write_png(image.data)
        binding.publish(
            replace(
                binding.capture,
                preview_status="captured",
                preview=RunPreview(str(path)),
                preview_images=(image,),
            )
        )

    def analyze(
        self,
        stage: AnalysisStage,
        *,
        params: Mapping[str, RecipeParameter] | None = None,
    ) -> AnalyzeOperation:
        """Immediately start primary/post analysis bound to this Run's source.

        params supplies named scalar/one-level-array GUI updates; None keeps defaults.
        Invalid stage or unsupported/non-finite parameters raise ValueError before
        GUI dispatch. Missing Run data or completed Primary raises GuiRpcError.
        Post uses this Run's completed Primary source. Pending between-yield cancel
        suppresses this one start and is consumed once. Rejection fails immediately;
        the yielded handle delivers cancellation or throws completion failures.
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
        question. Yield the handle to await accepted/skipped; no timeout or write
        is implied. The recipe must call tab.accept explicitly to write anything.
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
RecipeDelivery = tuple[
    RecipeRun | RecipeAnalysis | WritebackDecision | None, StepStatus
]
RecipeGenerator = Generator[RecipeOperation, RecipeDelivery, None]


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
