"""Author-facing generator recipe contracts; GUI policy stays in the GUI owner.

Recipes receive a RecipeSession from the driver, not a transport or cfg publication.
Run, analysis and writeback questions are yielded once. The driver delivers their
status or throws a failure back into that yield expression. Session close closes
the generator, so ordinary Python finally blocks still run.

Method bodies are declaration stubs while the Orchestrator prepares the caller
and formal execution tests. This module is not yet wired into tool assembly.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

from zcu_tools.mcp.measure.analysis_execution import (
    AnalysisWriteback,
    ExecutionSnapshot,
)
from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter

RecipeScalar = str | int | float | bool | None
RecipeParameter = RecipeScalar | list[RecipeScalar]
StepStatus = Literal["completed", "finished_early", "cancelled"]
WritebackDecision = Literal["accepted", "skipped"]
AnalysisStage = Literal["primary", "post"]


class RecipePropertySchema(TypedDict):
    """Handwritten JSON Schema for an existing scalar or array recipe parameter.

    type lists the accepted JSON types, including null when optional. minLength
    constrains strings; items describes array elements; minItems constrains the
    array length. No annotations-to-schema generation takes place.
    """

    type: str | list[str]
    minLength: NotRequired[int]
    items: NotRequired[RecipePropertySchema]
    minItems: NotRequired[int]


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


@dataclass(frozen=True)
class RawSaveReceipt:
    """Capture of an admitted raw save, not a promise about a reserved path.

    status is not_started, saving, saved, failed or unknown. reserved_path is the
    planned file path; path is a confirmed saved path. None means not captured.
    operation_outcome retains the native terminal reply, or None before capture.
    Native fields are opaque diagnostic facts, not a second operation policy.
    """

    status: Literal["not_started", "saving", "saved", "failed", "unknown"] = (
        "not_started"
    )
    reserved_path: str | None = None
    path: str | None = None
    operation_outcome: Mapping[str, object] | None = None


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


class RunOperation:
    """Opaque, non-iterable Run handle created by RecipeTab.run.

    Yield once to receive (RecipeRun, completed/finished_early/cancelled). A
    between-yield cancel instead delivers (None, cancelled) without sending Run.
    Failure or superseded source is thrown back at the yield expression.
    """


class AnalyzeOperation:
    """Opaque, non-iterable analysis handle created by RecipeRun.analyze.

    Yield once to receive (RecipeAnalysis, completed) or (None, cancelled).
    Interactive handoff does not finish the yield; GUI done resumes observation.
    Failure or superseded source is thrown back at the yield expression.
    """


class WritebackQuestion:
    """Opaque, non-iterable captured question created by propose_writeback.

    Yield once to pause until answer delivers (accepted/skipped, completed), or
    cancellation delivers (None, cancelled). Answer itself never writes data.
    """


class RecipeSession:
    """One execution's fixed GUI binding, created and passed by the driver.

    Helpers never reconnect, retry stale requests or synthesize GUI defaults.
    Fast edits remain available after cancellation; the recipe decides its policy.
    """

    def open_tab(self, adapter: str, *, reuse: str | None = None) -> RecipeTab:
        """Open adapter's tab, or validate/reset the named reusable tab.

        adapter is a public GUI adapter ID. reuse=None creates a fresh tab. Busy,
        mismatched adapters and invalid locators fail without creating a fallback.
        Capture the cfg ref for later edits and Run; propagate GUI/connection errors.
        """
        raise NotImplementedError("recipe session implementation is not prepared")


class RecipeTab:
    """Opaque tab handle returned by RecipeSession.open_tab.

    field names are public cfg fields, not wire path lists. Methods preserve the
    observed cfg ref and record parameter sources. GUI errors propagate without
    re-read, retry or rollback; required calibration gaps raise RecipeNeedsParameters.
    """

    def use_library(
        self, field: str, name: str | None, *, required: str | None = None
    ) -> None:
        """Select name for a module field; None retains its current GUI value.

        required names the missing tool parameter if no usable reference exists.
        Unknown or invalid explicit library references fail rather than fall back.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

    def disable_library(self, field: str) -> None:
        """Disable an optional module field explicitly; reject unknown fields."""
        raise NotImplementedError("recipe tab implementation is not prepared")

    def set(self, field: str, value: RecipeParameter | None) -> None:
        """Set a scalar cfg field; None keeps its GUI default and records that source.

        Reject unknown fields and values outside the GUI field's contract.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

    def set_frequency(
        self,
        field: str,
        frequency_mhz: float | None = None,
        *,
        calibration: Literal["resonator", "qubit"],
        prefer_library: bool = True,
        required: str,
    ) -> None:
        """Set a frequency scalar or readout module root in MHz.

        Choose explicit finite frequency_mhz, then usable current library when
        prefer_library, then the named calibration. If none exists, report required
        as a missing tool parameter. Handle pulse and single-frequency readout cfg
        in this helper; callers do not inspect the GUI's nested representation.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

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
        its calibrated linewidth expression. Missing calibration/linewidth raises
        RecipeNeedsParameters; non-finite inputs or invalid points raise ValueError.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

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
        RecipeNeedsParameters. Invalid endpoints or points raise ValueError.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

    def use_flux_device(
        self, name: str | None = None, *, unit: str | None = None
    ) -> None:
        """Select explicit name or device.flux.name; assert unit without conversion.

        Missing default source raises RecipeNeedsParameters. Unknown devices and
        unit mismatches fail as invalid_device. native is allowed only for a
        confirmed FakeDevice with original unit none and explicit unit="native".
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

    def set_sweep(
        self,
        field: str,
        *,
        start: float | None = None,
        stop: float | None = None,
        expts: int | None = None,
    ) -> None:
        """Edit only supplied sweep fields in their native unit; retain omitted ones.

        Supplied endpoints must be finite; expts must satisfy the GUI field's
        valid point range. Unknown fields and invalid values fail before Run.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

    def run(self) -> RunOperation:
        """Immediately send Run with captured cfg ref and return its yield handle.

        Capture actual conditions and result provenance. A pending between-yield
        cancel suppresses this one start and is consumed once. GUI rejection fails
        immediately; completion failures are thrown by the driver at the yield.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

    def accept(self, items: Sequence[str] | None = None) -> WritebackReceipt:
        """Write current draft items by stable target_name, Primary then Post.

        None selects all, an empty sequence selects none. Reject unknown, duplicate
        or cross-stage ambiguous names before any write. Ignore GUI checkboxes and
        resolve current IDs, not proposal snapshot IDs. Return confirmed progress;
        raise WritebackError with its receipt on the first failed stage. No refresh
        of guards, rollback, retry or matching-to-question-version is performed.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")


class RecipeRun:
    """Opaque capture of one tab Run, including a cancelled Run's partial result.

    The framework creates this handle at a RunOperation yield. The recipe decides
    whether to save partial data; cancellation does not make raw data saveable.
    """

    def save_raw(self) -> RawSaveReceipt:
        """Save this Run's raw data synchronously and return its actual receipt.

        Reject superseded or unavailable source data; await an admitted save's
        true outcome even when cancel arrives. Failures retain confirmed prefix
        facts and propagate. A reserved path is never reported as a saved path.
        """
        raise NotImplementedError("recipe run implementation is not prepared")

    def analyze(
        self,
        stage: AnalysisStage,
        *,
        params: Mapping[str, RecipeParameter] | None = None,
    ) -> AnalyzeOperation:
        """Immediately start primary/post analysis bound to this Run's source.

        params supplies named GUI analysis parameter updates; None keeps defaults.
        Post uses this Run's completed Primary source. Pending between-yield cancel
        suppresses this one start and is consumed once. Rejection fails immediately;
        the yielded handle delivers cancellation or throws completion failures.
        """
        raise NotImplementedError("recipe run implementation is not prepared")

    def propose_writeback(
        self, items: Sequence[str] | None = None
    ) -> WritebackQuestion:
        """Capture selected candidates from completed analyses without another read.

        items uses stable target_name, None selects all, empty selects none. Reject
        unknown, repeated or cross-stage ambiguous names before publishing the
        question. Yield the handle to await accepted/skipped; no timeout or write
        is implied. The recipe must call tab.accept explicitly to write anything.
        """
        raise NotImplementedError("recipe run implementation is not prepared")


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
