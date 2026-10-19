"""Shared recipe inputs, detached captures and writeback receipts.

These contracts do not depend on authoring or operation handles. Input codecs,
writeback and the execution handles share these same objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

from zcu_tools.format_version import YamlValue
from zcu_tools.mcp.core.reply import PngImage
from zcu_tools.mcp.measure.analysis_execution import (
    AnalysisWriteback,
    ExecutionSnapshot,
)
from zcu_tools.mcp.measure.raw_save import SaveReport
from zcu_tools.mcp.measure.recipe_capture import RecipeActual
from zcu_tools.mcp.measure.session import GuiRpcError

RecipeScalar = str | int | float | bool | None
RecipeParameter = RecipeScalar | list[RecipeScalar]
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

    field is a dotted public scalar cfg path. value is a direct scalar/list or
    RecipeExpression; None retains the current input, it does not write null.
    source is a non-blank provenance label; retained inputs use gui_default.
    """

    field: str
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


def recipe_gui_object(value: object) -> dict[str, object]:
    """Shallow-copy a GUI wire dict with string keys, retaining its values.

    value must be a dict, including an empty one; nested values remain opaque.
    Raise GuiRpcError(incompatible_wire) for non-dicts or non-string keys.
    No recursive decoding, mutation, GUI request or policy validation occurs.
    """
    if not isinstance(value, dict):
        raise GuiRpcError("Expected a GUI object", reason="incompatible_wire")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise GuiRpcError("Expected GUI string keys", reason="incompatible_wire")
        result[key] = item
    return result
