"""Missing-parameter and shared standalone writeback receipts."""

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from zcu_tools.mcp.measure.analysis_execution import AnalysisWriteback

AnalysisStage = Literal["primary", "post"]


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
