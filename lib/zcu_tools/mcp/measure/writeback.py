"""Stable-name selection and confirmed progress over GUI-owned writeback drafts."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from zcu_tools.mcp.measure.analysis_execution import AnalysisWriteback
from zcu_tools.mcp.measure.recipe import (
    AnalysisStage,
    RecipeWritebackPreview,
    WritebackItemReceipt,
    WritebackReceipt,
)
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

_PANES: tuple[tuple[AnalysisStage, Literal["analysis", "post_analysis"], str], ...] = (
    ("primary", "analysis", "tab.get_analyze_result"),
    ("post", "post_analysis", "tab.get_post_analyze_result"),
)


@dataclass(frozen=True)
class WritebackSelection:
    """Detached selected proposals and stable names in Primary-then-Post order.

    items names exactly the selected target_name values, never GUI session IDs.
    preview keeps each stage's captured draft and destination context, filtering
    only its items. It is not a guard observation or permission for a later write.
    """

    items: tuple[str, ...]
    preview: RecipeWritebackPreview


def _tab_locator(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("tab must be a non-empty string")
    return value


def _requested_names(items: Sequence[object] | None) -> tuple[str, ...] | None:
    if items is None:
        return None
    if isinstance(items, str):
        raise ValueError("items must be a sequence of non-empty stable names")
    names: list[str] = []
    for name in items:
        if not isinstance(name, str) or not name:
            raise ValueError("items must be a sequence of non-empty stable names")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate writeback item names")
    return tuple(names)


def _name(item: dict[str, object]) -> str:
    name = item["target_name"]
    if not isinstance(name, str) or not name:
        raise GuiRpcError("Invalid writeback target name", reason="incompatible_wire")
    return name


def _filtered(
    preview: AnalysisWriteback | None, names: set[str]
) -> AnalysisWriteback | None:
    if preview is None:
        return None
    copied = deepcopy(preview)
    copied["items"] = (
        [item for item in copied["items"] if _name(item) in names]
        if copied["has_draft"]
        else []
    )
    return copied


def select_writeback_items(
    preview: RecipeWritebackPreview, items: Sequence[str] | None = None
) -> WritebackSelection:
    """Filter completed captures by stable target_name without any GUI query.

    None selects all candidates; an empty sequence selects none. GUI selected
    flags do not affect selection. Reject duplicate requested names, any ambiguous
    name in the captured panes, or an unknown requested name with ValueError.
    Malformed native names raise GuiRpcError. Return a detached preview and stable
    names in pane/item order; do not mutate the captures or refresh observations.
    """
    requested = _requested_names(items)
    available: list[str] = []
    for stage in (preview.primary, preview.post):
        if stage is not None and stage["has_draft"]:
            available.extend(_name(item) for item in stage["items"])
    if len(available) != len(set(available)):
        raise ValueError("Ambiguous writeback item names across the captured panes")
    if requested is not None and set(requested) - set(available):
        unknown = sorted(set(requested) - set(available))
        raise ValueError(f"Unknown writeback item names: {', '.join(unknown)}")
    selected = set(available if requested is None else requested)
    return WritebackSelection(
        tuple(name for name in available if name in selected),
        RecipeWritebackPreview(
            _filtered(preview.primary, selected), _filtered(preview.post, selected)
        ),
    )


def _read_current_preview(
    ctx: MeasureToolContext, tab: str, subtab: Literal["analysis", "post_analysis"]
) -> AnalysisWriteback:
    """Detach the current preview and reject malformed stable names at this stage."""
    native = ctx.send_gui_rpc(
        "tab.writeback_preview", {"tab_id": tab, "subtab_id": subtab}
    )
    preview = AnalysisWriteback(
        has_draft=native["has_draft"],
        items=deepcopy(native["items"]),
        destination_context=deepcopy(native["destination_context"]),
    )
    if preview["has_draft"]:
        for item in preview["items"]:
            _name(item)
    return preview


def _skip_stage(receipt: WritebackReceipt, stage: AnalysisStage) -> None:
    """Retain a confirmed empty stage once, in canonical pane order."""
    if stage in receipt["skipped"]:
        return
    receipt["skipped"] = [
        entry for entry, _, _ in _PANES if entry == stage or entry in receipt["skipped"]
    ]
    receipt["not_started"].remove(stage)


def write_current_draft(
    ctx: MeasureToolContext, tab: str, items: Sequence[str] | None = None
) -> WritebackReceipt:
    """Write selected current candidates once, Primary then Post, without rollback.

    tab is a non-empty GUI locator. items uses stable target_name, not proposal IDs
    or checkbox flags; None means all and an empty sequence means none. Validate
    the arguments before binding, then read both current panes before any write so
    unknown/ambiguous names cannot cause a partial write. Never reuse proposal IDs.

    Return GUI-confirmed completed stages, skipped stages and not_started stages.
    A GuiRpcError stops the first failed step and returns a failed receipt. The
    partial-writes flag is true only after admission of that stage's write RPC.
    Unexpected errors propagate. No guard refresh, retry or rollback is performed;
    callers must already have the GUI observations required by writeback guards.
    """
    tab = _tab_locator(tab)
    requested = _requested_names(items)
    receipt: WritebackReceipt = {
        "tab": tab,
        "status": "finished",
        "completed": [],
        "skipped": [],
        "not_started": ["primary", "post"],
    }
    stage: AnalysisStage = "primary"
    write_attempted = False

    def admit_write() -> None:
        nonlocal write_attempted
        write_attempted = True

    previews: dict[AnalysisStage, AnalysisWriteback | None] = {
        "primary": None,
        "post": None,
    }
    try:
        ctx = ctx.bound()
        # These reads do not reveal guarded resources or authorize writes.
        present: dict[AnalysisStage, bool] = {}
        for stage, _, method in _PANES:
            present[stage] = (
                ctx.send_gui_rpc(method, {"tab_id": tab})["summary"] is not None
            )
        for stage, subtab, _ in _PANES:
            if present[stage]:
                previews[stage] = _read_current_preview(ctx, tab, subtab)
            preview = previews[stage]
            if preview is None or not preview["has_draft"] or not preview["items"]:
                _skip_stage(receipt, stage)
        selection = select_writeback_items(
            RecipeWritebackPreview(previews["primary"], previews["post"]), requested
        )
        selected = {
            "primary": selection.preview.primary,
            "post": selection.preview.post,
        }
        for stage, subtab, _ in _PANES:
            write_attempted = False
            preview = selected[stage]
            if preview is None or not preview["items"]:
                _skip_stage(receipt, stage)
                continue
            written: list[WritebackItemReceipt] = ctx.gui.send_gui_rpc(
                "tab.writeback_write",
                {
                    "tab_id": tab,
                    "subtab_id": subtab,
                    "write": [{"id": item["id"]} for item in preview["items"]],
                },
                before_send=admit_write,
            )["written"]
            receipt["completed"].append({"stage": stage, "written": written})
            receipt["not_started"].remove(stage)
    except GuiRpcError as error:
        receipt["not_started"].remove(stage)
        receipt["status"] = "failed"
        receipt["failed_stage"] = stage
        receipt["error"] = {
            "code": error.code,
            "reason": error.reason,
            "message": str(error),
        }
        receipt["failed_stage_may_have_partial_writes"] = write_attempted
    return receipt
