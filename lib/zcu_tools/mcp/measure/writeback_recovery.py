"""Submission reservation and recovery shared by standalone and recipe writes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.tool_context import MeasureToolContext

CHECK_WRITEBACK_STATUS = "Check status before deciding whether to resubmit; timeout does not prove write failure."


class WritebackQueryParams(TypedDict):
    """Owner-generated, GUI-lifetime submission identity to query."""

    submission_id: str


class WritebackRecovery(TypedDict):
    """Readonly receipt RPC and its exact parameters; never a retry command."""

    rpc: Literal["tab.writeback_receipt"]
    params: WritebackQueryParams


class _Begun(BaseModel):
    submission_id: str = Field(min_length=1)
    status: Literal["pending"]


def begin_writeback(ctx: MeasureToolContext, tab: str) -> str:
    """Ask the bound GUI owner for a once-only ID without writing or admission.

    Invalid native replies raise validation errors; RPC errors propagate. The
    returned ID is not a cancellable operation handle and cannot survive GUI exit.
    """
    return _Begun.model_validate(
        ctx.send_gui_rpc("tab.writeback_begin", {"tab_id": tab})
    ).submission_id


def writeback_recovery(submission_id: str) -> WritebackRecovery:
    """Build the readonly query for an ID already returned by the GUI owner."""
    return {
        "rpc": "tab.writeback_receipt",
        "params": {"submission_id": submission_id},
    }
