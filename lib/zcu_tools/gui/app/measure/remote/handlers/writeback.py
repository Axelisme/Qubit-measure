"""Writeback remote handlers — pane-qualified with destination projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Final, cast

from pydantic import TypeAdapter, ValidationError

from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    SampleWriteback,
    WritebackEdit,
    WritebackItem,
)
from zcu_tools.gui.app.measure.writeback.control import CapturedWritebackPane
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ..form_observation import entry_invalid_to_remote, form_writes
from ._common import follow_tab
from ._wire_values import coerce_wire_value, context_wire_value

_VALID_WRITEBACK_SUBTABS: Final = ("analysis", "post_analysis")


def _destination_context(adapter: RemoteControlAdapter) -> dict[str, object]:
    """Current active SessionEnv projection (destination at reply time).

    Does not compare or store source identity; draft source is opaque.
    """
    ctx = adapter.ctrl.get_session_env()
    return {
        "active_label": ctx.active_label,
        "database_path": ctx.database_path,
        "result_dir": ctx.result_dir,
        "has_active_context": bool(ctx.is_active()),
    }


def _writeback_wire_value(value: object) -> object:
    try:
        return context_wire_value(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"cannot fully project writeback value: {exc}",
            reason="unserializable_context",
        ) from exc


def _writeback_item_wire(item: WritebackItem) -> dict[str, object]:
    base: dict[str, object] = {
        "id": item.session_id,
        "target_name": item.target_name,
        "description": item.description,
        "selected": bool(item.selected),
    }
    if item.element is not None:
        base["element"] = item.element.copy()
    if isinstance(item, ParameterWriteback):
        base["kind"] = "parameter"
        base["proposed_value"] = _writeback_wire_value(item.proposed_value)
    elif isinstance(item, SampleWriteback):
        base["kind"] = "sample"
        base["proposed_value"] = _writeback_wire_value(item.proposed_value)
    elif isinstance(item, ModuleWriteback):
        base["kind"] = "module"
        if item.role_id is not None:
            base["role_id"] = item.role_id
    else:
        base["kind"] = "unknown"
    return base


def h_tab_writeback_preview(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Pure read of a pane's persistent writeback draft (not a dry-run).

    Requires (tab_id, subtab_id) with closed values analysis|post_analysis.
    Each item includes the service-owned captured actual_path and existing
    stderr (working units, or None), and the adapter's note (plain text or None).
    selected reflects current manual selection, initially the proposal's bool.
    destination_context is the active context
    at reply time, not the draft's captured destination.
    """
    tab_id = str(params["tab_id"])
    subtab_id = str(params["subtab_id"])
    if subtab_id not in _VALID_WRITEBACK_SUBTABS:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"invalid subtab_id {subtab_id!r}; expected one of {sorted(_VALID_WRITEBACK_SUBTABS)}",
        )
    if not adapter.writeback_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # tolerated(cast): ParamSpec validates optional INTEGER; Mapping retains object values.
    operation_id = cast(int | None, params.get("operation_id"))
    if operation_id is not None:
        adapter.run_analyze_control.require_analysis_operation(
            tab_id, subtab_id, operation_id
        )
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    if snap.capabilities is None:
        raise RemoteError(ErrorCode.INTERNAL, "snapshot has no capabilities")
    if subtab_id == "analysis":
        if snap.capabilities.analysis is AnalysisMode.NONE:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                f"tab {tab_id!r} does not support analysis",
            )
        pane = snap.analysis
    else:
        if not snap.capabilities.post_analysis:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                f"tab {tab_id!r} does not support post_analysis",
            )
        pane = snap.post_analysis
    if pane is None:
        raise RemoteError(ErrorCode.INTERNAL, "snapshot has no requested pane")
    try:
        values = (
            adapter.writeback_control.get_writeback_values_for_pane(
                tab_id, "analysis" if subtab_id == "analysis" else "post_analysis"
            )
            if pane.has_writeback_draft
            else {}
        )
    except LibraryEntryInvalid as exc:
        raise entry_invalid_to_remote(exc) from exc
    return {
        "has_draft": pane.has_writeback_draft,
        "items": [
            {
                **_writeback_item_wire(item),
                "actual_path": values[item.session_id].actual_path,
                "stderr": item.stderr,
                "note": item.note,
                "current": _writeback_wire_value(values[item.session_id].current),
                "proposed": _writeback_wire_value(values[item.session_id].proposed),
            }
            for item in pane.writeback_items
        ],
        "destination_context": _destination_context(adapter),
    }


def _batch_change(raw: object) -> WritebackEdit:
    if not isinstance(raw, dict) or set(raw) - {"id", "target", "value", "edits"}:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "invalid writeback item fields")
    session_id = raw.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "id must be a nonempty string")
    target = raw.get("target")
    if "target" in raw and (not isinstance(target, str) or not target):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "target must be a nonempty string")
    if "value" in raw and "edits" in raw:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "value and edits are exclusive")
    change = WritebackEdit(
        session_id,
        target_name=target,
        edits=form_writes(raw["edits"]) if "edits" in raw else None,
    )
    if "value" in raw:
        change = replace(change, proposed_value=coerce_wire_value(raw["value"]))
    return change


def h_tab_writeback_write(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    subtab = str(params["subtab_id"])
    if subtab not in _VALID_WRITEBACK_SUBTABS:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "invalid writeback subtab_id")
    if not adapter.writeback_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    raw = params["write"]
    if not isinstance(raw, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "write must be a list")
    changes = tuple(_batch_change(item) for item in raw)
    pane = "analysis" if subtab == "analysis" else "post_analysis"
    follow_tab(adapter, tab_id, pane)
    try:
        result = adapter.writeback_control.write_writeback_for_pane(
            tab_id,
            pane,
            changes,
            **(
                {"submission_id": _submission_id(params)}
                if params.get("submission_id") is not None
                else {}
            ),
        )
    except LibraryEntryInvalid as exc:
        raise entry_invalid_to_remote(exc) from exc
    return {"written": [_writeback_wire_value(asdict(item)) for item in result]}


def h_tab_writeback_accept(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Submit all captured panes in one owner dispatch, without editing them.

    stages contains subtab_id and ids for each selected pane. All IDs are checked
    before the first write. Values and destinations come from current owner drafts.
    Return stage-qualified confirmed IDs and the first write diagnostic.
    With a reserved submission_id, actual values remain queryable through
    tab.writeback_receipt even if this handler's reply times out.
    """
    tab_id = str(params["tab_id"])
    try:
        stages = TypeAdapter(tuple[CapturedWritebackPane, ...]).validate_python(
            params["stages"]
        )
    except ValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    if any(not row.ids or any(not identity for identity in row.ids) for row in stages):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "captured stages require nonempty item IDs"
        )
    try:
        result = adapter.writeback_control.accept_writeback(
            tab_id,
            stages,
            **(
                {"submission_id": _submission_id(params)}
                if params.get("submission_id") is not None
                else {}
            ),
        )
    except LibraryEntryInvalid as exc:
        raise entry_invalid_to_remote(exc) from exc
    return {
        "status": "failed" if result.error is not None else "completed",
        "written": [
            {"subtab_id": stage.subtab_id, "ids": list(ids)}
            for stage, ids in zip(stages, result.applied_ids, strict=True)
        ],
        "error": result.error,
    }


def _submission_id(params: Mapping[str, object]) -> str:
    identity = params["submission_id"]
    if not isinstance(identity, str) or not identity:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "submission_id must be nonempty")
    return identity


def h_tab_writeback_begin(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Reserve a once-only ID without validating or writing the draft."""
    return asdict(adapter.writeback_control.begin_writeback(str(params["tab_id"])))


def h_tab_writeback_receipt(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Read retained confirmed values, not current draft or destination values."""
    receipt = adapter.writeback_control.get_writeback_receipt(_submission_id(params))
    return {
        "submission_id": receipt.submission_id,
        "tab_id": receipt.tab_id,
        "status": receipt.status,
        "completed": [
            {
                "subtab_id": row.subtab_id,
                "item": _writeback_wire_value(asdict(row.item)),
            }
            for row in receipt.completed
        ],
        "error": receipt.error,
    }
