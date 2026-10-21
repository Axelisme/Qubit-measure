"""Writeback remote handlers — pane-qualified with destination projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Final, cast

from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    SampleWriteback,
    WritebackEdit,
    WritebackItem,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ..form_observation import entry_invalid_to_remote, form_writes, observe_entry
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
        "unavailable_reasons": list(pane.unavailable_reasons),
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


def h_tab_writeback_set(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Edit a pane's persistent writeback item by id — the single writeback editing
    surface (ADR-0008). Requires (tab_id, subtab_id).

    Module edits are one Form write batch of {path: [field, ...], value}; the
    reply adds the item's Form observation (input, values, errors)."""
    tab_id = str(params["tab_id"])
    subtab_id = str(params["subtab_id"])
    if subtab_id not in _VALID_WRITEBACK_SUBTABS:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"invalid subtab_id {subtab_id!r}; expected one of {sorted(_VALID_WRITEBACK_SUBTABS)}",
        )
    if not adapter.writeback_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    session_id = str(params["id"])
    # The wire collapses "omitted optional" and "explicit JSON null" to the same
    # thing (a null-valued key), so a null here means "not provided" — never a
    # value to write. ``selected``/``target_name`` can never legitimately be null.
    # ``proposed_value`` is only forwarded when present *and* non-null; a parameter
    # item that genuinely needs a null value is out of scope for this surface.
    changes: dict[str, object] = {}
    if params.get("selected") is not None:
        changes["selected"] = bool(params["selected"])
    if params.get("target_name") is not None:
        name = params["target_name"]
        if not isinstance(name, str) or not name:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "target_name must be a non-empty string"
            )
        changes["target_name"] = name
    has_proposed = params.get("proposed_value") is not None
    has_edits = params.get("edits") is not None
    if has_proposed and has_edits:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "'proposed_value' (parameter) and 'edits' (module) are mutually exclusive",
        )
    if has_proposed:
        # Structured tags (e.g. {"__complex__": [re, im]}) coerce back to their
        # Python type so the applied md value matches what preview serialized.
        changes["proposed_value"] = coerce_wire_value(params["proposed_value"])
    if has_edits:
        changes["edits"] = form_writes(params["edits"])
    pane = "analysis" if subtab_id == "analysis" else "post_analysis"
    adapter.writeback_control.set_writeback_item_for_pane(
        tab_id, pane, session_id, **changes
    )
    # Echo the edited item so the agent sees the post-edit state in one round-trip.
    item = _find_writeback_item_for_pane(adapter, tab_id, pane, session_id)
    reply: dict[str, object] = {"item": _writeback_item_wire(item)}
    if has_edits:
        form = adapter.writeback_control.get_writeback_item_form_for_pane(
            tab_id, pane, session_id
        )
        reply.update(observe_entry(form))
    return reply


def _find_writeback_item_for_pane(
    adapter: RemoteControlAdapter, tab_id: str, pane: str, session_id: str
) -> WritebackItem:
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    items: list[WritebackItem] = []
    if pane == "analysis" and snap.analysis is not None:
        items = list(snap.analysis.writeback_items)
    elif pane == "post_analysis" and snap.post_analysis is not None:
        items = list(snap.post_analysis.writeback_items)
    for item in items:
        if item.session_id == session_id:
            return item
    raise RemoteError(
        ErrorCode.INVALID_PARAMS, f"unknown writeback item id: {session_id!r}"
    )


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
            tab_id, pane, changes
        )
    except LibraryEntryInvalid as exc:
        raise entry_invalid_to_remote(exc) from exc
    return {"written": [_writeback_wire_value(asdict(item)) for item in result]}


def h_tab_writeback_apply(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply a pane's persistent draft after rpc_call on tab.writeback_set.
    Projects destination context at reply time."""
    tab_id = str(params["tab_id"])
    subtab_id = str(params["subtab_id"])
    if subtab_id not in _VALID_WRITEBACK_SUBTABS:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"invalid subtab_id {subtab_id!r}; expected one of {sorted(_VALID_WRITEBACK_SUBTABS)}",
        )
    if not adapter.writeback_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    pane = "analysis" if subtab_id == "analysis" else "post_analysis"
    item_ids = None
    ids = params.get("ids")
    if ids is not None:
        if not isinstance(ids, list) or any(
            not isinstance(item, str) or not item for item in ids
        ):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "ids must be a list of nonempty strings"
            )
        item_ids = tuple(ids)
    try:
        result = adapter.writeback_control.apply_writeback_for_pane(
            tab_id,
            pane,
            item_ids=item_ids,
        )
    except LibraryEntryInvalid as exc:
        raise entry_invalid_to_remote(exc) from exc
    context_version = adapter.writeback_control.get_context_version()
    return {
        "applied_ids": list(result["applied_ids"]),
        "written": result["written"],
        "context_version": context_version,
        "destination_context": _destination_context(adapter),
    }
