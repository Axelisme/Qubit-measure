"""Writeback remote handlers — pane-qualified with destination projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import (
    AnalysisMode,
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._wire_values import _coerce_wire_value, _json_safe

_VALID_WRITEBACK_SUBTABS = frozenset({"analysis", "post_analysis"})


def _destination_context(adapter: RemoteControlAdapter) -> dict[str, object]:
    """Current active ExpContext projection (destination at reply time).

    Does not compare or store source identity; draft source is opaque.
    """
    try:
        ctx = adapter.ctrl.get_exp_context()
    except Exception:
        return {"active_label": None}
    return {
        "active_label": ctx.active_label,
        "chip_name": ctx.chip_name,
        "qub_name": ctx.qub_name,
        "res_name": ctx.res_name,
        "database_path": ctx.database_path,
        "result_dir": ctx.result_dir,
        "has_active_context": bool(ctx.is_active()),
    }


def _writeback_item_wire(item) -> dict[str, object]:
    base: dict[str, object] = {
        "id": item.session_id,
        "target_name": item.target_name,
        "description": item.description,
        "selected": bool(item.selected),
    }
    if isinstance(item, MetaDictWriteback):
        base["kind"] = "metadict"
        base["proposed_value"] = _json_safe(item.proposed_value)
    elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
        is_module = isinstance(item, ModuleWriteback)
        base["kind"] = "module" if is_module else "waveform"
        # New opaque drafts never expose their cfg-editor identity. Keep the
        # legacy field only when an old caller explicitly supplied one, so the
        # temporary A4 adapter remains readable during migration.
        legacy_editor_id = getattr(item, "editor_id", None)
        if legacy_editor_id is not None:
            base["editor_id"] = legacy_editor_id
        base["has_edit_schema"] = item.edit_schema is not None
        if item.role_id is not None:
            base["role_id"] = item.role_id
    else:
        base["kind"] = "unknown"
    return base


def _h_tab_writeback_preview(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Pure read of a pane's persistent writeback draft (not a dry-run).

    Requires (tab_id, subtab_id) with closed values analysis|post_analysis.
    Projects current destination context at reply time; draft stores no source.
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
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    if snap.capabilities is None:
        raise RemoteError(ErrorCode.INTERNAL, "snapshot has no capabilities")
    if subtab_id == "analysis":
        if snap.capabilities.analysis is AnalysisMode.NONE:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                f"tab {tab_id!r} does not support analysis",
            )
        items = list(snap.analysis.writeback_items) if snap.analysis is not None else []
    else:
        if not snap.capabilities.post_analysis:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                f"tab {tab_id!r} does not support post_analysis",
            )
        items = (
            list(snap.post_analysis.writeback_items)
            if snap.post_analysis is not None
            else []
        )
    return {
        "has_draft": bool(items),
        "items": [_writeback_item_wire(it) for it in items],
        "destination_context": _destination_context(adapter),
    }


def _h_tab_writeback_set(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Edit a pane's persistent writeback item by id — the single writeback editing
    surface (ADR-0008). Requires (tab_id, subtab_id)."""
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
    # ``proposed_value`` is only forwarded when present *and* non-null; a metadict
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
            "'proposed_value' (metadict) and 'edits' (module/waveform) are "
            "mutually exclusive",
        )
    if has_proposed:
        # Structured tags (e.g. {"__complex__": [re, im]}) coerce back to their
        # Python type so the applied md value matches what preview serialized.
        changes["proposed_value"] = _coerce_wire_value(params["proposed_value"])
    if has_edits:
        raw_edits = params["edits"]
        if not isinstance(raw_edits, list):
            raise RemoteError(ErrorCode.INVALID_PARAMS, "'edits' must be a list")
        edits: list[dict[str, object]] = []
        for i, edit in enumerate(raw_edits):
            if not isinstance(edit, dict) or "path" not in edit or "value" not in edit:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"edits[{i}] must be an object with 'path' and 'value'",
                )
            edits.append({"path": str(edit["path"]), "value": edit["value"]})
        changes["edits"] = edits
    pane = subtab_id  # WritebackPane literal matches wire values
    agg = adapter.writeback_control.set_writeback_item_for_pane(
        tab_id,
        pane,
        session_id,
        **changes,  # type: ignore[arg-type]
    )
    # Echo the edited item so the agent sees the post-edit state in one round-trip.
    item = _find_writeback_item_for_pane(adapter, tab_id, pane, session_id)
    reply: dict[str, object] = {"item": _writeback_item_wire(item)}
    if has_edits:
        reply.update(agg)
    return reply


def _find_writeback_item(adapter: RemoteControlAdapter, tab_id: str, session_id: str):
    # Legacy helper retained for internal tests that call it directly; delegate to pane-aware
    # analysis pane by default.
    return _find_writeback_item_for_pane(adapter, tab_id, "analysis", session_id)


def _find_writeback_item_for_pane(
    adapter: RemoteControlAdapter, tab_id: str, pane: str, session_id: str
):
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    items: list = []
    if pane == "analysis" and snap.analysis is not None:
        items = list(snap.analysis.writeback_items)
    elif pane == "post_analysis" and snap.post_analysis is not None:
        items = list(snap.post_analysis.writeback_items)
    for item in items:
        if item.session_id == session_id:
            return item
    # Fallback to writeback_control direct query for completeness
    try:
        fallback = (
            adapter.writeback_control.get_tab_writeback_items(tab_id)
            if pane == "analysis"
            else []
        )
    except Exception:
        fallback = []
    for item in fallback:
        if item.session_id == session_id:
            return item
    raise RemoteError(
        ErrorCode.INVALID_PARAMS, f"unknown writeback item id: {session_id!r}"
    )


def _h_tab_writeback_apply(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply a pane's persistent writeback draft as-is (edit it first via
    gui_tab_writeback_set_item). Projects destination context at reply time."""
    tab_id = str(params["tab_id"])
    subtab_id = str(params["subtab_id"])
    if subtab_id not in _VALID_WRITEBACK_SUBTABS:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"invalid subtab_id {subtab_id!r}; expected one of {sorted(_VALID_WRITEBACK_SUBTABS)}",
        )
    if not adapter.writeback_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    pane = subtab_id  # type: ignore[assignment]
    result = adapter.writeback_control.apply_writeback_for_pane(tab_id, pane)  # type: ignore[arg-type]
    context_version = adapter.writeback_control.get_context_version()
    return {
        "applied_ids": list(result["applied_ids"]),
        "written": result["written"],
        "context_version": context_version,
        "destination_context": _destination_context(adapter),
    }
