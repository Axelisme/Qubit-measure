"""Writeback remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._wire_values import _coerce_wire_value, _json_safe

logger = logging.getLogger(__name__)


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
        base["editor_id"] = item.editor_id
        base["has_edit_schema"] = item.editor_id is not None
    else:
        base["kind"] = "unknown"
    return base


def _h_tab_writeback_preview(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Pure read of the tab's persistent writeback draft (not a dry-run): lists
    the items computed once at analyze time. ``has_draft`` is false before any
    analyze has produced a draft (empty item list)."""
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    items = adapter.ctrl.get_tab_writeback_items(tab_id)
    return {
        "has_draft": bool(items),
        "items": [_writeback_item_wire(it) for it in items],
    }


def _h_tab_writeback_set(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Edit a persistent writeback item by id — the single writeback editing
    surface (ADR-0008). ``selected`` / ``target_name`` apply to any item;
    ``proposed_value`` is the metadict-only facet; ``edits`` is the
    module/waveform-only facet (cfg edits applied through the item's editor
    session internally — the agent never handles its editor_id). ``proposed_value``
    and ``edits`` are mutually exclusive (they target different item kinds);
    None disambiguates which facet is supplied. Echoes the edited ``item``; an
    ``edits`` batch also folds the aggregated ``{valid, removed, added}``."""
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
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
    try:
        agg = adapter.ctrl.set_writeback_item(tab_id, session_id, **changes)
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    # Echo the edited item so the agent sees the post-edit state in one round-trip.
    item = _find_writeback_item(adapter, tab_id, session_id)
    reply: dict[str, object] = {"item": _writeback_item_wire(item)}
    if has_edits:
        reply.update(agg)
    return reply


def _find_writeback_item(adapter: RemoteControlAdapter, tab_id: str, session_id: str):
    for item in adapter.ctrl.get_tab_writeback_items(tab_id):
        if item.session_id == session_id:
            return item
    raise RemoteError(
        ErrorCode.INVALID_PARAMS, f"unknown writeback item id: {session_id!r}"
    )


def _h_tab_writeback_apply(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply the tab's persistent writeback draft as-is (edit it first via
    gui_tab_writeback_set_item). Echoes what was written: applied_ids, the
    destination names actually pushed (``written`` by kind), and the post-apply
    ``context`` resource version (apply bumps it once, ADR-0006)."""
    tab_id = str(params["tab_id"])
    if not adapter.ctrl.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    try:
        result = adapter.ctrl.apply_writeback(tab_id)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Read the context version AFTER apply so the agent sees the bumped value it
    # can pass back as an expected_versions guard on a follow-up write.
    context_version = adapter.ctrl.resources_versions().get("context", 0)
    return {
        "applied_ids": list(result["applied_ids"]),
        "written": result["written"],
        "context_version": context_version,
    }
