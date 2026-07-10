"""Tab remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._common import render_view

logger = logging.getLogger(__name__)


def _h_tab_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    tab_id = adapter.tab_control.new_tab(name)
    return {"tab_id": tab_id}


def _h_tab_close(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    adapter.tab_control.close_tab(tab_id)
    return {"ok": True}


def _h_tab_set_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    adapter.tab_control.set_active_tab(tab_id)
    return {"ok": True}


def _h_tab_list_all(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    running_tab_id = adapter.tab_control.get_running_tab_id()
    tabs = [
        {
            "tab_id": tid,
            "adapter_name": adapter.tab_control.get_tab_adapter_name(tid),
            "is_running": tid == running_tab_id,
        }
        for tid in adapter.tab_control.list_tab_ids()
    ]
    # active_tab_id is a view projection (which tab the user is focused on),
    # sourced from the same RenderView snapshot _assemble_overview reads.
    active_tab_id = render_view(adapter).get_view_snapshot().get("active_tab_id")
    return {
        "tabs": tabs,
        "active_tab_id": active_tab_id,
        "running_tab_id": running_tab_id,
    }


def _tab_snapshot_wire(adapter: RemoteControlAdapter, tab_id: str) -> dict[str, object]:
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    interaction = snap.interaction
    # Render snapshot always fills the live fields (persist/restore form is the
    # only one that leaves them None, and it never hits the wire).
    assert interaction is not None
    return {
        "tab_id": tab_id,
        "adapter_name": adapter.tab_control.get_tab_adapter_name(tab_id),
        # Shared cfg-editor session id for this tab (None until the tab's form
        # is populated). Address it with the editor.* methods to edit cfg with
        # the GUI reflecting every change. (A tab uses its tab_id as owner key.)
        "editor_id": adapter.ctrl.editor_id_for_owner(tab_id),
        "interaction": {
            "global_run_active": bool(interaction.global_run_active),
            "is_running": bool(interaction.is_running),
            "is_analyzing": bool(interaction.is_analyzing),
            "is_saving_data": bool(interaction.is_saving_data),
            "has_context": bool(interaction.has_context),
            "has_active_context": bool(interaction.has_active_context),
            "has_soc": bool(interaction.has_soc),
            "has_run_result": bool(interaction.has_run_result),
            "has_analyze_result": bool(interaction.has_analyze_result),
            "has_figure": bool(interaction.has_figure),
        },
        "save_paths": _save_paths_wire(snap.save_paths),
        "result_source_path": snap.result_source_path,
    }


def _h_tab_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Always returns {tabs: [...]} (a single tab_id yields a one-element list);
    # no shape-switch, so callers index reply["tabs"] uniformly.
    tab_id_raw = params.get("tab_id")
    if tab_id_raw is None:
        tab_ids = adapter.tab_control.list_tab_ids()
    else:
        tab_id = str(tab_id_raw)
        if not adapter.tab_control.has_tab(tab_id):
            raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
        tab_ids = [tab_id]
    return {"tabs": [_tab_snapshot_wire(adapter, tid) for tid in tab_ids]}


def _save_paths_wire(paths) -> dict[str, str] | None:
    if paths is None:
        return None
    return {"data_path": paths.data_path, "image_path": paths.image_path}


def _h_tab_get_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..path_resolver import build_settable_tree

    tab_id = str(params["tab_id"])
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # A tab's cfg draft is a CfgEditorService session keyed by its tab_id (the
    # same draft the open form attaches to). Build the settable tree off that
    # session's live root — the one tab.set_cfg/editor.set_field mutates — so
    # the tree mirrors exactly what can be edited and agent+user share one model
    # (ADR-0013 F11). Leaf values come straight off the live tree
    # (ADR-0010: None = unset).
    editor_id = adapter.ctrl.editor_id_for_owner(tab_id)
    if editor_id is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} cfg form has no live model yet",
        )
    raw_prefix = params.get("prefix")
    prefix = str(raw_prefix) if raw_prefix else None
    draft = adapter.ctrl.get_cfg_editor_draft(editor_id)
    return {"tree": build_settable_tree(draft, prefix=prefix)}


def _h_tab_set_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    tab_id = str(params["tab_id"])
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # Block edits while the tab is running — same guard the human gets via the
    # disabled form (ADR-0013 F11).
    if adapter.tab_control.get_running_tab_id() == tab_id:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} is currently running; cancel the run before editing cfg",
        )
    editor_id = adapter.ctrl.editor_id_for_owner(tab_id)
    if editor_id is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {tab_id!r} cfg form has no live model yet",
        )
    raw_edits = params.get("edits")
    if not isinstance(raw_edits, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'edits' must be a list")
    # Apply edits sequentially (fail-fast, non-atomic); caller orders ref-switch
    # edits before dependent inner-path edits. Delegate to cfg_editor_set_field
    # — the same path the editor.set_field handler uses — to avoid duplicating
    # path resolution and validation logic.
    all_removed: list[str] = []
    all_added: list[str] = []
    valid = True
    for i, edit in enumerate(raw_edits):
        if not isinstance(edit, dict) or "path" not in edit or "value" not in edit:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"edits[{i}] must be an object with 'path' and 'value'",
            )
        path = str(edit["path"])
        value = edit["value"]
        try:
            result = adapter.ctrl.cfg_editor_set_field(editor_id, path, value)
        except CfgEditorError as exc:
            raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
        except RemoteError:
            raise
        except (KeyError, RuntimeError) as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                str(exc),
                reason=getattr(exc, "reason_code", ""),
            ) from exc
        valid = bool(result.get("valid", True))
        removed = result.get("removed", [])
        added = result.get("added", [])
        if isinstance(removed, list):
            all_removed.extend(str(p) for p in removed)
        if isinstance(added, list):
            all_added.extend(str(p) for p in added)
    return {"valid": valid, "removed": all_removed, "added": all_added}
