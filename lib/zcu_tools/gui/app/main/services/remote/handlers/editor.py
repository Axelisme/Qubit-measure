"""Editor remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.services.ports import CfgEdit
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

# The method registry resolves handlers by name at runtime.
__all__ = ["_h_editor_set_fields"]


def _h_editor_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..cfg_observation import build_cfg_observation

    item_kind = str(params["item_kind"])
    from_name = str(params["from_name"])
    # editor.new is modify-only: it edits an existing ml entry. Creating a blank
    # entry goes through context.ml_create_from_role (role_id='<disc>:blank').
    editor_id, _ = adapter.ctrl.open_cfg_editor(item_kind, from_name=from_name)
    # Creation and explicit reads share the complete observation format.
    draft = adapter.ctrl.get_cfg_editor_draft(editor_id)
    return {"editor_id": editor_id, "tree": build_cfg_observation(draft)}


def _h_editor_set_field(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    editor_id = str(params["editor_id"])
    path = str(params["path"])
    value = params["value"]
    # A tab cfg draft is a session owned by the tab_id; editing it while that
    # tab runs is blocked — same guard the human gets via the disabled form
    # (ADR-0013 F11). owner-less / ml-entry sessions are unaffected.
    owner = adapter.ctrl.owner_of_editor(editor_id)
    if owner is not None and adapter.ctrl.get_running_tab_id() == owner:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {owner!r} is currently running; cancel the run before editing cfg",
        )
    return adapter.ctrl.cfg_editor_set_field(editor_id, path, value).to_wire()


def _h_editor_set_fields(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    editor_id = str(params["editor_id"])
    owner = adapter.ctrl.owner_of_editor(editor_id)
    if owner is not None and adapter.ctrl.get_running_tab_id() == owner:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"tab {owner!r} is currently running; cancel the run before editing cfg",
        )
    raw_edits = params["edits"]
    if not isinstance(raw_edits, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'edits' must be a list")
    edits: list[CfgEdit] = []
    for i, edit in enumerate(raw_edits):
        if not isinstance(edit, dict) or "path" not in edit or "value" not in edit:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"edits[{i}] must have 'path' and 'value'",
            )
        edits.append(CfgEdit(str(edit["path"]), edit["value"]))
    return adapter.ctrl.cfg_editor_set_fields(
        editor_id, edits, agent_edit=True
    ).to_wire()


def _h_editor_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..cfg_observation import build_cfg_observation

    editor_id = str(params["editor_id"])
    raw_prefix = params.get("prefix")
    prefix = str(raw_prefix) if raw_prefix else None
    # Unknown sessions retain CfgEditorError -> INVALID_PARAMS translation.
    draft = adapter.ctrl.get_cfg_editor_draft(editor_id)
    return {"tree": build_cfg_observation(draft, prefix=prefix)}


def _h_editor_commit(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    editor_id = str(params["editor_id"])
    name = str(params["name"])
    adapter.ctrl.commit_cfg_editor(editor_id, name)
    return {}


def _h_editor_discard(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    editor_id = str(params["editor_id"])
    adapter.ctrl.discard_cfg_editor(editor_id)
    return {}
