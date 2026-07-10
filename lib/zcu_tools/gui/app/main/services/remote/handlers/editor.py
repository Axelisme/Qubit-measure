"""Editor remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.services.context import MlEntryValidationError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


logger = logging.getLogger(__name__)


def _h_editor_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    from ..path_resolver import build_settable_tree

    item_kind = str(params["item_kind"])
    from_name = str(params["from_name"])
    # editor.new is modify-only: it edits an existing ml entry. Creating a blank
    # entry goes through context.ml_create_from_role (role_id='<disc>:blank').
    try:
        editor_id, _ = adapter.ctrl.open_cfg_editor(
            item_kind, discriminator=None, from_name=from_name
        )
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    # The agent reads every cfg view as a nested tree (same shape as
    # tab.get_cfg / editor.get), so the open reply carries the freshly-opened
    # draft as {tree} rather than the flat current_paths the session also tracks
    # internally for change-push / set_field diffing.
    draft = adapter.ctrl.get_cfg_editor_draft(editor_id)
    return {"editor_id": editor_id, "tree": build_settable_tree(draft)}


def _h_editor_set_field(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

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
    try:
        return adapter.ctrl.cfg_editor_set_field(editor_id, path, value)
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


def _h_editor_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    from ..path_resolver import build_settable_tree

    editor_id = str(params["editor_id"])
    raw_prefix = params.get("prefix")
    prefix = str(raw_prefix) if raw_prefix else None
    # Build the nested current-value tree off the session's live root — the same
    # tree shape tab.get_cfg returns, so the agent reads every cfg view as a tree
    # and edits leaves via editor.set_field (dotted paths). An unknown
    # editor_id raises CfgEditorError from get_cfg_editor_draft → INVALID_PARAMS.
    try:
        draft = adapter.ctrl.get_cfg_editor_draft(editor_id)
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"tree": build_settable_tree(draft, prefix=prefix)}


def _h_editor_commit(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    name = str(params["name"])
    try:
        adapter.ctrl.commit_cfg_editor(editor_id, name)
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except MlEntryValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_editor_discard(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError

    editor_id = str(params["editor_id"])
    try:
        adapter.ctrl.discard_cfg_editor(editor_id)
    except CfgEditorError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {}
