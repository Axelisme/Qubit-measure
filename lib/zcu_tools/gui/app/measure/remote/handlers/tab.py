"""Tab remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.cfg.edit_codec import decode_edits, decode_ref, encode_ref
from zcu_tools.gui.cfg.resource import (
    CfgInputError,
    CfgPreconditionError,
    CfgStaleError,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._common import follow_tab, render_view


def h_tab_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    tab_id = adapter.tab_control.new_tab(name)
    return {"tab_id": tab_id}


def h_tab_open_file(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from dataclasses import asdict

    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    return asdict(
        adapter.tab_control.open_tab_from_file(name, str(params["data_path"]))
    )


def h_tab_close(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    from .lifecycle import require_idle, require_saved

    require_idle(adapter, tab_id)
    if not params["discard_unsaved"]:
        require_saved(adapter, [tab_id])
    adapter.tab_control.close_tab(tab_id)
    return {"ok": True}


def h_tab_set_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    adapter.tab_control.set_active_tab(tab_id)
    return {"ok": True}


def h_tab_list_all(
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
    # sourced from the RenderView snapshot, separate from the status tool.
    active_tab_id = render_view(adapter).get_view_snapshot().get("active_tab_id")
    return {
        "tabs": tabs,
        "active_tab_id": active_tab_id,
        "running_tab_id": running_tab_id,
    }


def tab_operation_state(
    adapter: RemoteControlAdapter, tab_id: str
) -> dict[str, object]:
    snap = adapter.tab_control.get_tab_snapshot(tab_id)
    interaction = snap.interaction
    # Render snapshot always fills the live fields (persist/restore form is the
    # only one that leaves them None, and it never hits the wire).
    assert interaction is not None
    assert snap.run is not None
    assert snap.analysis is not None
    assert snap.post_analysis is not None
    versions = adapter.ctrl.resources_versions()
    return {
        "tab_id": tab_id,
        "adapter_name": adapter.tab_control.get_tab_adapter_name(tab_id),
        "cfg_ref": encode_ref(adapter.cfg_lookup(tab_id).observe().ref),
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
            "has_post_analyze_result": bool(interaction.has_post_analyze_result),
            "has_figure": bool(interaction.has_figure),
        },
        "save_paths": _save_paths_wire(snap.paths),
        "artifacts": [
            {
                "kind": artifact.kind.value,
                "status": artifact.status.value,
                "default_path": artifact.default_path,
                "last_saved_path": artifact.last_saved_path,
                "is_saveable": artifact.is_saveable,
            }
            for artifact in snap.artifacts
        ],
        "result_source_path": snap.run.source_path,
        # Revisions distinguish replacements even when availability and source
        # path stay unchanged. Payload arrays remain with the application owner.
        "result_state": {
            "revision": versions.get(f"tab:{tab_id}:result", 0),
            "available": snap.run.result is not None,
            "source_path": snap.run.source_path,
        },
        "analysis_state": {
            "revision": versions.get(f"tab:{tab_id}:analyze", 0),
            "available": snap.analysis.result is not None,
            "has_figure": snap.analysis.figure is not None,
            "has_writeback_draft": snap.analysis.has_writeback_draft,
        },
        "post_analysis_state": {
            "revision": versions.get(f"tab:{tab_id}:post_analyze", 0),
            "available": snap.post_analysis.result is not None,
            "has_figure": snap.post_analysis.figure is not None,
            "has_writeback_draft": snap.post_analysis.has_writeback_draft,
        },
    }


def h_tab_snapshot(
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
    return {"tabs": [tab_operation_state(adapter, tid) for tid in tab_ids]}


def _save_paths_wire(paths) -> dict[str, str | None] | None:
    if paths is None:
        return None
    # Three independent effective path resources (no single generic image_path).
    return {
        "data_path": paths.data.path,
        "analysis_image_path": paths.analysis_image.path,
        "post_analysis_image_path": paths.post_analysis_image.path,
    }


def h_tab_get_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..cfg_observation import build_resource_observation

    return build_resource_observation(
        adapter.cfg_lookup(str(params["tab_id"])).observe()
    )


def h_tab_reset_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..cfg_observation import build_resource_observation, cfg_error_to_remote

    tab_id = str(params["tab_id"])
    try:
        expected = decode_ref(params["expected"])
        editor = adapter.cfg_lookup(tab_id)
        actual = editor.observe().ref
        if expected != actual:
            raise CfgStaleError(expected, actual)
        result = editor.reset(expected.revision)
    except (CfgInputError, CfgPreconditionError) as exc:
        raise cfg_error_to_remote(exc) from exc
    follow_tab(adapter, tab_id, "run")
    return build_resource_observation(result)


def h_tab_edit_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..cfg_observation import build_resource_observation, cfg_error_to_remote

    tab_id = str(params["tab_id"])
    try:
        expected = decode_ref(params["expected"])
        edits = decode_edits(params["edits"])
        editor = adapter.cfg_lookup(tab_id)
        actual = editor.observe().ref
        if expected != actual:
            raise CfgStaleError(expected, actual)
        result = editor.edit(expected.revision, edits)
    except (CfgInputError, CfgPreconditionError) as exc:
        raise cfg_error_to_remote(exc) from exc
    follow_tab(adapter, tab_id, "run")
    return build_resource_observation(result)
