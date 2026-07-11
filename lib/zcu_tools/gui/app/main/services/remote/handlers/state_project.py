"""State Project remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from zcu_tools.gui.result_scope import ResultScope

    from ..service import RemoteControlAdapter


def _h_state_has_project(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_project())}


def _h_state_has_context(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_context())}


def _h_state_has_active_context(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_active_context())}


def _h_state_has_soc(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_soc())}


def _h_soc_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    include_cfg = bool(params["include_cfg"])  # ParamSpec(_bool_default)-validated
    return adapter.ctrl.get_soc_info(include_cfg=include_cfg)


def _h_project_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Project identity (the chip / qubit / resonator names + their output roots).
    # It lives only on the in-process ExpContext, so this is the sole wire query
    # that exposes it — _assemble_overview folds {chip, qub, res} from here. The
    # res_name field is measure-specific (the other GUIs' shared project.info
    # carries only chip/qub/result_dir/database_path).
    del params
    if not adapter.ctrl.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; apply a project first (gui_project_apply).",
            reason="no_project",
        )
    ctx = adapter.ctrl.get_exp_context()
    return {
        "chip_name": ctx.chip_name,
        "qub_name": ctx.qub_name,
        "res_name": ctx.res_name,
        "result_dir": ctx.result_dir,
        "database_path": ctx.database_path,
    }


def _result_scope_wire(scope: ResultScope) -> dict[str, object]:
    return {
        "scope_id": scope.scope_id,
        "chip_name": scope.chip_name,
        "qub_name": scope.qub_name,
        "result_dir": scope.result_dir,
        "params_path": scope.params_path,
        "source": scope.source,
    }


def _h_result_scope_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    scopes = list(adapter.ctrl.list_result_scopes())
    return {
        "scopes": [_result_scope_wire(scope) for scope in scopes],
        "chip_names": sorted({scope.chip_name for scope in scopes}),
    }


def _h_resources_versions(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"versions": adapter.ctrl.resources_versions()}
