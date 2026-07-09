"""Remote method routing for autofluxdep-gui.

This module binds the Qt-free method specs to small synchronous handlers. The
handlers deliberately depend on ``adapter.read_model`` instead of the full
Controller, keeping the remote bridge read-only by construction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.method_spec import build_method_registry

from .method_specs import METHOD_SPECS

if TYPE_CHECKING:
    from .service import RemoteControlAdapter

Handler = Callable[["RemoteControlAdapter", Mapping[str, object]], Mapping[str, object]]


def _h_project_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return adapter.read_model.project_info()


def _h_workflow_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return adapter.read_model.workflow_list()


def _h_node_cfg(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = params["name"]
    if not isinstance(name, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "node.cfg name must be a string")
    return adapter.read_model.node_cfg(name)


def _h_result_summary(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return adapter.read_model.result_summary()


def _h_ui_screenshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    target = params.get("target", "window")
    if not isinstance(target, str) or target != "window":
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"target must be 'window', got {target!r}",
        )
    try:
        data = adapter.take_screenshot(target)
    except (RuntimeError, ValueError) as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    if not isinstance(data, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"screenshot returned non-bytes {type(data).__name__}",
        )
    path = Path(gettempdir()) / "zcu_tools_autofluxdep_window_screenshot.png"
    path.write_bytes(bytes(data))
    return {"target": target, "path": str(path), "bytes": len(data)}


def _h_state_check(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return adapter.read_model.state_check()


def _h_resources_versions(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return adapter.read_model.resources_versions()


_HANDLERS: dict[str, Handler] = {
    "project.info": _h_project_info,
    "workflow.list": _h_workflow_list,
    "node.cfg": _h_node_cfg,
    "result.summary": _h_result_summary,
    "ui.screenshot": _h_ui_screenshot,
    "resources.versions": _h_resources_versions,
    "state.check": _h_state_check,
}

METHOD_REGISTRY = build_method_registry(_HANDLERS, METHOD_SPECS)

__all__ = ["METHOD_REGISTRY", "_HANDLERS"]
