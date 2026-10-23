"""View remote handlers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ..guide_types import AdapterGuideView
from ._common import render_view


def h_adapter_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"adapters": list(adapter.ctrl.get_adapter_names())}


def h_adapter_guide(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return guide Markdown and an absolute resource directory string or null.

    params['adapter_name'] names a registered adapter. Unknown names raise
    invalid_params; guide source errors propagate to the shared RPC dispatcher.
    Referenced images are not read or attached.
    """
    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    guide = adapter.ctrl.get_adapter_guide(name)
    view: AdapterGuideView = {
        "markdown": guide.markdown,
        "resource_dir": (
            str(guide.resource_dir.resolve())
            if guide.resource_dir is not None
            else None
        ),
    }
    return {"guide": view}


def h_app_shutdown(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Validate acceptance here. The adapter registers the owner-thread close
    # continuation with shared dispatch, after this reply reaches terminal send.
    from .lifecycle import require_idle, require_saved

    require_idle(adapter)
    if not params["discard_unsaved"]:
        require_saved(adapter, adapter.tab_control.list_tab_ids())
    render_view(adapter)  # Reject headless shutdown before accepting the command.
    return {"shutting_down": True, "pid": os.getpid()}


def h_view_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return render_view(adapter).get_view_snapshot()


def h_dialog_screenshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from ..dialogs import parse_dialog_name

    name_str = str(params["name"])
    dialog_name = parse_dialog_name(name_str)
    png = render_view(adapter).take_dialog_screenshot(dialog_name)
    return _png_reply(png, params)


def h_view_screenshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Not off_main_thread → MainWindow.grab() is auto-marshalled to the Qt main
    # thread, the same path as dialog.screenshot. The whole window always exists
    # (headless is already fast-failed by _render_view), so there is no
    # PRECONDITION branch like the per-dialog grab.
    png = render_view(adapter).take_window_screenshot()
    return _png_reply(png, params, description="window screenshot")


def _png_reply(
    png: object, params: Mapping[str, object], *, description: str = "screenshot"
) -> dict[str, object]:
    import base64

    if not isinstance(png, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"{description} returned non-bytes {type(png).__name__}",
        )
    out_path = params.get("out_path")
    if out_path is not None:
        path = str(out_path)
        Path(path).write_bytes(bytes(png))
        return {"saved_to": path, "bytes": len(png)}
    return {"png_b64": base64.b64encode(bytes(png)).decode("ascii"), "bytes": len(png)}


_VALID_SUBTABS = frozenset({"run", "analysis", "post_analysis"})


def h_tab_get_figure(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    tab_id = str(params["tab_id"])
    subtab_id = str(params["subtab_id"])
    if subtab_id not in _VALID_SUBTABS:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"invalid subtab_id {subtab_id!r}; expected one of {sorted(_VALID_SUBTABS)}",
        )
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    # tolerated(cast): ParamSpec validates optional INTEGER; Mapping retains object values.
    operation_id = cast(int | None, params.get("operation_id"))
    # tolerated(cast): ParamSpec validates optional INTEGER; Mapping retains object values.
    run_operation_id = cast(int | None, params.get("run_operation_id"))
    if operation_id is not None and run_operation_id is not None:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "Operation tokens are mutually exclusive"
        )
    if run_operation_id is not None:
        if subtab_id != "run":
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "run_operation_id requires the run pane"
            )
        adapter.run_analyze_control.require_run_operation(tab_id, run_operation_id)
    if operation_id is not None:
        if subtab_id not in ("analysis", "post_analysis"):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "operation_id requires an analysis pane"
            )
        adapter.run_analyze_control.require_analysis_operation(
            tab_id, subtab_id, operation_id
        )
    png = render_view(adapter).take_figure_screenshot_for_subtab(tab_id, subtab_id)
    return _png_reply(png, params, description="figure screenshot")
