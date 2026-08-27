"""View remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._common import render_view


def _h_adapter_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"adapters": list(adapter.ctrl.get_adapter_names())}


def _h_adapter_guide(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["adapter_name"])
    if name not in adapter.ctrl.get_adapter_names():
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown adapter: {name!r}")
    return {"guide": adapter.ctrl.get_adapter_guide(name)}


def _h_app_shutdown(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Graceful close: trigger the window's normal close path (persist session,
    # tear down remote, cleanup) on the main thread. request_shutdown defers the
    # actual close to the next event-loop turn so this reply is sent before the
    # remote service tears down. No kill / OS signal — that path is the agent's
    # cross-platform-safe way to stop the GUI.
    del params
    render_view(adapter).request_shutdown()
    return {"shutting_down": True}


def _h_view_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    snap = render_view(adapter).get_view_snapshot()
    if not isinstance(snap, dict):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"view snapshot returned non-dict {type(snap).__name__}",
        )
    return snap


def _h_dialog_screenshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import base64

    from ..dialogs import parse_dialog_name

    name_str = str(params["name"])
    dialog_name = parse_dialog_name(name_str)
    png = render_view(adapter).take_dialog_screenshot(dialog_name)
    if not isinstance(png, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"screenshot returned non-bytes {type(png).__name__}",
        )
    payload = base64.b64encode(bytes(png)).decode("ascii")
    return {"png_b64": payload, "bytes": len(png)}


def _h_view_screenshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import base64

    del params
    # Not off_main_thread → MainWindow.grab() is auto-marshalled to the Qt main
    # thread, the same path as dialog.screenshot. The whole window always exists
    # (headless is already fast-failed by _render_view), so there is no
    # PRECONDITION branch like the per-dialog grab.
    png = render_view(adapter).take_window_screenshot()
    if not isinstance(png, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"window screenshot returned non-bytes {type(png).__name__}",
        )
    payload = base64.b64encode(bytes(png)).decode("ascii")
    return {"png_b64": payload, "bytes": len(png)}


_VALID_SUBTABS = frozenset({"run", "analysis", "post_analysis"})


def _h_tab_get_figure(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    import base64
    from pathlib import Path

    from zcu_tools.gui.app.main.adapter import AnalysisMode

    tab_id = str(params["tab_id"])
    subtab_id = str(params["subtab_id"])
    if subtab_id not in _VALID_SUBTABS:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"invalid subtab_id {subtab_id!r}; expected one of {sorted(_VALID_SUBTABS)}",
        )
    # Existence check via tab control (same gate as other tab handlers)
    if not adapter.tab_control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    out_path_raw = params.get("out_path")
    out_path: str | None = str(out_path_raw) if out_path_raw is not None else None
    png: bytes | bytearray | None = None
    if subtab_id == "run":
        view = render_view(adapter)
        # Prefer pane-aware view helper if present; otherwise fallback to explicit widget access.
        # MainWindow (ViewProtocol) now exposes take_figure_screenshot_for_subtab;
        # tests inject a MagicMock render_view with the same stub.
        pane_helper = getattr(view, "take_figure_screenshot_for_subtab", None)
        if callable(pane_helper):
            png = pane_helper(tab_id, subtab_id)  # type: ignore[misc]
        else:
            # Legacy fallback for injected fakes that only expose take_figure_screenshot
            # — still require explicit run routing via widget helper if available.
            tab_w = (
                getattr(view, "_tab_widgets", {}).get(tab_id)
                if hasattr(view, "_tab_widgets")
                else None
            )
            if tab_w is not None and hasattr(tab_w, "get_current_figure_for_pane"):
                fig = tab_w.get_current_figure_for_pane("run")  # type: ignore[attr-defined]
                if fig is None:
                    raise RemoteError(
                        ErrorCode.PRECONDITION_FAILED,
                        f"tab {tab_id!r} run pane has no figure yet",
                    )
                from zcu_tools.gui.app.main.figure_export import render_figure_png

                png = render_figure_png(fig)
            else:
                png = view.take_figure_screenshot(tab_id)  # type: ignore[attr-defined]
    else:
        snap = adapter.tab_control.get_tab_snapshot(tab_id)
        if snap.capabilities is None:
            raise RemoteError(ErrorCode.INTERNAL, "snapshot has no capabilities")
        if subtab_id == "analysis":
            if snap.capabilities.analysis is AnalysisMode.NONE:
                raise RemoteError(
                    ErrorCode.PRECONDITION_FAILED,
                    f"tab {tab_id!r} does not support analysis",
                )
            fig = snap.analysis.figure if snap.analysis is not None else None
            if fig is None:
                raise RemoteError(
                    ErrorCode.PRECONDITION_FAILED,
                    f"tab {tab_id!r} analysis has no figure yet",
                )
        else:  # post_analysis
            if not snap.capabilities.post_analysis:
                raise RemoteError(
                    ErrorCode.PRECONDITION_FAILED,
                    f"tab {tab_id!r} does not support post_analysis",
                )
            fig = snap.post_analysis.figure if snap.post_analysis is not None else None
            if fig is None:
                raise RemoteError(
                    ErrorCode.PRECONDITION_FAILED,
                    f"tab {tab_id!r} post_analysis has no figure yet",
                )
        from zcu_tools.gui.app.main.figure_export import render_figure_png

        png = render_figure_png(fig)  # type: ignore[arg-type]
    if not isinstance(png, (bytes, bytearray)):
        raise RemoteError(
            ErrorCode.INTERNAL,
            f"figure screenshot returned non-bytes {type(png).__name__}",
        )
    if out_path:
        Path(out_path).write_bytes(bytes(png))
        return {"bytes": len(png), "saved_to": out_path}
    return {"png_b64": base64.b64encode(bytes(png)).decode("ascii"), "bytes": len(png)}
