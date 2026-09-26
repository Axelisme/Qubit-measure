"""GUI-side wire projection for one active interactive analysis session."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from matplotlib.figure import Figure

from zcu_tools.gui.app.main.figure_export import render_figure_png
from zcu_tools.gui.app.main.interactive import PluginDefinition
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.param_spec import build_input_schema, validate_params

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def _decode_payload(
    payload: object, plugin: PluginDefinition[Any, Any]
) -> tuple[str, dict[str, object]] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "payload must be an object")
    if set(payload) - {"command", "args"}:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "unknown interactive payload keys")
    name = payload.get("command")
    if not isinstance(name, str) or not name:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "payload.command must be a non-empty string"
        )
    args = payload.get("args", {})
    if not isinstance(args, dict):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "payload.args must be an object")
    if name == "done":
        if args:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "done takes no args")
        return name, {}
    command = next((item for item in plugin.commands if item.name == name), None)
    if command is None:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"unknown interactive command {name!r}"
        )
    unexpected = set(args) - {spec.name for spec in command.params}
    if unexpected:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"unknown args for {name!r}: {sorted(unexpected)}",
        )
    return name, validate_params(command.params, args)


def _project(
    plugin: PluginDefinition[Any, Any],
    state: object,
    figure: Figure | None,
    *,
    preview_active: bool,
) -> dict[str, object]:
    image = None
    if figure is not None:
        png = render_figure_png(figure)
        image = {"png_b64": base64.b64encode(png).decode("ascii"), "bytes": len(png)}
    commands = [
        {"name": item.name, "params": build_input_schema(item.params)}
        for item in plugin.commands
    ]
    commands.append({"name": "done", "params": build_input_schema(())})
    return {
        "plugin": plugin.plugin_id,
        "info": dict(plugin.info()),
        "state": state,
        "commands": commands,
        "figure": image,
        "preview_active": preview_active,
    }


def _h_tab_interact(  # pyright: ignore[reportUnusedFunction] - dynamically resolved method entry
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Dispatch on the owner loop after the existing resource-version guard."""
    tab_id = cast(str, params["tab_id"])
    control = adapter.run_analyze_control
    if not control.has_tab(tab_id):
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown tab_id: {tab_id!r}")
    active = control.get_interactive(tab_id)
    if active is None:
        raise FailedPreconditionError(
            f"tab {tab_id!r} has no active interactive analysis"
        )
    plugin, session = active.plugin, active.session
    decoded = _decode_payload(params.get("payload"), plugin)
    if decoded is not None:
        name, args = decoded
        if name == "done":
            prior_result = control.get_tab_analyze_result(tab_id)
            state = plugin.project_state(session.snapshot())
            # The facet discards the local preview before finishing from state.
            control.finish_interactive(tab_id)
            result = control.get_tab_analyze_result(tab_id)
            candidate = getattr(result, "figure", None)
            figure = (
                candidate
                if result is not prior_result and isinstance(candidate, Figure)
                else None
            )
            return _project(plugin, state, figure, preview_active=False)
        plugin.execute_command(session, name, args)
    state = plugin.project_state(session.snapshot())
    presentation = (
        adapter.render_view.interactive_presentation(tab_id)
        if adapter.render_view is not None
        else None
    )
    figure, preview_active = presentation if presentation is not None else (None, False)
    return _project(plugin, state, figure, preview_active=preview_active)
