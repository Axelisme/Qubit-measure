"""Measure MCP tools-soc override tools."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import (
    METHOD_SPECS,
    MeasureToolContext,
    _is_timeout_error,
    bind_context,
    send_gui_rpc,
)

_SOC_CONNECT_TIMEOUT_SLACK = 0.25


_SOC_CONNECT_RECONCILE_TIMEOUT = 1.0


def _soc_connect_rpc_timeout() -> float:
    return METHOD_SPECS["soc.connect"].timeout_seconds + _SOC_CONNECT_TIMEOUT_SLACK


def _is_soc_connect_timeout(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, GuiRpcError) and exc.code == "timeout":
        return True
    return isinstance(exc, RuntimeError) and _is_timeout_error(exc)


def _soc_summary_from_info(info: dict[str, Any]) -> dict[str, Any]:
    return {"description": info.get("description"), "is_mock": info.get("is_mock")}


def _soc_timeout_message(timeout: float, detail: str) -> str:
    return (
        f"GUI RPC 'soc.connect' did not return within {timeout:g}s. "
        "The GUI may still complete the SoC connection and post-connect side "
        f"effects after the MCP timeout; {detail}. Re-run gui_overview before retrying."
    )


def _reconcile_soc_connect_timeout(timeout: float, exc: Exception) -> dict[str, Any]:
    try:
        has_soc = bool(
            send_gui_rpc("state.has_soc", {}, _SOC_CONNECT_RECONCILE_TIMEOUT).get(
                "value", False
            )
        )
    except Exception as reconcile_exc:
        raise TimeoutError(
            _soc_timeout_message(
                timeout,
                f"post-timeout reconciliation also failed ({reconcile_exc})",
            )
        ) from exc
    if not has_soc:
        raise TimeoutError(
            _soc_timeout_message(
                timeout, "post-timeout reconciliation found no connected SoC"
            )
        ) from exc
    try:
        info = send_gui_rpc("soc.info", {}, _SOC_CONNECT_RECONCILE_TIMEOUT)
    except Exception as info_exc:
        raise TimeoutError(
            _soc_timeout_message(
                timeout,
                f"post-timeout reconciliation found a SoC but soc.info failed ({info_exc})",
            )
        ) from exc
    return {
        "soc": _soc_summary_from_info(info),
        "warning": (
            f"soc.connect timed out after {timeout:g}s at the MCP layer, but "
            "post-timeout reconciliation found the GUI connected."
        ),
    }


def tool_gui_soc_connect(arguments: dict[str, Any]) -> dict[str, Any]:
    """Connect the SoC SYNCHRONOUSLY and return its hardware summary.

    Unlike run / analyze / device ops, connect is no longer a degrading async
    handle: the soc.connect RPC runs the connect on the GUI's main thread and
    returns once the board is connected and all post-connect side effects are
    applied, so this is one blocking call with no _wait / _poll follow-up.
    kind='mock' (offline) or kind='remote' with ip+port. Returns
    {soc:{description, is_mock}}; call gui_soc_info for the structured cfg. A
    remote connect to an unreachable board fails fast (~1s).

    The SoC has no teardown (Pyro4-backed): there is no disconnect / reconnect /
    health-check tool — those are deferred (E3).
    """
    params: dict[str, Any] = {"kind": str(arguments["kind"])}
    if "ip" in arguments:
        params["ip"] = str(arguments["ip"])
    if "port" in arguments:
        params["port"] = int(arguments["port"])
    # Use the wire spec's 3s budget plus a tiny transport slack so the GUI-side
    # handler budget fires first; do NOT fall back to the 30s generic default.
    timeout = _soc_connect_rpc_timeout()
    try:
        result = send_gui_rpc("soc.connect", params, timeout)
    except Exception as exc:
        if _is_soc_connect_timeout(exc):
            return _reconcile_soc_connect_timeout(timeout, exc)
        raise
    return {"soc": result.get("soc")}


NON_GENERATED_METHODS = frozenset(
    {
        "soc.connect",
    }
)


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_soc_connect": {
        "handler": tool_gui_soc_connect,
        "description": (
            "Connect the SoC SYNCHRONOUSLY (NOT a degrading async handle — there is "
            "no gui_soc_connect_wait / _poll). One blocking call returns "
            "{soc:{description, is_mock}} once the board is connected (call "
            "gui_soc_info for the structured cfg). kind='mock' (offline) or "
            "kind='remote' with ip+port. A remote connect to an unreachable board "
            "fails fast (~1s). The SoC has no teardown (Pyro4-backed): there is no "
            "disconnect / reconnect / health-check tool — those are deferred (E3)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "'mock' or 'remote'"},
                "ip": {"type": "string", "description": "Board IP (remote)"},
                "port": {"type": "integer", "description": "Board port (remote)"},
            },
            "required": ["kind"],
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
