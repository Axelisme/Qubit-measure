"""MCP stdio server: tool generation from method specs and the JSON-RPC loop.

Every MCP server in this repo speaks the same stdio protocol to its host
(Claude / Gemini / VS Code). This module owns that protocol and the generic tool
surface, independent of any GUI connection:

  - :class:`McpServerConfig` carries the tool prefix, display name, and
    instructions. In-process servers (agent-memory) need only this; GUI bridges
    extend it with launch knobs in :mod:`zcu_tools.mcp.core.bridge`.
  - ``coerce_arg`` / ``make_forwarder`` / ``generate_tools`` build one MCP tool per
    wire method spec; ``assemble_tools`` merges hand-written overrides.
  - ``build_initialize_result`` / ``run_stdio_loop`` run the MCP stdio protocol on
    the main thread until stdin closes.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO

from zcu_tools.gui.remote.param_spec import JsonType, build_input_schema
from zcu_tools.mcp.core.reply import ToolReply

_GENERATED_RPC_TRANSPORT_SLACK_SECONDS = 1.0

# The type of a generated/override MCP tool entry.
Tool = dict[str, Any]
ToolTable = dict[str, Tool]
# A function issuing one GUI RPC and returning the result dict (raises on error).
# Read-only apps pass a thin wrapper over McpBridge.send_rpc_raw; measure-gui
# passes its guarded send_gui_rpc.
SendFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class McpServerConfig:
    """The minimal config the stdio loop + tool generation need.

    ``tool_prefix`` is the wire-method -> tool-name prefix (e.g. ``fluxdep_``). A
    no-subprocess server (e.g. agent-memory, dispatching in-process) needs only
    this; the GUI bridges add the launch fields via :class:`MCPBridgeConfig`.
    """

    tool_prefix: str
    server_display_name: str
    server_instructions: str


# ---------------------------------------------------------------------------
# Tool generation from a method-spec table (the wire SSOT)
# ---------------------------------------------------------------------------


# JSON-decoded tool arguments are untyped; each converter accepts what the JSON
# decoder can produce for that declared type (e.g. "4" or 4 for INTEGER).
_SCALAR_COERCERS: dict[JsonType, Callable[[Any], object]] = {
    JsonType.STRING: str,
    JsonType.INTEGER: int,
    JsonType.NUMBER: float,
    JsonType.BOOLEAN: bool,
    JsonType.OBJECT: dict,
}


def coerce_arg(value: Any, json_type: JsonType) -> object:
    """Project one JSON-decoded tool argument onto its declared wire type."""
    if value is None:
        return None
    if json_type is JsonType.ARRAY:
        # param_spec.validate_params enforces this on the wire path, but the
        # forwarder calls coerce_arg directly, so the list guard lives here too.
        if not isinstance(value, list):
            raise TypeError(
                f"expected list for ARRAY param, got {type(value).__name__!r}"
            )
        return list(value)
    coerce = _SCALAR_COERCERS.get(json_type)
    return value if coerce is None else coerce(value)  # JSON: pass through


def generated_rpc_timeout_seconds(spec: Any) -> float:
    """Transport ceiling for generated tools: handler budget plus wire slack."""

    return float(spec.timeout_seconds) + _GENERATED_RPC_TRANSPORT_SLACK_SECONDS


def make_forwarder(method: str, spec, send_fn: SendFn):
    """Build an MCP forwarder that projects arguments into RPC params per spec.

    ``send_fn`` issues the RPC: read-only apps pass a thin error-raising wrapper
    over :meth:`McpBridge.send_rpc_raw`; measure-gui passes its guarded
    ``send_gui_rpc``.
    """
    rpc_timeout = generated_rpc_timeout_seconds(spec)

    def _forwarder(arguments: dict[str, Any]) -> dict[str, Any]:
        rpc_params: dict[str, Any] = {}
        for p in spec.params:
            if p.required:
                if p.name not in arguments or arguments[p.name] is None:
                    raise ValueError(f"missing {p.name!r}")
                rpc_params[p.name] = coerce_arg(arguments[p.name], p.json_type)
            elif arguments.get(p.name) is not None:
                rpc_params[p.name] = coerce_arg(arguments[p.name], p.json_type)
        return send_fn(method, rpc_params, timeout_seconds=rpc_timeout)

    return _forwarder


def generate_tools(
    config: McpServerConfig,
    method_specs: dict[str, Any],
    non_generated: frozenset[str],
    send_fn: SendFn,
) -> ToolTable:
    """Generate one MCP tool per method spec (skipping ``non_generated``)."""
    out: ToolTable = {}
    for method, spec in method_specs.items():
        if method in non_generated:
            continue
        tool_name = spec.tool_name or config.tool_prefix + method.replace(".", "_")
        out[tool_name] = {
            "handler": make_forwarder(method, spec, send_fn),
            "description": spec.description or method,
            "inputSchema": build_input_schema(spec.params),
        }
    return out


def assemble_tools(
    generated: ToolTable, overrides: ToolTable, override_names: frozenset[str]
) -> ToolTable:
    """Merge generated + selected override tools; fail-fast on name collision."""
    selected = {
        name: spec for name, spec in overrides.items() if name in override_names
    }
    collisions = set(generated) & set(selected)
    if collisions:
        raise RuntimeError(f"override/generated tool collision: {sorted(collisions)}")
    return {**generated, **selected}


# ---------------------------------------------------------------------------
# MCP stdio protocol loop
# ---------------------------------------------------------------------------


def build_initialize_result(
    config: McpServerConfig, server_version: str = "1.0.0"
) -> dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": config.server_display_name, "version": server_version},
        "instructions": config.server_instructions,
    }


@dataclass(frozen=True)
class StdioLoopHooks:
    """Optional lifecycle hooks for :func:`run_stdio_loop`.

    - ``on_start`` runs once after stdin/stdout are reconfigured to UTF-8, before
      the loop (measure-gui attaches its per-session file logging here).
    - ``on_cleanup`` runs once when stdin closes (e.g. stop a server-launched GUI).
    - ``on_each_reply`` lets an app append ready-made content blocks to each
      successful tool reply: a list of ``{"type": "text", "text": ...}`` dicts,
      appended after the tool's own content. The hook owns the wording (returns
      ``[]`` for nothing). Measure-gui does not register this hook; it reads
      Stop feedback through operation request/reply instead.
    - ``on_error`` is called from within each ``except`` block with a
      preformatted context message (measure-gui passes ``logger.exception``) so
      the active exception is logged with its traceback.
    """

    on_start: Callable[[], None] | None = None
    on_cleanup: Callable[[], None] | None = None
    on_each_reply: Callable[[], list[dict[str, Any]]] | None = None
    on_error: Callable[[str], None] | None = None


def run_stdio_loop(
    config: McpServerConfig,
    tools: ToolTable,
    *,
    hooks: StdioLoopHooks | None = None,
    server_version: str = "1.0.0",
) -> None:
    """Run the MCP stdio JSON-RPC loop until stdin closes.

    ``server_version`` is the ``serverInfo.version`` reported on ``initialize``.

    A ``RuntimeError`` carrying a ``reason`` attribute (set from the GUI wire error
    envelope) has its tag appended to the tool-error text, so an agent can branch
    on the machine-readable reason without parsing the prose.
    """
    hooks = hooks or StdioLoopHooks()
    _use_utf8(sys.stdout)
    _use_utf8(sys.stdin)
    if hooks.on_start is not None:
        hooks.on_start()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                if hooks.on_cleanup is not None:
                    hooks.on_cleanup()
                break
            line = line.strip()
            if not line:
                continue
            resp = _handle_request(
                json.loads(line), config, tools, hooks, server_version
            )
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as e:  # noqa: BLE001 — loop isolation keeps the server alive
            if hooks.on_error is not None:
                hooks.on_error("MCP loop exception")
            sys.stderr.write(f"MCP Loop Exception: {e}\n{traceback.format_exc()}\n")
            sys.stderr.flush()


def _use_utf8(stream: TextIO) -> None:
    if not isinstance(stream, io.TextIOWrapper):
        raise RuntimeError(
            f"MCP stdio needs text-wrapper streams, got {type(stream).__name__}"
        )
    stream.reconfigure(encoding="utf-8")


def _handle_request(
    req: dict[str, Any],
    config: McpServerConfig,
    tools: ToolTable,
    hooks: StdioLoopHooks,
    server_version: str,
) -> dict[str, Any] | None:
    """Return the JSON-RPC response for one request, or None for a notification."""
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        result = build_initialize_result(config, server_version)
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        tools_list = [
            {
                "name": name,
                "description": info["description"],
                "inputSchema": info["inputSchema"],
            }
            for name, info in tools.items()
        ]
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": tools_list}}
    if method == "tools/call":
        params = req.get("params", {})
        return _call_tool(
            rid, params.get("name"), params.get("arguments", {}), tools, hooks
        )
    if rid is None:
        return None
    return _not_found(rid, method)


def _not_found(rid: object, name: object) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"Method not found: {name}"},
    }


def _call_tool(
    rid: object,
    name: str | None,
    arguments: dict[str, Any],
    tools: ToolTable,
    hooks: StdioLoopHooks,
) -> dict[str, Any]:
    tool = tools.get(name) if name is not None else None
    if not tool:
        return _not_found(rid, name)
    try:
        handler: Callable[[dict[str, Any]], Any] = tool["handler"]
        res = handler(arguments)
        # Compact separators (no indent, no spaces) keep the tool reply
        # token-light. ensure_ascii stays default (True): the outer JSON-RPC
        # envelope re-escapes non-ASCII anyway, so turning it off buys nothing.
        data = res.data if isinstance(res, ToolReply) else res
        text = (
            data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
        )
        content = [{"type": "text", "text": text}]
        if isinstance(res, ToolReply):
            content.extend(
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": base64.b64encode(image.data).decode("ascii"),
                }
                for image in res.images
            )
        if hooks.on_each_reply is not None:
            content.extend(hooks.on_each_reply())
    except Exception as e:  # noqa: BLE001 — tool isolation returns an isError reply
        if hooks.on_error is not None:
            hooks.on_error(f"MCP tool {name!r} dispatch failed")
        content = [{"type": "text", "text": _tool_error_text(name, e)}]
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"isError": True, "content": content},
        }
    result: dict[str, Any] = {"content": content}
    if isinstance(res, ToolReply) and res.is_error:
        result["isError"] = True
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _tool_error_text(name: str | None, exc: Exception) -> str:
    # GUI-side business errors (RuntimeError with an already-clear message) carry
    # no useful Python stack for the agent: the traceback is always the same
    # forwarder frames. Keep the full traceback only for unexpected bridge-side
    # failures, where the stack is the actual debugging signal.
    if not isinstance(exc, RuntimeError):
        return f"Error executing tool {name!r}: {exc}\n{traceback.format_exc()}"
    text = f"Error executing tool {name!r}: {exc}"
    # Surface the machine-readable reason tag (e.g. no_run_result / no_project)
    # when the wire carried one, so the agent can branch on it without parsing.
    reason = getattr(exc, "reason", None)
    if reason:
        text += f"\nreason: {reason}"
    return text


__all__ = [
    "McpServerConfig",
    "SendFn",
    "StdioLoopHooks",
    "Tool",
    "ToolTable",
    "assemble_tools",
    "build_initialize_result",
    "coerce_arg",
    "generate_tools",
    "generated_rpc_timeout_seconds",
    "make_forwarder",
    "run_stdio_loop",
]
