"""The three fixed low-frequency tools backed by the live GUI catalog."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import CatalogEntry, GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def _entry(ctx: MeasureToolContext, name: object) -> CatalogEntry:
    ctx.session.ensure_connected()
    if not isinstance(name, str) or name not in ctx.session.catalog:
        raise GuiRpcError(f"unknown GUI method {name!r}", reason="unknown_method")
    return ctx.session.catalog[name]


def rpc_list(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    ctx.session.ensure_connected()
    domain = arguments.get("domain")
    if domain is not None and not isinstance(domain, str):
        raise ValueError("domain must be a string")
    return {
        "methods": [
            {
                "method": entry["method"],
                "description": entry["description"].split(".", 1)[0],
                "tool_names": entry["tool_names"],
            }
            for entry in ctx.session.catalog.values()
            if domain is None or entry["method"].split(".", 1)[0] == domain
        ]
    }


def rpc_describe(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    return dict(_entry(ctx, arguments.get("method")))


def rpc_call(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    entry = _entry(ctx, arguments.get("method"))
    if entry["exposure"] == "tool":
        tools = ", ".join(entry["tool_names"])
        raise GuiRpcError(f"use {tools} for {entry['method']}", reason="use_tool")
    params = arguments.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    return ctx.session.send_gui_rpc(entry["method"], params, rpc_only=True)


RPC_TOOLS: dict[str, dict[str, Any]] = {
    "rpc_list": {
        "handler": rpc_list,
        "description": "List low-frequency live GUI wire methods, optionally by domain. Tool-bound methods name their required tool.",
        "inputSchema": {"type": "object", "properties": {"domain": {"type": "string"}}},
    },
    "rpc_describe": {
        "handler": rpc_describe,
        "description": "Describe a live GUI method, its parameter schema, guard and tool routing.",
        "inputSchema": {
            "type": "object",
            "properties": {"method": {"type": "string"}},
            "required": ["method"],
        },
    },
    "rpc_call": {
        "handler": rpc_call,
        "description": "Call a low-frequency live GUI method. GUI validates params; mutations are not retried.",
        "inputSchema": {
            "type": "object",
            "properties": {"method": {"type": "string"}, "params": {"type": "object"}},
            "required": ["method"],
        },
    },
}


def build_rpc_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in RPC_TOOLS.items()
    }
