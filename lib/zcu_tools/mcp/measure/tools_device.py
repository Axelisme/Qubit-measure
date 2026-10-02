"""Fixed device tools over the GUI's device service and operation handles."""

from __future__ import annotations

import time
from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.tools_operation import wait

_TERMINAL_WAIT_SECONDS = 30.0


def devices(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> list[dict[str, Any]] | dict[str, Any]:
    name = arguments.get("name")
    if name is None:
        return [
            {
                "name": item["name"],
                "type": item["type_name"],
                "connected": item["status"] in ("connected", "setting_up"),
            }
            for item in ctx.gui.read_internal("device.list", {})["devices"]
        ]

    ctx = ctx.bound()
    snapshot = ctx.gui.read_internal("device.snapshot", {"name": name})["snapshot"]
    status = snapshot["status"]
    connected = status in ("connected", "setting_up")
    return {
        "name": snapshot["name"],
        "type": snapshot["type_name"],
        "address": snapshot["address"],
        "connected": connected,
        "error": snapshot["error"],
        "fields": (
            _live_fields(ctx, name)
            if status == "connected"
            else snapshot["fields"]
            if status == "setting_up"
            else []
        ),
    }


def _live_fields(ctx: MeasureToolContext, name: str) -> list[dict[str, Any]]:
    return ctx.gui.read_internal("device.setup_spec", {"name": name})["fields"]


def _check_terminal(op: int, outcome: dict[str, Any]) -> bool:
    status = outcome["status"]
    if status == "finished":
        return True
    if status == "running":
        return False
    error = outcome.get("error", {})
    raise GuiRpcError(
        f"device operation {op} {status}: {error.get('message', status)}",
        reason=f"operation_{status}",
        code="precondition_failed",
    )


def _await_device(ctx: MeasureToolContext, op: int) -> None:
    deadline = time.monotonic() + _TERMINAL_WAIT_SECONDS
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        outcome = wait(ctx, {"op": op, "timeout": remaining})
        if _check_terminal(op, outcome):
            return
        if time.monotonic() >= deadline:
            raise GuiRpcError(
                f"device operation pending; use wait(op={op}) to recover",
                reason="device_pending",
                code="timeout",
            )


def device_connect(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    name = arguments["name"]
    has_type = "type" in arguments
    has_address = "address" in arguments
    if has_type != has_address:
        raise ValueError("provide both type and address, or name only to reconnect")
    ctx = ctx.bound()
    if has_type:
        started = ctx.send_gui_rpc(
            "device.connect",
            {
                "name": name,
                "type_name": arguments["type"],
                "address": arguments["address"],
            },
        )
    else:
        started = ctx.send_gui_rpc("device.reconnect", {"name": name})
    _await_device(ctx, started["handle"])
    result = devices(ctx, {"name": name})
    if not isinstance(result, dict):
        raise GuiRpcError("invalid device detail", reason="incompatible_wire")
    return result


def device_disconnect(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    name = arguments["name"]
    forget = arguments.get("forget", False)
    ctx = ctx.bound()
    started = ctx.send_gui_rpc(
        "device.disconnect", {"name": name, "remember": not forget}
    )
    _await_device(ctx, started["handle"])
    return {"name": name, "connected": False, "forgotten": forget}


def device_set(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = arguments["name"]
    values = arguments["values"]
    ctx = ctx.bound()
    fields = _live_fields(ctx, name)
    by_name = {field["name"]: field for field in fields}
    legal = sorted(key for key, field in by_name.items() if field["settable"])
    for key, value in values.items():
        field = by_name.get(key)
        if field is None or not field["settable"]:
            raise ValueError(f"invalid device field {key!r}; legal fields: {legal}")
        if "choices" in field and value not in field["choices"]:
            raise ValueError(
                f"invalid choice for device field {key!r}; "
                f"choices: {field['choices']}; legal fields: {legal}"
            )

    started = ctx.send_gui_rpc("device.setup", {"name": name, "updates": values})
    op = started["handle"]
    outcome = wait(ctx, {"op": op, "timeout": 0.25})
    if not _check_terminal(op, outcome):
        return {"status": "running", "op": op}
    return {"fields": _live_fields(ctx, name)}


DEVICE_TOOLS: dict[str, dict[str, Any]] = {
    "devices": {
        "handler": devices,
        "description": "List devices by name/type/connected, or read one device with its address, error and field choices. A setting_up device remains connected; detail reports its State-cached fields without polling hardware while the ramp runs.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1}},
        },
    },
    "device_connect": {
        "handler": device_connect,
        "description": "Connect a device synchronously. Supply both type and address for a first connection, or name only to reconnect a remembered device. Returns devices(name).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "type": {"type": "string", "minLength": 1},
                "address": {"type": "string", "minLength": 1},
            },
            "required": ["name"],
        },
    },
    "device_disconnect": {
        "handler": device_disconnect,
        "description": "Disconnect a connected device synchronously. If forget is true, also remove its remembered entry after successful disconnection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "forget": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        },
    },
    "device_set": {
        "handler": device_set,
        "description": "Validate and set native-unit device fields; return complete fields on a quick finish or status=running and an opaque op for a long ramp.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "values": {"type": "object", "minProperties": 1},
            },
            "required": ["name", "values"],
        },
    },
}


def build_device_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in DEVICE_TOOLS.items()
    }
