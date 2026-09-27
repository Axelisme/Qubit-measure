"""Fixed device tools over the GUI's device service and operation handles."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def devices(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> list[dict[str, Any]] | dict[str, Any]:
    raise NotImplementedError("device listing and projection")


def device_connect(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    raise NotImplementedError("synchronous device connect")


def device_disconnect(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    raise NotImplementedError("synchronous device disconnect")


def device_set(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("validated device setup with short wait")


DEVICE_TOOLS: dict[str, dict[str, Any]] = {
    "devices": {
        "handler": devices,
        "description": "List devices by name/type/connected, or read one device with its address, error and live field choices.",
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
