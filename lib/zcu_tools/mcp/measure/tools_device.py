"""Measure MCP tools-device override tools."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    MeasureToolContext,
    _start_op_with_short_wait,
    bind_context,
    send_gui_rpc,
)


def _device_snapshot(name: str) -> Any:
    """Fetch one device's snapshot (now including its live ``info`` params)."""
    return send_gui_rpc("device.snapshot", {"name": name}).get("snapshot")


def tool_gui_device_connect(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments["name"])
    # type_name/address omitted => reconnect a remembered (memory-only) device,
    # reusing its stored type/address (E4: reconnect folded into connect). Both
    # wire methods key on device:{name} in _OP_BY_KEY, so the short-wait/handle
    # path is identical regardless of which one ran.
    type_name = arguments.get("type_name")
    address = arguments.get("address")
    if type_name is None and address is None:
        send_gui_rpc("device.reconnect", {"name": name})
    else:
        params: dict[str, Any] = {
            "type_name": str(type_name),
            "name": name,
            "address": str(address),
        }
        if "remember" in arguments:
            params["remember"] = bool(arguments["remember"])
        send_gui_rpc("device.connect", params)  # operation_id captured into _OP_BY_KEY
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} connect",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        "poll/wait the returned handle with gui_op_poll / gui_op_wait.",
    )


def tool_gui_device_disconnect(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments["name"])
    params: dict[str, Any] = {"name": name}
    if "remember" in arguments:
        params["remember"] = bool(arguments["remember"])
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("device.disconnect", params)
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} disconnect",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        "poll/wait the returned handle with gui_op_poll / gui_op_wait.",
    )


def tool_gui_device_setup(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments["name"])
    updates = arguments.get("updates", {})
    if not isinstance(updates, dict):
        raise ValueError("'updates' must be an object")
    wait_seconds = float(arguments.get("wait_seconds", 1.0))
    send_gui_rpc("device.setup", {"name": name, "updates": dict(updates)})
    return _start_op_with_short_wait(
        f"device:{name}",
        f"Device {name!r} apply",
        wait_seconds,
        lambda: {"snapshot": _device_snapshot(name)},
        "poll/wait the returned handle with gui_op_poll / gui_op_wait.",
    )


NON_GENERATED_METHODS = frozenset(
    {
        "device.connect",
        "device.disconnect",
        "device.setup",
        "device.reconnect",
    }
)


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_device_connect": {
        "handler": tool_gui_device_connect,
        "description": (
            "Connect a hardware device. Two modes by which params you pass:\n"
            "  - FIRST connect / re-register: pass type_name (driver class, e.g. "
            "'YOKOGS200', 'SGS100A') AND address (VISA/GPIB/IP). remember defaults "
            "to true (device persists across sessions); set remember=false for a "
            "memory-only device.\n"
            "  - RECONNECT a remembered device: pass ONLY name (omit type_name and "
            "address) — the stored type/address are reused (this also covers a "
            "memory-only device that was disconnected with remember=true).\n"
            "Waits up to wait_seconds (default 1.0): if it lands in time returns "
            "{status:'finished', handle, snapshot:{...}} (snapshot includes the "
            "device's live info params); otherwise {status:'pending', handle} — "
            "poll/wait the handle with gui_op_poll / gui_op_wait. The reply always "
            "carries 'handle'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Friendly name for this device",
                },
                "type_name": {
                    "type": "string",
                    "description": (
                        "Driver class name, e.g. 'YOKOGS200'. Omit (with address) "
                        "to reconnect a remembered device by name."
                    ),
                },
                "address": {
                    "type": "string",
                    "description": (
                        "VISA or IP address. Omit (with type_name) to reconnect a "
                        "remembered device by name."
                    ),
                },
                "remember": {
                    "type": "boolean",
                    "description": "Persist device across sessions (default true)",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["name"],
        },
    },
    "gui_device_disconnect": {
        "handler": tool_gui_device_disconnect,
        "description": (
            "Disconnect a device. Waits up to wait_seconds (default 1.0): if it "
            "lands in time returns {status:'finished', handle, snapshot:{...}}; "
            "otherwise {status:'pending', handle} — poll/wait the handle with "
            "gui_op_poll / gui_op_wait. The reply always carries 'handle'. Two "
            "terminal states by 'remember': remember=true (default) keeps the "
            "device in persistent storage as memory-only (reconnect later via "
            "gui_device_connect with name only); remember=false also removes it "
            "from persistent storage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "remember": {
                    "type": "boolean",
                    "description": "Keep device in persistent storage (default true)",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["name"],
        },
    },
    "gui_device_apply": {
        "handler": tool_gui_device_setup,
        "description": (
            "Apply device-field updates: patch the device's info fields via "
            "'updates' (e.g. {'value': 0.5} to ramp a source's output value — this "
            "is the way to set an output value, ramped/cancellable, no separate "
            "set_value). Waits up to wait_seconds (default 1.0): if it lands in "
            "time returns {status:'finished', handle, snapshot:{...}}; otherwise "
            "{status:'pending', handle} — poll/wait the handle with gui_op_poll / "
            "gui_op_wait (a 'running' poll reply carries the live progress bars, "
            "e.g. a ramp). The reply always carries 'handle'. The device must "
            "already be connected. Read the settable fields with gui_device_fields."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device name"},
                "updates": {
                    "type": "object",
                    "description": "Device info field updates (e.g. {'value': 0.5})",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before degrading to a handle (default 1.0)",
                },
            },
            "required": ["name", "updates"],
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
