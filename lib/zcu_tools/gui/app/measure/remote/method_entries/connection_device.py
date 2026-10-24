"""Connection Device remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    required_integer,
    required_object,
    required_string,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "soc.connect",
        "connection_device:h_soc_connect",
        MethodSpec(
            # Synchronous connect (runs on the main thread; the IO worker blocks on it).
            # A margin above make_soc_proxy's 1s COMMTIMEOUT lets its error return.
            3.0,
            "Connect a real SoC SYNCHRONOUSLY with kind='remote', ip and port. "
            "Mock connections are rejected; use a trusted startup script (--script) to "
            "install simulation resources. Returns {soc: {description, is_mock}} once connected; "
            "read soc.info for structured cfg. Unreachable boards fail fast (~1s).",
            (
                required_string("kind", "Must be 'remote'"),
                required_string("ip", "Real board IP"),
                required_integer("port", "Real board port"),
            ),
        ),
    ),
    method_entry(
        "device.setup",
        "connection_device:h_device_setup",
        MethodSpec(
            30.0,
            "Apply a complete Info cfg to a device by name via rpc_call. "
            "All model-required fields must be supplied; defaults follow that Info "
            "constructor, never cached values. Extra fields and validator errors "
            "are rejected before setup. YOKO requires mode; omitted value_unit "
            "is generated from mode. The GUI starts an asynchronous operation; "
            "MCP returns a handle. Use wait(op=handle) for terminal status, then "
            "device.snapshot for cached facts. For partial cfg and a fixed bounded "
            "wait use the device_setup tool.",
            (
                required_string("name", "Device name"),
                required_object("cfg", "Complete Info constructor cfg"),
            ),
        ),
        agent=AgentMethodPolicy(
            operation_key="device:{name}",
        ),
    ),
    method_entry(
        "device.list",
        "connection_device:h_device_list",
        MethodSpec(
            5.0,
            "List registered devices with their current lifecycle status: "
            "{devices: [{name, type_name, status}, ...]} where status is one of "
            "memory_only | connecting | connected | disconnecting | setting_up "
            "(same status vocabulary as device.snapshot). "
            "'memory_only' means remembered but not "
            "live (no driver).",
            readonly=True,
        ),
    ),
    method_entry(
        "device.snapshot",
        "connection_device:h_device_snapshot",
        MethodSpec(
            5.0,
            "Read one device's full cached snapshot — the richest single-device read: "
            "{snapshot: {name, type_name, address, status, error, info, fields}} "
            "where 'info' is the State-cached device parameter dict (or null "
            "without info) and 'fields' is its cached field/choice projection. "
            "During setting_up these remain readable without driver I/O. "
            "'status' uses the same vocabulary as device.list. An unknown device "
            "name raises INVALID_PARAMS.",
            (required_string("name", "Device name"),),
            readonly=True,
        ),
    ),
)
