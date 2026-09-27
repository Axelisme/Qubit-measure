"""Connection Device remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _bool_default,
    _int_opt,
    _obj,
    _str,
    _str_opt,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "soc.connect",
        "connection_device:_h_soc_connect",
        MethodSpec(
            # Synchronous connect (runs on the main thread; the IO worker blocks on it).
            # Bounded by make_soc_proxy's 1s COMMTIMEOUT for a remote board (mock is
            # instant); a small margin above that keeps the timeout from firing before
            # make_soc_proxy's own clean error does.
            3.0,
            "Connect the SoC SYNCHRONOUSLY and return its summary. kind='mock' for an "
            "offline mock board, or kind='remote' with ip + port for a real board "
            "(ip/port required only when kind='remote'). Returns {soc: {description, "
            "is_mock}} once connected (the structured cfg is read on demand via "
            "soc.info). A remote connect fails fast (~1s) if the board is unreachable.",
            (
                _str("kind", "'mock' or 'remote'"),
                _str_opt("ip", "Board IP (required when kind='remote')"),
                _int_opt("port", "Board port (required when kind='remote')"),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "startup.apply",
        "connection_device:_h_startup_apply",
        MethodSpec(
            30.0,
            "Atomically update project chip / qubit / resonator names; omitted names "
            "inherit an already applied project. Without a project all three names "
            "are required. scope_id selects a discovered result scope; when omitted "
            "after a chip/qubit change, the GUI uses the new identity's generated "
            "scope; otherwise it retains the old scope. Effective changes deactivate "
            "the selected context; no-op/failed updates leave it selected. Explicit "
            "result_dir/database_path overrides are not accepted. Echoes the resolved "
            "project: {chip_name, qub_name, res_name, result_dir, database_path, "
            "params_path, scope_id}.",
            (
                _str_opt("chip_name", "Chip identity; required for first project"),
                _str_opt("qub_name", "Qubit identity; required for first project"),
                _str_opt("res_name", "Resonator identity; required for first project"),
                _str_opt(
                    "scope_id",
                    "Optional scope_id returned by result_scope.list",
                ),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "device.connect",
        "connection_device:_h_device_connect",
        MethodSpec(
            30.0,
            "Connect a hardware device by driver type, friendly name, and address. "
            "The connection runs asynchronously. The GUI returns operation_id; "
            "rpc_call maps it to a handle. Call wait(op=handle) to observe terminal "
            "status before reading device.snapshot. 'remember' persists the "
            "device across sessions (default true).",
            (
                _str(
                    "type_name", "Driver class name, e.g. 'YOKOGS200' or 'FakeDevice'"
                ),
                _str("name", "Friendly name for this device"),
                _str("address", "VISA, GPIB, or IP address"),
                _bool_default(
                    "remember",
                    True,
                    "Persist device across sessions (default true)",
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            operation_key="device:{name}", refresh_after_write=True
        ),
    ),
    method_entry(
        "device.disconnect",
        "connection_device:_h_device_disconnect",
        MethodSpec(
            30.0,
            "Disconnect a registered device by name via rpc_call. The call starts "
            "an asynchronous operation and returns an MCP handle; use "
            "wait(op=handle) for the terminal status. 'remember' keeps the "
            "device in persistent storage (default true).",
            (
                _str("name", "Device name"),
                _bool_default(
                    "remember",
                    True,
                    "Keep device in persistent storage (default true)",
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            operation_key="device:{name}", refresh_after_write=True
        ),
    ),
    method_entry(
        "device.reconnect",
        "connection_device:_h_device_reconnect",
        MethodSpec(
            30.0,
            "Reconnect a remembered (memory-only) device by name, reusing its stored "
            "type/address. Call via rpc_call with name; the asynchronous "
            "operation returns an MCP handle. Use wait(op=handle) for its "
            "terminal status, then rpc_call on device.snapshot for state.",
            (_str("name", "Device name"),),
        ),
        agent=AgentMethodPolicy(
            operation_key="device:{name}", refresh_after_write=True
        ),
    ),
    method_entry(
        "device.forget",
        "connection_device:_h_device_forget",
        MethodSpec(
            5.0,
            "Forget a memory-only device (synchronous). Echoes {forgotten: name}.",
            (_str("name", "Device name"),),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "device.setup",
        "connection_device:_h_device_setup",
        MethodSpec(
            30.0,
            "Apply 'updates' to a connected device by name via rpc_call. "
            "The GUI starts an asynchronous setup operation; MCP returns a "
            "handle. Use wait(op=handle) for terminal status and Stop feedback, "
            "then rpc_call on device.snapshot for current values.",
            (_str("name", "Device name"), _obj("updates", "Field updates")),
        ),
        agent=AgentMethodPolicy(
            operation_key="device:{name}", refresh_after_write=True
        ),
    ),
    method_entry(
        "device.setup_spec",
        "connection_device:_h_device_setup_spec",
        MethodSpec(
            5.0,
            "List the fields accepted by device.setup's 'updates' for a connected "
            "device: {fields: [{name, type, current, settable, choices?}, ...]} — each "
            "field's name, type, choices (for enum/Literal fields like output/mode), "
            "current value, and whether it is settable (the protected type/address are "
            "reported settable=false). Use this RPC before rpc_call on "
            "device.setup. The device must be connected.",
            (_str("name", "Device name"),),
        ),
    ),
    method_entry(
        "device.cancel_operation",
        "connection_device:_h_device_cancel_operation",
        MethodSpec(
            5.0,
            "Request cancellation of the named device's in-flight operation. Returns "
            "{ok: true, cancelled: true}. Note: only a device APPLY (setup ramp) has a "
            "cancellation point; a connect/disconnect has none and cannot be "
            "cancelled (it raises PRECONDITION_FAILED).",
            (_str("name", "Device name"),),
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "device.active_operations",
        "connection_device:_h_device_active_operations",
        MethodSpec(
            5.0,
            "List EVERY in-flight device operation (connect / disconnect / apply run "
            "concurrently): {operations: [{handle, device_name, kind, type_name, "
            "address, status, error}, ...]} (empty list if none), sorted by device "
            "name. 'handle' here is a GUI-local id, not the MCP op. Use "
            "status() to obtain MCP handles before wait(op). 'kind' is "
            "device_connect / device_disconnect / device_setup.",
        ),
    ),
    method_entry(
        "device.list",
        "connection_device:_h_device_list",
        MethodSpec(
            5.0,
            "List registered devices with their current lifecycle status: "
            "{devices: [{name, type_name, status}, ...]} where status is one of "
            "memory_only | connecting | connected | disconnecting | setting_up "
            "(same status vocabulary as device.snapshot and "
            "device.active_operations). 'memory_only' means remembered but not "
            "live (no driver).",
        ),
        agent=AgentMethodPolicy(reveals=("devices:__set__",)),
    ),
    method_entry(
        "device.snapshot",
        "connection_device:_h_device_snapshot",
        MethodSpec(
            5.0,
            "Read one device's full cached snapshot — the richest single-device read: "
            "{snapshot: {name, type_name, address, status, error, info, fields}} "
            "where 'info' is the State-cached device parameter dict (or null "
            "without info) and 'fields' is its cached field/choice projection. "
            "During setting_up these remain readable without driver I/O. "
            "'status' uses the same vocabulary as device.list. An unknown device "
            "name raises INVALID_PARAMS.",
            (_str("name", "Device name"),),
        ),
        agent=AgentMethodPolicy(reveals=("device:{name}",)),
    ),
)
