"""Connection Device remote method specs."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _bool_default,
    _int_opt,
    _obj,
    _str,
    _str_opt,
)

SPECS: dict[str, MethodSpec] = {
    "soc.connect": MethodSpec(
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
    "startup.apply": MethodSpec(
        30.0,
        "Set the project: chip / qubit / resonator names, plus an optional "
        "scope_id returned by result_scope.list. Omitting scope_id uses or creates "
        "the generated result scope at <project-root>/result/<chip>/<qub>; explicit "
        "result_dir/database_path overrides are not accepted. Echoes the resolved "
        "project: {chip_name, qub_name, res_name, result_dir, database_path, "
        "params_path, scope_id}.",
        (
            _str("chip_name"),
            _str("qub_name"),
            _str("res_name"),
            _str_opt(
                "scope_id",
                "Optional scope_id returned by result_scope.list",
            ),
        ),
        tool_name="gui_project_apply",
    ),
    "device.connect": MethodSpec(
        30.0,
        "Connect a hardware device by driver type, friendly name, and address. "
        "Returns an operation_id; the connection runs asynchronously. "
        "'remember' persists the device across sessions (default true).",
        (
            _str("type_name", "Driver class name, e.g. 'YOKOGS200' or 'FakeDevice'"),
            _str("name", "Friendly name for this device"),
            _str("address", "VISA, GPIB, or IP address"),
            _bool_default(
                "remember",
                True,
                "Persist device across sessions (default true)",
            ),
        ),
    ),
    "device.disconnect": MethodSpec(
        30.0,
        "Disconnect a registered device by name. Returns an operation_id; the "
        "disconnection runs asynchronously. 'remember' keeps the device in "
        "persistent storage so it can be reconnected next session (default true).",
        (
            _str("name", "Device name"),
            _bool_default(
                "remember",
                True,
                "Keep device in persistent storage (default true)",
            ),
        ),
    ),
    "device.reconnect": MethodSpec(
        30.0,
        "Reconnect a remembered (memory-only) device by name, reusing its stored "
        "type/address. Returns an operation_id; the reconnection runs "
        "asynchronously. Wire-only: the MCP layer reaches this via "
        "gui_device_connect with type_name/address omitted.",
        (_str("name", "Device name"),),
    ),
    "device.forget": MethodSpec(
        5.0,
        "Forget a memory-only device (synchronous). Echoes {forgotten: name}.",
        (_str("name", "Device name"),),
    ),
    "device.setup": MethodSpec(
        30.0,
        "Setup device",
        (_str("name", "Device name"), _obj("updates", "Field updates")),
    ),
    "device.setup_spec": MethodSpec(
        5.0,
        "List the fields settable via gui_device_apply's 'updates' for a connected "
        "device: {fields: [{name, type, current, settable, choices?}, ...]} — each "
        "field's name, type, choices (for enum/Literal fields like output/mode), "
        "current value, and whether it is settable (the protected type/address are "
        "reported settable=false). This is the input source for gui_device_apply. "
        "The device must be connected.",
        (_str("name", "Device name"),),
        tool_name="gui_device_fields",
    ),
    "device.cancel_operation": MethodSpec(
        5.0,
        "Request cancellation of the named device's in-flight operation. Returns "
        "{ok: true, cancelled: true}. Note: only a device APPLY (setup ramp) has a "
        "cancellation point; a connect/disconnect has none and cannot be "
        "cancelled (it raises PRECONDITION_FAILED).",
        (_str("name", "Device name"),),
        tool_name="gui_device_cancel",
    ),
    "device.active_operations": MethodSpec(
        5.0,
        "List EVERY in-flight device operation (connect / disconnect / apply run "
        "concurrently): {operations: [{handle, device_name, kind, type_name, "
        "address, status, error}, ...]} (empty list if none), sorted by device "
        "name. 'handle' is the operation handle for gui_op_poll / gui_op_wait; "
        "'kind' is device_connect / device_disconnect / device_setup. Use "
        "gui_op_poll(handle) / gui_op_wait(handle) to track each one.",
        tool_name="gui_device_list_operations",
    ),
    "device.list": MethodSpec(
        5.0,
        "List registered devices with their current lifecycle status: "
        "{devices: [{name, type_name, status}, ...]} where status is one of "
        "memory_only | connecting | connected | disconnecting | setting_up "
        "(same status vocabulary as gui_device_snapshot and "
        "gui_device_list_operations). 'memory_only' means remembered but not "
        "live (no driver).",
    ),
    "device.snapshot": MethodSpec(
        5.0,
        "Read one device's full cached snapshot — the richest single-device read: "
        "{snapshot: {name, type_name, address, status, error, info}} where 'info' "
        "is the live device parameter dict (or null when not connected) and "
        "'status' uses the same vocabulary as gui_device_list. An unknown device "
        "name raises INVALID_PARAMS.",
        (_str("name", "Device name"),),
    ),
}
