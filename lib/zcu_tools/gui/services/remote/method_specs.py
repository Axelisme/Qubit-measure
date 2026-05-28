"""Qt-free wire-method contract table — the single source of truth for every
remote method's parameter schema, timeout and description.

This module is intentionally free of Qt and of any handler/Controller code so
that the lightweight ``mcp_server`` bridge can import it (to generate MCP tool
schemas) without pulling in the Qt-bound service layer. ``dispatch`` binds a
synchronous handler to each spec here to form its runtime registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .param_spec import JsonType, ParamSpec


@dataclass(frozen=True)
class MethodSpec:
    """Contract for one wire method, independent of its handler.

    ``timeout_seconds`` is the main-thread handler budget. ``params`` is the
    parameter contract used both for runtime validation (dispatch/service) and
    MCP ``inputSchema`` generation (mcp_server). ``tool_name`` overrides the
    derived ``gui_<method>`` MCP tool name when non-empty.
    """

    timeout_seconds: float
    description: str
    params: tuple[ParamSpec, ...] = ()
    tool_name: str = ""


# ---------------------------------------------------------------------------
# ParamSpec factory shorthands — keep the table readable.
# ---------------------------------------------------------------------------


def _str(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.STRING, required=True, description=desc)


def _str_opt(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.STRING, required=False, description=desc)


def _str_default(name: str, default: str, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.STRING, required=False, default=default, description=desc
    )


def _obj(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.OBJECT, required=True, description=desc)


def _obj_default(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.OBJECT, required=False, default={}, description=desc
    )


def _json(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.JSON, required=True, description=desc)


def _num(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.NUMBER, required=True, description=desc)


def _num_opt(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.NUMBER, required=False, description=desc)


def _num_default(name: str, default: float, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.NUMBER, required=False, default=default, description=desc
    )


def _int_default(name: str, default: int, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.INTEGER, required=False, default=default, description=desc
    )


def _comment() -> ParamSpec:
    return ParamSpec(
        "comment", JsonType.STRING, required=False, default="", description="Comment"
    )


# ---------------------------------------------------------------------------
# The contract table. Keys are dotted wire-method names.
# ---------------------------------------------------------------------------


METHOD_SPECS: dict[str, MethodSpec] = {
    # Tab
    "tab.new": MethodSpec(
        10.0, "Create a new tab", (_str("adapter_name", "Adapter to instantiate"),)
    ),
    "tab.close": MethodSpec(5.0, "Close a tab", (_str("tab_id"),)),
    "tab.set_active": MethodSpec(5.0, "Activate a tab", (_str("tab_id"),)),
    "tab.list": MethodSpec(5.0, "List tabs"),
    "tab.snapshot": MethodSpec(
        5.0, "Tab summary", (_str_opt("tab_id", "Tab to inspect; omit for all tabs"),)
    ),
    "tab.get_cfg": MethodSpec(5.0, "Read tab cfg raw", (_str("tab_id"),)),
    "tab.list_paths": MethodSpec(
        5.0,
        "List every settable cfg dotted path with its current value, type and "
        "(when applicable) choices. Each path is guaranteed usable with "
        "cfg.set_field. kind ∈ scalar / sweep_edge / moduleref_key / deviceref.",
        (_str("tab_id"),),
    ),
    "tab.update_cfg": MethodSpec(
        10.0,
        "Replace tab cfg raw",
        (_str("tab_id"), _obj("raw", "Full tagged cfg form")),
    ),
    "cfg.set_field": MethodSpec(
        5.0,
        "Set a single cfg field by dotted path",
        (
            _str("tab_id"),
            _str("path", "Dotted field path"),
            _json("value", "New field value"),
        ),
    ),
    # Run
    "run.start": MethodSpec(5.0, "Start a run (fire-and-forget)", (_str("tab_id"),)),
    "run.cancel": MethodSpec(5.0, "Cancel current run"),
    "run.running_tab": MethodSpec(5.0, "Current running tab"),
    "run.progress": MethodSpec(
        5.0,
        "Read current run progress bars. Returns active=false, bars=[] when idle. "
        "When active=true, each bar has: token (stable id), format (human-readable "
        "string e.g. 'Rounds 23/100 [0:25<1:15]'), maximum (total steps; 0 if "
        "unknown), value (current step), percent (0-100 convenience, null when "
        "total unknown). Prefer subscribing to 'run_lock_changed' via "
        "gui_events_subscribe to detect completion rather than polling this.",
    ),
    # Save
    "save.data": MethodSpec(
        30.0,
        "Save data file",
        (_str("tab_id"), _str_opt("data_path", "Override data path"), _comment()),
    ),
    "save.image": MethodSpec(
        30.0,
        "Save image file",
        (_str("tab_id"), _str_opt("image_path", "Override image path")),
    ),
    "save.both": MethodSpec(
        30.0,
        "Save data and image",
        (
            _str("tab_id"),
            _str_opt("data_path", "Override data path"),
            _str_opt("image_path", "Override image path"),
            _comment(),
        ),
    ),
    "save.set_paths": MethodSpec(
        5.0,
        "Set tab save path overrides",
        (_str("tab_id"), _str("data_path"), _str("image_path")),
    ),
    # Context
    "context.use": MethodSpec(
        5.0, "Switch context", (_str("label", "Context label to switch to"),)
    ),
    "context.new": MethodSpec(
        10.0,
        "Create new context",
        (
            _num_opt("value", "Flux value"),
            _str_default("unit", "A", "Flux unit"),
            ParamSpec(
                "clone_from_current",
                JsonType.BOOLEAN,
                required=False,
                default=False,
                description="Clone current context",
            ),
        ),
    ),
    "context.labels": MethodSpec(5.0, "List context labels"),
    "context.active": MethodSpec(5.0, "Active context label"),
    "context.get_md": MethodSpec(5.0, "List MetaDict keys"),
    "context.get_md_attr": MethodSpec(
        5.0, "Read one MetaDict attribute", (_str("key", "MetaDict key"),)
    ),
    "context.get_ml": MethodSpec(5.0, "List ModuleLibrary module/waveform names"),
    "context.set_md_attr": MethodSpec(
        5.0,
        "Set one MetaDict attribute",
        (_str("key", "MetaDict key"), _json("value", "JSON-safe value")),
    ),
    "context.del_md_attr": MethodSpec(
        5.0, "Delete one MetaDict attribute", (_str("key", "MetaDict key"),)
    ),
    "context.set_ml_module": MethodSpec(
        10.0,
        "Set one ModuleLibrary module from raw dict",
        (_str("name", "Module name"), _obj("raw", "Module cfg dict")),
    ),
    "context.del_ml_module": MethodSpec(
        5.0, "Delete one ModuleLibrary module", (_str("name", "Module name"),)
    ),
    "context.set_ml_waveform": MethodSpec(
        10.0,
        "Set one ModuleLibrary waveform from raw dict",
        (_str("name", "Waveform name"), _obj("raw", "Waveform cfg dict")),
    ),
    "context.del_ml_waveform": MethodSpec(
        5.0, "Delete one ModuleLibrary waveform", (_str("name", "Waveform name"),)
    ),
    # State (fan-out at MCP; individual booleans here)
    "state.has_project": MethodSpec(5.0, ""),
    "state.has_context": MethodSpec(5.0, ""),
    "state.has_active_context": MethodSpec(5.0, ""),
    "state.has_soc": MethodSpec(5.0, ""),
    # Session
    "session.persist": MethodSpec(10.0, "Persist tab session"),
    "session.restore": MethodSpec(10.0, "Restore tab session"),
    # Connection / startup (multi-field coercion at the handler)
    "connect.start": MethodSpec(30.0, "Connect to SoC"),
    "startup.apply": MethodSpec(30.0, "Apply startup project"),
    # Device
    "device.connect": MethodSpec(30.0, "Connect device"),
    "device.disconnect": MethodSpec(30.0, "Disconnect device"),
    "device.reconnect": MethodSpec(
        30.0, "Reconnect device", (_str("name", "Device name"),)
    ),
    "device.forget": MethodSpec(
        5.0, "Forget memory-only device", (_str("name", "Device name"),)
    ),
    "device.set_value": MethodSpec(30.0, "Set device value"),
    "device.setup": MethodSpec(
        30.0,
        "Setup device",
        (_str("name", "Device name"), _obj("updates", "Field updates")),
    ),
    "device.cancel_operation": MethodSpec(
        5.0, "Cancel active device setup", (_str("name", "Device name"),)
    ),
    "device.active_setup": MethodSpec(5.0, "Read active device setup progress"),
    "device.active_operation": MethodSpec(5.0, "Read active device operation"),
    "device.wait_setup": MethodSpec(
        130.0,
        "Block until device setup completes",
        (
            _str("name", "Device name"),
            _num_default("timeout", 120.0, "Seconds to wait"),
        ),
    ),
    "device.list": MethodSpec(5.0, "List registered devices"),
    "device.snapshot": MethodSpec(
        5.0, "Read one device cached snapshot", (_str("name", "Device name"),)
    ),
    "adapter.list": MethodSpec(5.0, "List available adapters"),
    # Dialog / view
    "dialog.open": MethodSpec(
        10.0, "Open a named dialog", (_str("name", "Dialog name"),)
    ),
    "dialog.close": MethodSpec(
        5.0, "Close a named dialog", (_str("name", "Dialog name"),)
    ),
    "dialog.list_open": MethodSpec(5.0, "List open dialogs"),
    "dialog.screenshot": MethodSpec(
        10.0,
        "Capture a named dialog as base64 PNG",
        (_str("dialog_name", "Dialog name"),),
    ),
    "view.snapshot": MethodSpec(5.0, "Capture view state summary"),
    "view.screenshot": MethodSpec(
        10.0,
        "Capture window or tab as base64 PNG",
        (_str_opt("tab_id", "Tab to capture; omit for whole window"),),
    ),
    "tab.figure_screenshot": MethodSpec(
        10.0,
        "Capture tab figure area as PNG",
        (
            _str("tab_id"),
            _str_opt("out_path", "Write PNG here instead of returning base64"),
        ),
    ),
    # Predictor
    "predictor.load": MethodSpec(
        30.0,
        "Load FluxoniumPredictor",
        (
            _str("path", "Predictor file path"),
            _num_default("flux_bias", 0.0, "Flux bias"),
        ),
    ),
    "predictor.clear": MethodSpec(5.0, "Clear predictor"),
    "predictor.predict": MethodSpec(
        10.0,
        "Predict transition frequency",
        (
            _num("value", "Flux value"),
            _int_default("from_lvl", 0, "From level"),
            _int_default("to_lvl", 1, "To level"),
        ),
    ),
    "predictor.info": MethodSpec(5.0, "Get predictor info"),
    # Analyze
    "tab.get_analyze_result": MethodSpec(
        5.0, "Read tab analyze result scalar summary", (_str("tab_id"),)
    ),
    "tab.get_analyze_params": MethodSpec(
        5.0, "Read current analyze params", (_str("tab_id"),)
    ),
    "analyze.start": MethodSpec(
        30.0,
        "Start analyze (fire-and-forget)",
        (_str("tab_id"), _obj_default("updates", "Analyze param updates")),
    ),
    "tab.get_cfg_summary": MethodSpec(
        5.0, "Read tab cfg as clean scalar dict", (_str("tab_id"),)
    ),
}
