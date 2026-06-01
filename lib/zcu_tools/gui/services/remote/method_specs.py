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

    ``off_main_thread`` marks a blocking handler that must NOT be marshalled
    onto the Qt main thread — it runs on the IO worker thread instead. Required
    for handlers that block waiting on a worker-thread completion (e.g.
    ``operation.await``): marshalling them onto the main thread would deadlock
    (the handler occupies the event loop that must dispatch the very signal it
    awaits). Off-main handlers must only do thread-safe waiting and must not
    touch main-thread-owned state, the stale guard, or the origin scope.
    """

    timeout_seconds: float
    description: str
    params: tuple[ParamSpec, ...] = ()
    tool_name: str = ""
    off_main_thread: bool = False


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


def _expected_versions() -> ParamSpec:
    """Wire-only optimistic-concurrency guard param (mcp-filled, MCP-hidden).

    The mcp layer attaches the resource->version map this op depends on; the
    server compares it atomically. Hidden from the agent-facing MCP schema.
    """
    return ParamSpec(
        "expected_versions",
        JsonType.OBJECT,
        required=False,
        default={},
        description="Resource versions the caller depends on (mcp bookkeeping)",
        mcp_hidden=True,
    )


def _int(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.INTEGER, required=True, description=desc)


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


def _int_opt(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.INTEGER, required=False, description=desc)


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
        "(when applicable) choices. Edit a path with editor.set_field on the "
        "tab's editor_id (from tab.snapshot). kind ∈ scalar / sweep_edge / "
        "moduleref_key / deviceref.",
        (_str("tab_id"),),
    ),
    "tab.update_cfg": MethodSpec(
        10.0,
        "Replace tab cfg from a full tagged form (missing keys reset to spec "
        "defaults, not preserved; use editor.set_field for single-field edits)",
        (_str("tab_id"), _obj("raw", "Full tagged cfg form")),
    ),
    # Run
    "run.start": MethodSpec(
        5.0, "Start a run (fire-and-forget)", (_str("tab_id"), _expected_versions())
    ),
    "run.cancel": MethodSpec(5.0, "Cancel current run"),
    "run.running_tab": MethodSpec(5.0, "Current running tab"),
    "run.progress": MethodSpec(
        5.0,
        "Read current run progress bars. Returns active=false, bars=[] when idle. "
        "When active=true, each bar has: token (stable id), format (human-readable "
        "string e.g. 'Rounds 23/100 [0:25<1:15]'), maximum (Qt-scaled total; 0 if "
        "unknown), value (Qt-scaled position), percent (0-100 convenience, null "
        "when total unknown), and raw n/total (precise counts; total null when "
        "unknown). Prefer the auto-subscribed 'run_finished' event (via "
        "gui_events_poll) to detect completion rather than polling this.",
    ),
    # Save
    "save.data": MethodSpec(
        30.0,
        "Save data file",
        (
            _str("tab_id"),
            _str_opt("data_path", "Override data path"),
            _comment(),
            _expected_versions(),
        ),
    ),
    "save.image": MethodSpec(
        30.0,
        "Save image file",
        (
            _str("tab_id"),
            _str_opt("image_path", "Override image path"),
            _expected_versions(),
        ),
    ),
    "save.both": MethodSpec(
        30.0,
        "Save data and image",
        (
            _str("tab_id"),
            _str_opt("data_path", "Override data path"),
            _str_opt("image_path", "Override image path"),
            _comment(),
            _expected_versions(),
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
    "context.del_ml_module": MethodSpec(
        5.0, "Delete one ModuleLibrary module", (_str("name", "Module name"),)
    ),
    "context.del_ml_waveform": MethodSpec(
        5.0, "Delete one ModuleLibrary waveform", (_str("name", "Waveform name"),)
    ),
    "context.rename_ml_module": MethodSpec(
        5.0,
        "Rename a ModuleLibrary module old→new (clash fails fast). cfg refs to "
        "'old' degrade to inline Custom (value kept); to re-link, edit them.",
        (_str("old", "Current module name"), _str("new", "New module name")),
    ),
    "context.rename_ml_waveform": MethodSpec(
        5.0,
        "Rename a ModuleLibrary waveform old→new (clash fails fast).",
        (_str("old", "Current waveform name"), _str("new", "New waveform name")),
    ),
    "ml.list_roles": MethodSpec(
        5.0,
        "List experiment-role templates for ml.create_from_role. Returns "
        "{roles: [{role_id, label, item_kind}]}. Each role seeds a blank module/"
        "waveform with md-linked defaults (e.g. 'res_probe', 'bath_reset').",
    ),
    "ml.create_from_role": MethodSpec(
        10.0,
        "Create a blank ModuleLibrary module/waveform from a named role "
        "(ml.list_roles) and register it under 'name'. One-shot: seeds the role's "
        "md-linked defaults (lowered to the md's current values) — it does NOT "
        "open an editing session. To then change the entry use "
        "editor.open(from_name=name).",
        (
            _str("item_kind", "'module' or 'waveform'"),
            _str("role_id", "role id from ml.list_roles"),
            _str("name", "new ml entry name"),
        ),
    ),
    # State (fan-out at MCP; individual booleans here)
    "state.has_project": MethodSpec(5.0, ""),
    "state.has_context": MethodSpec(5.0, ""),
    "state.has_active_context": MethodSpec(5.0, ""),
    "state.has_soc": MethodSpec(5.0, ""),
    "soc.info": MethodSpec(
        5.0,
        "Read the connected SoC's hardware summary (QICK soccfg): a "
        "human-readable 'description' (DAC/ADC channels, sample rates, freq "
        "ranges, tiles) plus a structured 'cfg' and 'is_mock'. Requires a "
        "connected SoC.",
    ),
    # Resource version table (optimistic-concurrency guard baseline). Full
    # snapshot the mcp layer reads to track last-seen versions; the version
    # integers are mcp/RPC bookkeeping and are never surfaced to the agent.
    "resources.versions": MethodSpec(5.0, "Snapshot of all resource versions"),
    # Session
    "session.persist": MethodSpec(10.0, "Persist tab session"),
    "session.restore": MethodSpec(10.0, "Restore tab session"),
    # Connection / startup
    "connect.start": MethodSpec(
        30.0,
        "Connect the SoC. kind='mock' for an offline mock board, or "
        "kind='remote' with ip + port for a real board (ip/port required only "
        "when kind='remote').",
        (
            _str("kind", "'mock' or 'remote'"),
            _str_opt("ip", "Board IP (required when kind='remote')"),
            _int_opt("port", "Board port (required when kind='remote')"),
        ),
    ),
    "startup.apply": MethodSpec(
        30.0,
        "Set the project: chip / qubit / resonator names, plus optional "
        "result_dir and database_path. Omit result_dir to leave the context in "
        "DRAFT (editable but not runnable); set it to enable runs/saves.",
        (
            _str("chip_name"),
            _str("qub_name"),
            _str("res_name"),
            _str_opt("result_dir", "Result directory; omit → DRAFT context"),
            _str_opt("database_path", "Database path; optional"),
        ),
    ),
    # Device
    "device.connect": MethodSpec(30.0, "Connect device"),
    "device.disconnect": MethodSpec(30.0, "Disconnect device"),
    "device.reconnect": MethodSpec(
        30.0, "Reconnect device", (_str("name", "Device name"),)
    ),
    "device.forget": MethodSpec(
        5.0, "Forget memory-only device", (_str("name", "Device name"),)
    ),
    "device.setup": MethodSpec(
        30.0,
        "Setup device",
        (_str("name", "Device name"), _obj("updates", "Field updates")),
    ),
    "device.setup_spec": MethodSpec(
        5.0,
        "List the fields settable via device.setup's 'updates' for a connected "
        "device: each field's name, type, choices (for enum/Literal fields like "
        "output/mode), current value, and whether it is settable (the protected "
        "type/address are reported settable=false). The device must be connected.",
        (_str("name", "Device name"),),
    ),
    "device.cancel_operation": MethodSpec(
        5.0, "Cancel active device setup", (_str("name", "Device name"),)
    ),
    "device.active_setup": MethodSpec(
        5.0, "Which device (if any) is currently setting up: {device_name} or null"
    ),
    "device.setup_progress": MethodSpec(
        5.0,
        "Read the active device setup's progress bars — same shape as run.progress "
        "(active, bars[token/format/maximum/value/percent/n/total]). Prefer the "
        "auto-subscribed device_setup_finished event to detect completion.",
    ),
    "device.active_operation": MethodSpec(5.0, "Read active device operation"),
    # Async operation handle: block until an operation (device.connect /
    # device.disconnect / device.setup / run.start / connect.start, identified by
    # the operation_id they return) settles. mcp bookkeeping only — agents drive
    # it via semantic wait tools, never raw.
    "operation.await": MethodSpec(
        130.0,
        "Block until an async operation settles (by operation_id)",
        (
            _int("operation_id", "Operation handle returned by the start op"),
            _num_default("timeout", 120.0, "Seconds to wait"),
        ),
        off_main_thread=True,
    ),
    "device.list": MethodSpec(5.0, "List registered devices"),
    "device.snapshot": MethodSpec(
        5.0, "Read one device cached snapshot", (_str("name", "Device name"),)
    ),
    "adapter.list": MethodSpec(5.0, "List available adapters"),
    "adapter.cfg_spec": MethodSpec(
        5.0,
        "List an adapter's static cfg paths (path/kind/type/choices/label) "
        "without building a tab. ModuleRef/WaveformRef expose every allowed "
        "option's sub-fields under value.<label>. Use tab.list_paths for a "
        "live tab's current values.",
        (_str("adapter_name", "Adapter to introspect"),),
    ),
    "adapter.analyze_spec": MethodSpec(
        5.0,
        "Describe an adapter's analyze params (name/type/choices/label/default). "
        "Returns empty params when the adapter has no analysis.",
        (_str("adapter_name", "Adapter to introspect"),),
    ),
    "adapter.guide": MethodSpec(
        5.0,
        "Read an adapter's human-facing orientation guide BEFORE running it: "
        "prose (not a contract) on {behavior, expects_md, expects_ml, "
        "typical_writeback, recommended} — what the experiment measures, what it "
        "assumes is already in the MetaDict/ModuleLibrary, what a run tends to "
        "write back, and recommended analysis settings. How you actually use it "
        "is your call. Empty fields mean the adapter has no guide written yet.",
        (_str("adapter_name", "Adapter to introspect"),),
    ),
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
    # Writeback workflow — a persistent draft computed once at analyze time.
    "writeback.preview": MethodSpec(
        5.0,
        "List the tab's persistent writeback draft. Each item: id "
        "(<kind>-<n>, kind∈md|ml|wf), target_name (apply destination, editable), "
        "kind (metadict|module|waveform), description, selected; metadict adds "
        "proposed_value; module/waveform add editor_id + has_edit_schema. Edit a "
        "module/waveform's cfg via editor.* on its editor_id; the user's Edit "
        "dialog renders the same model (WYSIWYG).",
        (_str("tab_id"),),
    ),
    "writeback.set": MethodSpec(
        5.0,
        "Edit a persistent writeback item by id: selected? / target_name? / "
        "proposed_value? (metadict only). Module/waveform cfg edits go through "
        "editor.set_field on the item's editor_id, not here.",
        (
            _str("tab_id"),
            _str("id", "writeback item session id (<kind>-<n>)"),
            # Boolean (not JSON): a JSON schema of {type: boolean} makes the
            # client send a real boolean. Declared as JSON, the client may send
            # the string "false", which ``bool("false")`` wrongly reads as True.
            ParamSpec("selected", JsonType.BOOLEAN, required=False),
            _str_opt("target_name", "new apply destination name"),
            ParamSpec("proposed_value", JsonType.JSON, required=False),
            _expected_versions(),
        ),
    ),
    "writeback.apply": MethodSpec(
        10.0,
        "Apply the tab's persistent writeback draft as-is (edit it first via "
        "writeback.set / editor.*). Applies items currently selected. Returns "
        "applied_ids.",
        (_str("tab_id"), _expected_versions()),
    ),
    # CfgEditor sessions — headless, stateful ml-entry editing for the agent.
    "editor.open": MethodSpec(
        5.0,
        "Open a stateful editing session over an EXISTING ModuleLibrary "
        "module/waveform (by 'from_name'). To create a new blank/shaped entry, "
        "use ml.create_from_role (e.g. role_id='pulse:blank' or a named role) "
        "then editor.open(from_name=name) to edit it. item_kind is 'module' or "
        "'waveform'. Returns {editor_id, paths} (paths = settable dotted paths "
        "with current values, same shape as tab.list_paths).",
        (
            _str("item_kind", "'module' or 'waveform'"),
            _str("from_name", "Existing ml entry name to load for editing"),
        ),
    ),
    "editor.set_field": MethodSpec(
        5.0,
        "Set one field in an editing session. 'path' is a dotted path from "
        "editor.open/get (ModuleRef sub-fields descend directly, no 'value' "
        "segment); 'value' is a JSON scalar, or an md-reference expression as "
        '{"__kind":"eval","expr":"r_f - 0.1"} (resolved against MetaDict at '
        "commit). Returns {paths, removed, added, valid}: 'paths' is the sub-tree "
        "rooted at the changed path; 'removed'/'added' list settable paths that a "
        "ModuleRef key switch ('<path>.ref') dropped/created, so you need not "
        "re-list the tab; 'valid' is whether the whole draft is currently valid.",
        (
            _str("editor_id"),
            _str("path", "Dotted field path"),
            _json("value", "JSON scalar or {__kind:eval, expr}"),
        ),
    ),
    "editor.get": MethodSpec(
        5.0,
        "List all settable paths + current values of an editing session.",
        (_str("editor_id"),),
    ),
    "editor.commit": MethodSpec(
        10.0,
        "Lower the session (eval expressions resolved against MetaDict to concrete "
        "numbers) and register it into the ModuleLibrary under 'name'. On success "
        "the session is destroyed; on validation failure it is kept so you can fix "
        "and retry.",
        (
            _str("editor_id"),
            _str("name", "ml entry name to register under"),
            _expected_versions(),
        ),
    ),
    "editor.discard": MethodSpec(
        5.0,
        "Discard an editing session without writing to the ModuleLibrary.",
        (_str("editor_id"),),
    ),
}
