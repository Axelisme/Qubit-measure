"""Qt-free wire-method contract table — the single source of truth for every
remote method's parameter schema, timeout and description.

This module is intentionally free of Qt and of any handler/Controller code so
that the lightweight ``mcp_server`` bridge can import it (to generate MCP tool
schemas) without pulling in the Qt-bound service layer. ``dispatch`` binds a
synchronous handler to each spec here to form its runtime registry.
"""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

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


def _bool_default(name: str, default: bool, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.BOOLEAN, required=False, default=default, description=desc
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
    "tab.get_cfg": MethodSpec(
        5.0,
        "Read tab cfg as the raw tagged form (not a set_field path source — use "
        "tab.list_paths for editable dotted paths).",
        (_str("tab_id"),),
    ),
    "tab.list_paths": MethodSpec(
        5.0,
        "List the settable cfg dotted paths — the path source for "
        "editor.set_field (edit a path with editor.set_field on the tab's "
        "editor_id, from tab.snapshot). kind ∈ scalar / sweep_edge "
        "/ moduleref_key / deviceref. A sweep_edge (a sweep's start/stop/expts/"
        "step) accepts ONLY a number/int — NOT an eval/ref; an adapter's default "
        "eval edge cannot be overwritten through this path, pass a numeric value "
        "instead. (A scalar leaf, by contrast, also accepts an eval reference.) "
        "'under' restricts to the sub-tree at that "
        "dotted path (e.g. 'modules.readout'); omit for the whole cfg. "
        "'verbosity' shapes each entry: 'compact' (default) = {path, kind, "
        "choices?}; 'full' adds current value + type; 'paths' = a bare list of "
        "path strings (smallest).",
        (
            _str("tab_id"),
            _str_opt("under", "Restrict to the sub-tree at this dotted path"),
            _str_default("verbosity", "compact", "compact (default) / full / paths"),
        ),
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
    "analyze.cancel": MethodSpec(
        5.0,
        "Cancel the tab's in-flight (interactive) analyze: settle its handle as "
        "cancelled and clear is_analyzing so the tab can then be closed. This is "
        "the agent-side counterpart of the GUI 'Done' button for an interactive "
        "picker — interactive analyze is a separate operation from run, so "
        "run.cancel does NOT settle it. Returns {ok, cancelled}: ok is always "
        "true (the call succeeded); cancelled is true when an interactive analyze "
        "was settled, or false (a graceful no-op) when none was in flight.",
        (_str("tab_id"),),
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
    "save.post_image": MethodSpec(
        30.0,
        "Save the post-analysis figure image file (the post sub-tab's own Save "
        "Image). Mirrors save.image but targets the tab's post-analysis figure; "
        "requires a post-analysis result.",
        (
            _str("tab_id"),
            _str_opt("image_path", "Override image path"),
            _expected_versions(),
        ),
    ),
    "save.result": MethodSpec(
        30.0,
        "Save the result's data and image",
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
            _str_opt(
                "bind_device",
                "Connected flux device to bind: its current value/unit name the "
                "context (whitelist: FakeDevice->none, YOKOGS200->A). Omit for an "
                "unbound context (unit=none, no value).",
            ),
            _str_opt("clone_from", "Label of an existing context to clone ml/md from"),
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
        "Read the connected SoC's hardware summary (QICK soccfg): a compact "
        "human-readable 'description' table (per-channel generator/readout type, "
        "converter port, sample rate, max pulse/buffer length) plus a structured "
        "'cfg' carrying the full detail, and 'is_mock'. Requires a connected SoC.",
    ),
    "project.info": MethodSpec(
        5.0,
        "Read the applied project identity: chip_name / qub_name / res_name plus "
        "the resolved result_dir and database_path. Fast-fails with "
        "precondition_failed (no_project) when no project is applied yet.",
    ),
    # Resource version table (optimistic-concurrency guard baseline). Full
    # snapshot the mcp layer reads to track last-seen versions; the version
    # integers are mcp/RPC bookkeeping and are never surfaced to the agent.
    "resources.versions": MethodSpec(5.0, "Snapshot of all resource versions"),
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
        "result_dir and database_path. Omit them to use the default per-qubit "
        "roots (<cwd>/result/<chip>/<qub> and <cwd>/Database/<chip>/<qub>, the "
        "same the setup dialog pre-fills) — the project is runnable either way. "
        "Pass explicit paths to override.",
        (
            _str("chip_name"),
            _str("qub_name"),
            _str("res_name"),
            _str_opt(
                "result_dir",
                "Result directory; omit → default <cwd>/result/<chip>/<qub>",
            ),
            _str_opt(
                "database_path",
                "Database path; omit → default <cwd>/Database/<chip>/<qub>",
            ),
        ),
    ),
    # Device
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
    "device.active_operations": MethodSpec(
        5.0,
        "List EVERY in-flight device operation (connect / disconnect / setup run "
        "concurrently): {active_operations: [{device_name, kind, name, type_name, "
        "address, status, error}, ...]} (empty list if none), sorted by device "
        "name. 'kind' is device_connect / device_disconnect / device_setup. Use "
        "gui_device_poll(name) / gui_device_wait_operation(name) per device to "
        "track each one.",
    ),
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
    "operation.progress": MethodSpec(
        5.0,
        "Read one operation's live progress bars by operation_id (run or device "
        "setup alike). active=false/bars=[] when idle; each bar has token, format "
        "(human-readable e.g. 'Rounds 23/100 [0:25<1:15]'), maximum/value "
        "(Qt-scaled), percent (0-100, null when total unknown), raw n/total. "
        "Internal: agents read progress folded into the gui_*_poll reply.",
        (_int("operation_id", "Operation handle returned by the start op"),),
    ),
    "device.list": MethodSpec(5.0, "List registered devices"),
    "device.snapshot": MethodSpec(
        5.0, "Read one device cached snapshot", (_str("name", "Device name"),)
    ),
    "adapter.list": MethodSpec(5.0, "List available adapters"),
    "adapter.cfg_spec": MethodSpec(
        5.0,
        "List an adapter's static cfg paths (path/kind/type/choices/label) "
        "without building a tab — the shape skeleton. ModuleRef/WaveformRef "
        "nodes list ONLY their '.ref' selector + allowed choices, NOT any "
        "variant's inner fields (which variant is the live default is a "
        "context-dependent value-layer decision a static spec can't know). To "
        "read a chosen variant's fields, build a tab and use tab.list_paths "
        "(live, with under/verbosity).",
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
    "app.shutdown": MethodSpec(
        5.0,
        "Gracefully close the GUI: runs the normal window-close path (persist "
        "session, disconnect devices, cleanup) — the same as a user closing the "
        "window. Returns immediately; the close happens just after. No OS kill. "
        "Prefer this over gui_stop's force path to stop a GUI cleanly.",
    ),
    "dialog.screenshot": MethodSpec(
        10.0,
        "Capture a named dialog as base64 PNG",
        (_str("name", "Dialog name"),),
    ),
    "view.snapshot": MethodSpec(5.0, "Capture view state summary"),
    "tab.get_current_figure": MethodSpec(
        10.0,
        "Get the tab's current figure (run 2D map, or analysis fit) as PNG. The "
        "PNG is rendered at a fixed small geometry (token-light), independent of "
        "the GUI window size; the live figure is never permanently resized.",
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
    "predictor.set_model_params": MethodSpec(
        10.0,
        "Build+install a FluxoniumPredictor directly from typed model params "
        "(no params.json). EJ/EC/EL are the Fluxonium energies in GHz "
        "(e.g. 4:1:1); flux_half/flux_period are the value->flux affine anchors "
        "in device-value units; flux_bias is the bias correction. Replaces any "
        "currently loaded predictor.",
        (
            _num("EJ", "Josephson energy E_J (GHz)"),
            _num("EC", "Charging energy E_C (GHz)"),
            _num("EL", "Inductive energy E_L (GHz)"),
            _num("flux_half", "Half-flux (Phi0/2) value->flux anchor (device units)"),
            _num("flux_period", "Flux period (device units); must be non-zero"),
            _num_default("flux_bias", 0.0, "Flux bias correction (device units)"),
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
        "Start analyzing the tab's run result. Analyze runs on a worker thread "
        "and returns an operation_id (like run/connect/device); the mcp "
        "gui_analyze tool awaits it so the agent sees one synchronous call. "
        "'updates' optionally overrides analyze params (see "
        "gui_adapter_analyze_spec). Makes the tab busy while it runs; a "
        "concurrent save/edit returns precondition_failed until it settles. "
        "Read the fit summary with gui_tab_get_analyze_result.",
        (_str("tab_id"), _obj_default("updates", "Analyze param updates")),
        tool_name="gui_analyze",
    ),
    # Post-analysis (second analysis layer; mirrors the analyze trio above). It
    # runs on top of an existing primary analyze result, so every entry gates
    # server-side on that result existing (no version guard — same as
    # analyze.start, which gates on has_run_result rather than expected_versions).
    "tab.get_post_analyze_result": MethodSpec(
        5.0, "Read tab post-analysis result scalar summary", (_str("tab_id"),)
    ),
    "tab.get_post_analyze_params": MethodSpec(
        5.0, "Read current post-analysis params", (_str("tab_id"),)
    ),
    "post_analyze.start": MethodSpec(
        30.0,
        "Start the second-layer (post) analysis on the tab's PRIMARY analyze "
        "result. Runs on a worker thread and returns an operation_id (like "
        "analyze.start); the mcp gui_post_analyze tool awaits it so the agent "
        "sees one synchronous call. Fast-fails with precondition_failed when the "
        "tab has no primary analyze result to build on. 'updates' optionally "
        "overrides post params (see gui_tab_get_post_analyze_params). Read the "
        "fit summary with gui_tab_get_post_analyze_result.",
        (_str("tab_id"), _obj_default("updates", "Post-analysis param updates")),
        tool_name="gui_post_analyze",
    ),
    "tab.get_cfg_summary": MethodSpec(
        5.0,
        "Read the tab cfg as a nested values view (read-only). NOT a set_field "
        "path source (use tab.list_paths for editable dotted paths). Mirrors the "
        "cfg tree and KEEPS info lowering would drop — EvalValue fields stay as "
        "their "
        "expression string (e.g. 'r_f - 0.1', not the evaluated number) and each "
        "module/waveform ref node is shown as {chosen, value:{...}}. That ref "
        "wrapper means its key shape is NOT the editable path shape: a field reads "
        "as modules.readout.value.pulse_cfg.freq here but edits (editor.set_field) "
        "use the no-'value' form modules.readout.pulse_cfg.freq. A sweep's derived "
        "'step' reads null here when an edge is an unresolved eval expression (the "
        "stored step cannot be trusted against an expr edge; it is recomputed only "
        "for numeric edges). Use this to read values/expressions; for editable "
        "paths use list_paths.",
        (_str("tab_id"),),
    ),
    # Writeback workflow — a persistent draft computed once at analyze time.
    "writeback.preview": MethodSpec(
        5.0,
        "List the tab's persistent writeback draft. Each item: id "
        "(<kind>-<n>, kind∈md|ml|wf), target_name (apply destination, editable), "
        "kind (metadict|module|waveform), description, selected; metadict adds "
        "proposed_value; module/waveform add editor_id + has_edit_schema. A "
        'complex metadict proposed_value is carried as {"__complex__": [re, im]} '
        "(JSON has no complex). Edit a module/waveform's cfg via editor.* on its "
        "editor_id; the user's Edit dialog renders the same model (WYSIWYG).",
        (_str("tab_id"),),
    ),
    "writeback.set": MethodSpec(
        5.0,
        "Edit a persistent writeback item by id: selected? / target_name? / "
        "proposed_value? (metadict only). A complex proposed_value is passed as "
        '{"__complex__": [re, im]} (the same shape preview emits); it applies as '
        "a Python complex. Module/waveform cfg edits go through "
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
        "commit). NOTE: the eval form is accepted ONLY on a scalar leaf — a "
        "sweep_edge (a sweep's start/stop/expts/step) accepts ONLY a number/int, "
        "never an eval/ref; an adapter's default eval edge cannot be overwritten "
        "this way, pass a numeric value instead. "
        "Returns {valid, removed, added} — does NOT echo cfg content "
        "(that would force a lowering pass that eagerly evaluates EvalValue). "
        "'valid' is whether the whole draft is currently valid; 'removed'/'added' "
        "list settable paths a ModuleRef key switch ('<path>.ref') dropped/"
        "created so you need not re-list after a variant switch. To read cfg use "
        "tab.list_paths / editor.get (which have under/verbosity).",
        (
            _str("editor_id"),
            _str("path", "Dotted field path"),
            _json("value", "JSON scalar or {__kind:eval, expr}"),
        ),
    ),
    "editor.get": MethodSpec(
        5.0,
        "List the settable paths of an editing session. 'under' restricts to the "
        "sub-tree at that dotted path; omit for the whole draft. 'verbosity': "
        "'compact' (default) = {path, kind, choices?}; 'full' adds value + type; "
        "'paths' = a bare list of path strings.",
        (
            _str("editor_id"),
            _str_opt("under", "Restrict to the sub-tree at this dotted path"),
            _str_default("verbosity", "compact", "compact (default) / full / paths"),
        ),
    ),
    "editor.commit": MethodSpec(
        10.0,
        "Save the current draft as a ModuleLibrary module/waveform: lower the "
        "session (eval expressions resolved against MetaDict to concrete numbers) "
        "and register it into the ModuleLibrary under 'name'. This is NOT 'apply "
        "a tab cfg edit' — tab cfg set_field edits are already live (WYSIWYG); "
        "this persists the draft as a named ml entry. On success the session is "
        "destroyed; on validation failure it is kept so you can fix and retry.",
        (
            _str("editor_id"),
            _str("name", "ml entry name to register under"),
            _expected_versions(),
        ),
        tool_name="gui_editor_save_as_module",
    ),
    "editor.discard": MethodSpec(
        5.0,
        "Discard an editing session without writing to the ModuleLibrary.",
        (_str("editor_id"),),
    ),
    # Notify prompt — agent-initiated user question (Stage 4b, ADR-0025).
    # Two-RPC design: notify.open mints the token + opens the dialog on the
    # main thread; notify.await blocks the off-main worker until the user replies,
    # dismisses, or the dialog's QTimer fires. Both are excluded from
    # auto-generation via _NON_GENERATED_METHODS in the MCP server so that only
    # the hand-written gui_notify_user tool is exposed to the agent.
    "notify.open": MethodSpec(
        30.0,
        "Open a non-modal agent-prompt dialog on the main thread. Returns {token}.",
        (
            _str("message", "Message to display to the user"),
            _num_default("timeout", 600.0, "Prompt auto-close timeout in seconds"),
        ),
    ),
    "notify.await": MethodSpec(
        # Nominal only — off_main_thread handlers bypass the main-thread budget
        # watchdog (control_service), so the real bound is the caller's `timeout`
        # param. Kept >= the default consumer backstop (600 + slack) for clarity.
        615.0,
        "Block the IO worker until the notify prompt settles. Returns "
        "{reason, reply?}. reason in {'reply', 'dismiss', 'timeout'}.",
        (
            _int("token", "Token returned by notify.open"),
            _num_default("timeout", 600.0, "Consumer backstop timeout in seconds"),
        ),
        off_main_thread=True,
    ),
}
