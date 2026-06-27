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
        10.0,
        "Create a new tab for the named adapter. Returns {tab_id}.",
        (_str("adapter_name", "Adapter to instantiate"),),
    ),
    "tab.close": MethodSpec(5.0, "Close a tab. Returns {ok: true}.", (_str("tab_id"),)),
    "tab.set_active": MethodSpec(
        5.0,
        "Activate a tab. VIEW-ONLY: this changes which tab the user sees, NOT your "
        "operation target (you always act on an explicit tab_id). Returns {ok: true}.",
        (_str("tab_id"),),
    ),
    "tab.list_all": MethodSpec(
        5.0,
        "List all open tabs. Returns {tabs, active_tab_id, running_tab_id}: tabs "
        "is a list of {tab_id, adapter_name, is_running} objects; active_tab_id is "
        "the tab the USER is focused on (a collaboration cue, NOT your operation "
        "target); running_tab_id is the tab currently running (or null when "
        "nothing is running).",
        tool_name="gui_tab_list",
    ),
    "tab.snapshot": MethodSpec(
        5.0, "Tab summary", (_str_opt("tab_id", "Tab to inspect; omit for all tabs"),)
    ),
    "tab.get_cfg": MethodSpec(
        5.0,
        "Read the tab's settable cfg as a NESTED tree of current values (the "
        "read-only view; edit a leaf with tab.set_cfg or editor.set_field on the "
        "tab's editor_id from tab.snapshot, using the leaf's dotted path). Node "
        "shape, distinguished by '$'-prefixed reserved keys: a SCALAR leaf is "
        "its bare current value (null = unset); an ENUM scalar leaf is "
        "{'$value': current, '$choices': [...]}; a SWEEP is a sub-tree of bare "
        "edges {start, stop, expts, step} (each edge accepts ONLY a number/int "
        "via tab.set_cfg — NOT an eval/ref); a REF node "
        "(module/waveform/device selector) is {'$ref': {'current': <chosen>, "
        "'options': [<names>]}, <chosen variant's settable sub-tree>} — only the "
        "CURRENTLY-CHOSEN variant is expanded; 'options' lists bare names while "
        "'current' may be a tagged internal key — switch by passing a bare "
        "'options' name to tab.set_cfg on the ref's dotted path. "
        "Any other dict is a plain section sub-tree (its keys are child fields). "
        "'prefix' (optional, dotted) returns just the sub-tree rooted at that "
        "node (a prefix at a sweep edge returns the whole sweep node); a prefix "
        "matching nothing returns {}.",
        (
            _str("tab_id"),
            _str_opt(
                "prefix",
                "Return only the sub-tree rooted at this dotted path "
                "(e.g. 'modules.readout'); omit for the whole cfg. No match → {}",
            ),
        ),
    ),
    "tab.set_cfg": MethodSpec(
        5.0,
        "Batch-set cfg fields on a tab in order (fail-fast, non-atomic). 'edits' "
        "is an ORDERED list of {path, value} objects. Apply ref-switch edits "
        "before dependent inner-path edits (a ref switch removes child paths). "
        "'value' is a JSON scalar, an md-reference eval tag "
        '{"__kind":"eval","expr":"r_f"}, or a registered value-source tag '
        '{"__kind":"value_ref","key":"device.active_flux.value","type":"float"}; '
        "value_ref is resolved immediately at set time and stored as a direct "
        "scalar. Discover keys with value.list / value.read. "
        "Returns {valid, removed, added} aggregated across the batch — the same "
        "shape as editor.set_field. A tab that is currently running is rejected "
        "(cancel the run first). Use tab.get_cfg to read the current tree.",
        (
            _str("tab_id"),
            _json("edits", "Ordered list of {path, value} edits"),
        ),
    ),
    # Run
    "tab.run_start": MethodSpec(
        5.0, "Start a run (fire-and-forget)", (_str("tab_id"), _expected_versions())
    ),
    "tab.load_data": MethodSpec(
        30.0,
        "Load a canonical result file into an already-open adapter tab. The tab "
        "then has a run result and can be analyzed without a SoC connection. "
        "First release does not backfill Config from cfg_snapshot.",
        (
            _str("tab_id"),
            _str("data_path", "Canonical HDF5 result file to load"),
            _expected_versions(),
        ),
    ),
    "tab.run_cancel": MethodSpec(
        5.0,
        "Request cancellation of the current run (op-specific cancel; there is no "
        "generic cancel — see ADR-0026 §8). Returns {ok, cancelled}: ok is always "
        "true (the call succeeded); cancelled is BEST-EFFORT — true when a live run "
        "was signalled to stop, false (a graceful no-op) when no run was in flight. "
        "It does NOT mean the worker has stopped: the run's true terminal "
        "('cancelled') is observed by gui_op_wait/gui_op_poll on the run handle.",
    ),
    "run.running_tab": MethodSpec(5.0, "Current running tab"),
    "analyze.cancel": MethodSpec(
        5.0,
        "Cancel the tab's in-flight (interactive) analyze: settle its handle as "
        "cancelled and clear is_analyzing so the tab can then be closed. This is "
        "the agent-side counterpart of the GUI 'Done' button for an interactive "
        "picker — interactive analyze is a separate operation from run, so "
        "gui_tab_run_cancel does NOT settle it. This cancel is op-specific (an "
        "interactive analyze needs View teardown that no generic handle cancel can "
        "do — ADR-0026 §8). Returns {ok, cancelled}: ok is always true (the call "
        "succeeded); cancelled is true when an interactive analyze was settled, or "
        "false (a graceful no-op) when none was in flight.",
        (_str("tab_id"),),
        tool_name="gui_tab_analyze_cancel",
    ),
    # Save (under tab.* namespace — Phase 170c)
    "tab.save_data": MethodSpec(
        30.0,
        "Save data file",
        (
            _str("tab_id"),
            _str_opt("data_path", "Override data path"),
            _comment(),
            _expected_versions(),
        ),
    ),
    "tab.save_image": MethodSpec(
        30.0,
        "Save image file",
        (
            _str("tab_id"),
            _str_opt("image_path", "Override image path"),
            _expected_versions(),
        ),
    ),
    "tab.save_post_image": MethodSpec(
        30.0,
        "Save the post-analysis figure image file (the post sub-tab's own Save "
        "Image). Mirrors tab.save_image but targets the tab's post-analysis figure; "
        "requires a post-analysis result.",
        (
            _str("tab_id"),
            _str_opt("image_path", "Override image path"),
            _expected_versions(),
        ),
    ),
    "tab.save_result": MethodSpec(
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
    "tab.save_set_paths": MethodSpec(
        5.0,
        "Set the tab's default save destinations (data + image). Echoes the "
        "applied {data_path, image_path}. Version-guarded on the tab's save_path: "
        "rejects with precondition_failed if a concurrent edit moved it.",
        (
            _str("tab_id"),
            _str("data_path"),
            _str("image_path"),
            _expected_versions(),
        ),
        tool_name="gui_tab_set_save_paths",
    ),
    # Context
    "context.use": MethodSpec(
        5.0,
        "Switch the active context to 'label'. Echoes {label, has_active_context}. "
        "An unknown label fails fast (invalid_params) with the available labels; no "
        "applied project fails with precondition_failed.",
        (_str("label", "Context label to switch to"),),
        tool_name="gui_context_switch",
    ),
    "context.new": MethodSpec(
        10.0,
        "Create a new context and make it active. Echoes {label, has_active_context} "
        "— the auto-derived label (the agent cannot name it directly).",
        (
            _str_opt(
                "bind_device",
                "Connected flux device to bind: its current value/unit name the "
                "context (whitelist: FakeDevice->none, YOKOGS200->A). Omit for an "
                "unbound context (unit=none, no value).",
            ),
            _str_opt("clone_from", "Label of an existing context to clone ml/md from"),
        ),
        tool_name="gui_context_create",
    ),
    # context.labels / context.active stay as wire methods (no generated tool):
    # gui_context_list folds both into {active, has_active_context, labels} at the
    # MCP layer. Per-label unit/value is NOT available — unit/value are transient
    # creation metadata consumed by the device + auto-label, never persisted (FC2).
    "context.labels": MethodSpec(5.0, "List context labels"),
    "context.active": MethodSpec(5.0, "Active context label"),
    # context.md_get / md_get_attr stay as wire methods feeding the merged
    # gui_context_md_read tool (no generated tool of their own).
    "context.md_get": MethodSpec(5.0, "List MetaDict keys"),
    "context.md_get_attr": MethodSpec(
        5.0, "Read one MetaDict attribute", (_str("key", "MetaDict key"),)
    ),
    "value.list": MethodSpec(
        5.0,
        "List registered read-only value sources. Returns "
        "{values: [{key, type, owner, description}]}. These are escape-hatch "
        "resolve-once sources for rare defaults and agent reads; prefer typed "
        "RPCs when a stable API exists.",
    ),
    "value.read": MethodSpec(
        5.0,
        "Resolve one registered value source immediately. Returns "
        "{key, type, owner, description, value}. Optional 'type' is one of "
        "int|float|str|bool and must match the registered source type.",
        (
            _str("key", "Registered value source key, e.g. device.active_flux.value"),
            _str_opt("type", "Optional expected type: int, float, str, or bool"),
        ),
    ),
    "context.ml_get": MethodSpec(
        5.0,
        "List ModuleLibrary entries with their discriminator: returns "
        "{modules: [{name, kind}], waveforms: [{name, style}]}, sorted by name. "
        "'kind' is the module type tag (e.g. 'pulse', 'reset/bath'); 'style' is the "
        "waveform style (e.g. 'gauss', 'const'). Read one entry's full cfg with "
        "gui_context_ml_inspect.",
        tool_name="gui_context_ml_list",
    ),
    # context.md_set_attr stays as a wire method feeding gui_context_md_write's
    # fan-out (no generated tool — the single-attr MCP surface is retired in E5).
    "context.md_set_attr": MethodSpec(
        5.0,
        "Set one MetaDict attribute",
        (_str("key", "MetaDict key"), _json("value", "JSON-safe value")),
    ),
    # context.md_del_attr stays as a wire method feeding gui_context_md_delete's
    # batch fan-out (no generated tool).
    "context.md_del_attr": MethodSpec(
        5.0, "Delete one MetaDict attribute", (_str("key", "MetaDict key"),)
    ),
    "context.ml_del_module": MethodSpec(
        5.0,
        "Delete one ModuleLibrary module. Echoes {deleted: name}. cfg refs pointing "
        "at this entry degrade to inline Custom (the value is kept inline, not lost); "
        "to re-link, edit them.",
        (_str("name", "Module name"),),
        tool_name="gui_context_ml_delete_module",
    ),
    "context.ml_del_waveform": MethodSpec(
        5.0,
        "Delete one ModuleLibrary waveform. Echoes {deleted: name}. cfg refs pointing "
        "at this entry degrade to inline Custom (the value is kept inline, not lost); "
        "to re-link, edit them.",
        (_str("name", "Waveform name"),),
        tool_name="gui_context_ml_delete_waveform",
    ),
    "context.ml_rename_module": MethodSpec(
        5.0,
        "Rename a ModuleLibrary module old→new (clash fails fast). Echoes "
        "{renamed: new}. cfg refs to 'old' degrade to inline Custom (the value is "
        "kept inline, not lost); to re-link, edit them.",
        (_str("old", "Current module name"), _str("new", "New module name")),
    ),
    "context.ml_rename_waveform": MethodSpec(
        5.0,
        "Rename a ModuleLibrary waveform old→new (clash fails fast). Echoes "
        "{renamed: new}. cfg refs to 'old' degrade to inline Custom (the value is "
        "kept inline, not lost); to re-link, edit them.",
        (_str("old", "Current waveform name"), _str("new", "New waveform name")),
    ),
    "context.ml_list_roles": MethodSpec(
        5.0,
        "List experiment-role templates for gui_context_ml_create_from_role. Returns "
        "{roles: [{role_id, label, item_kind, default_name}]}. Each role seeds a "
        "blank module/waveform with md-linked defaults (e.g. 'res_probe', "
        "'bath_reset'); 'default_name' is the suggested entry name.",
    ),
    "context.ml_create_from_role": MethodSpec(
        10.0,
        "Create a blank ModuleLibrary module/waveform from a named role "
        "(gui_context_ml_list_roles) and register it under 'name'. The item kind "
        "(module/waveform) is derived from 'role_id'. One-shot: seeds the role's "
        "md-linked defaults (lowered to the md's current values) — it does NOT open "
        "an editing session. Echoes {created: name}. To then change the entry use "
        "gui_editor_open(from_name=name).",
        (
            _str("role_id", "role id from gui_context_ml_list_roles"),
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
        "converter port, sample rate, max pulse/buffer length) plus 'is_mock'. "
        "The structured 'cfg' (the full ~2 KB QICK config) is included only when "
        "include_cfg=true (default false), so the common case pays nothing for it. "
        "Requires a connected SoC. The SoC has no teardown (Pyro4-backed): there "
        "is no disconnect / reconnect / health-check tool (deferred, E3).",
        (_bool_default("include_cfg", False, "Include the full ~2 KB QICK cfg"),),
    ),
    "project.info": MethodSpec(
        5.0,
        "Read the applied project identity: chip_name / qub_name / res_name plus "
        "the resolved result_dir and database_path. Fast-fails with "
        "precondition_failed (no_project) when no project is applied yet.",
    ),
    # Qubit-scoped arbitrary waveform assets
    "arb_waveform.list": MethodSpec(
        5.0,
        "List qubit-scoped arbitrary waveform data keys. Returns {waveforms: [name]}.",
        tool_name="list_arb_waveform",
    ),
    "arb_waveform.preview": MethodSpec(
        10.0,
        "Load one arbitrary waveform asset and render a normalized I/Q/Abs preview "
        "PNG. Returns {recipe, preview_figure}; recipe is null for raw imported "
        "assets.",
        (_str("name", "Arbitrary waveform data_key"),),
        tool_name="get_arb_waveform_preview",
    ),
    "arb_waveform.set": MethodSpec(
        10.0,
        "Create or overwrite an arbitrary waveform from a formula recipe. The recipe "
        "fully replaces waveform data and is embedded into the single .npz asset. "
        "Returns {success, status, preview_figure}.",
        (
            _str("name", "Arbitrary waveform data_key"),
            _json("recipe", "Formula recipe object"),
            _bool_default("overwrite", False, "Allow replacing an existing data_key"),
            _expected_versions(),
        ),
        tool_name="set_arb_waveform",
    ),
    # Resource version table (optimistic-concurrency guard baseline). Full
    # snapshot the mcp layer reads to track last-seen versions; the version
    # integers are mcp/RPC bookkeeping and are never surfaced to the agent.
    "resources.versions": MethodSpec(5.0, "Snapshot of all resource versions"),
    # Connection / startup
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
        "Set the project: chip / qubit / resonator names, plus optional "
        "result_dir and database_path. Omit them to use the default per-qubit "
        "roots (<cwd>/result/<chip>/<qub> and <cwd>/Database/<chip>/<qub>, the "
        "same the setup dialog pre-fills) — the project is runnable either way. "
        "Pass explicit paths to override. Echoes the resolved project: "
        "{chip_name, qub_name, res_name, result_dir, database_path} (the paths "
        "are the defaults filled in when omitted).",
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
        tool_name="gui_project_apply",
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
    # Async operation handle: block until an operation (device.connect /
    # device.disconnect / device.setup / tab.run_start, identified by the operation_id
    # they return) settles. mcp bookkeeping only — agents drive it via semantic
    # wait tools, never raw. (soc.connect is NOT here: it is a synchronous RPC.)
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
    "adapter.list": MethodSpec(
        5.0, "List available adapters. Returns {adapters: [name]}."
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
    "view.screenshot": MethodSpec(
        10.0,
        "Capture the WHOLE main window (client area + floating widgets) as base64 "
        "PNG. Runs MainWindow.grab() on the main thread (auto-marshalled, like "
        "dialog.screenshot).",
    ),
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
        "Install a FluxoniumPredictor from a params.json file (its fluxdep_fit "
        "section). Replaces any currently loaded predictor. Echoes the installed "
        "model: {loaded: true, path, flux_bias, flux_half, flux_period, EJ, EC, EL}.",
        (
            _str("path", "Predictor file path"),
            _num_default("flux_bias", 0.0, "Flux bias"),
        ),
        tool_name="gui_predictor_install_from_file",
    ),
    "predictor.set_model_params": MethodSpec(
        10.0,
        "Build+install a FluxoniumPredictor directly from typed model params "
        "(no params.json). EJ/EC/EL are the Fluxonium energies in GHz "
        "(e.g. 4:1:1); flux_half/flux_period are the value->flux affine anchors "
        "in device-value units; flux_bias is the bias correction. Replaces any "
        "currently loaded predictor. Echoes the installed model: {loaded: true, "
        "path: null, flux_bias, flux_half, flux_period, EJ, EC, EL} (path is null "
        "because this predictor has no backing file).",
        (
            _num("EJ", "Josephson energy E_J (GHz)"),
            _num("EC", "Charging energy E_C (GHz)"),
            _num("EL", "Inductive energy E_L (GHz)"),
            _num("flux_half", "Half-flux (Phi0/2) value->flux anchor (device units)"),
            _num("flux_period", "Flux period (device units); must be non-zero"),
            _num_default("flux_bias", 0.0, "Flux bias correction (device units)"),
        ),
        tool_name="gui_predictor_install_params",
    ),
    "predictor.clear": MethodSpec(
        5.0,
        "Unload the current predictor (idempotent — succeeds with no predictor "
        "loaded). Returns {loaded: false}.",
        tool_name="gui_predictor_unload",
    ),
    "predictor.predict": MethodSpec(
        10.0,
        "Predict a transition frequency at a device-value setpoint. Returns "
        "{freq_mhz}.",
        (
            _num(
                "device_value",
                "Device-value setpoint in the instrument's native unit (e.g. "
                "current in A for YOKOGS200) — NOT a flux quantum. The predictor "
                "applies an internal value-to-flux affine conversion; passing a "
                "flux quantum (e.g. 0.5) will silently yield a wrong frequency.",
            ),
            _int_default("from_level", 0, "From level"),
            _int_default("to_level", 1, "To level"),
        ),
    ),
    "predictor.info": MethodSpec(
        5.0,
        "Read the current predictor's installed model. Returns {loaded: false} "
        "when none is loaded, else {loaded: true, path, flux_bias, flux_half, "
        "flux_period, EJ, EC, EL} (path is null for an in-memory install).",
    ),
    # Analyze
    "tab.get_analyze_result": MethodSpec(
        5.0, "Read tab analyze result scalar summary", (_str("tab_id"),)
    ),
    "tab.get_analyze_params": MethodSpec(
        5.0, "Read current analyze params", (_str("tab_id"),)
    ),
    "tab.analyze": MethodSpec(
        30.0,
        "Start analyzing the tab's run result. Analyze runs on a worker thread "
        "and returns an operation_id (like run/connect/device); the mcp "
        "gui_tab_analyze_start tool awaits it so the agent sees one synchronous "
        "call. 'updates' optionally overrides analyze params (read the current "
        "ones with gui_tab_get_analyze_params). Makes the tab busy while it runs; "
        "a concurrent save/edit returns precondition_failed until it settles. "
        "Read the fit summary with gui_tab_get_analyze_result.",
        (_str("tab_id"), _obj_default("updates", "Analyze param updates")),
    ),
    # Post-analysis (second analysis layer; mirrors the analyze trio above). It
    # runs on top of an existing primary analyze result, so every entry gates
    # server-side on that result existing (no version guard — same as
    # tab.analyze, which gates on has_run_result rather than expected_versions).
    "tab.get_post_analyze_result": MethodSpec(
        5.0, "Read tab post-analysis result scalar summary", (_str("tab_id"),)
    ),
    "tab.get_post_analyze_params": MethodSpec(
        5.0, "Read current post-analysis params", (_str("tab_id"),)
    ),
    "tab.post_analyze": MethodSpec(
        30.0,
        "Start the second-layer (post) analysis on the tab's PRIMARY analyze "
        "result. Runs on a worker thread and returns an operation_id (like "
        "tab.analyze); the mcp gui_tab_post_analyze_start tool awaits it so the "
        "agent sees one synchronous call. Fast-fails with precondition_failed when the "
        "tab has no primary analyze result to build on. 'updates' optionally "
        "overrides post params (see gui_tab_get_post_analyze_params). Read the "
        "fit summary with gui_tab_get_post_analyze_result.",
        (_str("tab_id"), _obj_default("updates", "Post-analysis param updates")),
    ),
    # Writeback workflow (under tab.* namespace — Phase 170c) — a persistent
    # draft computed once at analyze time.
    "tab.writeback_preview": MethodSpec(
        5.0,
        "List the tab's persistent writeback draft (pure read — not a dry-run; the "
        "draft was computed once at analyze time). Returns {has_draft, items}; "
        "has_draft is false before any analyze produced a draft. Each item: id "
        "(<kind>-<n>, kind∈md|ml|wf), target_name (apply destination, editable), "
        "kind (metadict|module|waveform), description, selected; metadict adds "
        "proposed_value; module/waveform add editor_id + has_edit_schema. A "
        'complex metadict proposed_value is carried as {"__complex__": [re, im]} '
        "(JSON has no complex). Edit an item via gui_tab_writeback_set_item; the "
        "user's Edit dialog renders the same model (WYSIWYG).",
        (_str("tab_id"),),
        tool_name="gui_tab_writeback_list",
    ),
    "tab.writeback_set": MethodSpec(
        5.0,
        "Edit a persistent writeback item by id — the single writeback editing "
        "surface. selected? / target_name? apply to any item. proposed_value? is "
        "the METADICT-only facet (a complex value is passed as "
        '{"__complex__": [re, im]}, the same shape the list emits; it applies as '
        "a Python complex). edits? is the MODULE/WAVEFORM-only facet: an ORDERED "
        "list of {path, value} cfg edits applied to the item's draft (no editor_id "
        "needed — the surface resolves it internally). Apply ref-switch edits "
        "before dependent inner-path edits (a ref switch removes child paths); "
        "fail-fast and non-atomic. proposed_value and edits are mutually exclusive "
        "(different item kinds). Echoes the edited {item}; an edits batch also "
        "returns {valid, removed, added} aggregated across the batch (same shape "
        "as tab.set_cfg). Read the item's current paths via tab.writeback_preview.",
        (
            _str("tab_id"),
            _str("id", "writeback item session id (<kind>-<n>)"),
            # Boolean (not JSON): a JSON schema of {type: boolean} makes the
            # client send a real boolean. Declared as JSON, the client may send
            # the string "false", which ``bool("false")`` wrongly reads as True.
            ParamSpec("selected", JsonType.BOOLEAN, required=False),
            _str_opt("target_name", "new apply destination name"),
            ParamSpec(
                "proposed_value",
                JsonType.JSON,
                required=False,
                description="Proposed metadict scalar (metadict items only)",
            ),
            ParamSpec(
                "edits",
                JsonType.JSON,
                required=False,
                description="Ordered list of {path, value} cfg edits "
                "(module/waveform items only)",
            ),
            _expected_versions(),
        ),
        tool_name="gui_tab_writeback_set_item",
    ),
    "tab.writeback_apply": MethodSpec(
        10.0,
        "Apply the tab's persistent writeback draft as-is (edit it first via "
        "gui_tab_writeback_set_item). Applies items currently selected. Returns "
        "{applied_ids, written, context_version}: written lists the destination "
        "names actually pushed, split by kind ({md, ml_modules, ml_waveforms}); "
        "context_version is the bumped 'context' resource version after apply (use "
        "it as an expected_versions guard on a follow-up write).",
        (_str("tab_id"), _expected_versions()),
    ),
    # CfgEditor sessions — headless, stateful ml-entry editing for the agent.
    "editor.new": MethodSpec(
        5.0,
        "Open a stateful editing session over an EXISTING ModuleLibrary "
        "module/waveform (by 'from_name'). To create a new blank/shaped entry, "
        "use context.ml_create_from_role (e.g. role_id='pulse:blank' or a named role) "
        "then editor.new(from_name=name) to edit it. item_kind is 'module' or "
        "'waveform'. Returns {editor_id, tree} (tree = the nested current-value "
        "view, same shape as editor.get / tab.get_cfg).",
        (
            _str("item_kind", "'module' or 'waveform'"),
            _str("from_name", "Existing ml entry name to load for editing"),
        ),
    ),
    "editor.set_field": MethodSpec(
        5.0,
        "Set one field in an editing session. 'path' is a dotted path from "
        "editor.new/get (ModuleRef sub-fields descend directly, no 'value' "
        "segment); 'value' is a JSON scalar, or an md-reference expression as "
        '{"__kind":"eval","expr":"r_f - 0.1"} (resolved against MetaDict at '
        "commit), or a registered value source as "
        '{"__kind":"value_ref","key":"device.active_flux.value","type":"float"} '
        "(resolved immediately at set time and stored as a direct scalar; discover "
        "keys with value.list / value.read). NOTE: eval/value_ref forms are "
        "accepted ONLY on a scalar leaf — a "
        "sweep_edge (a sweep's start/stop/expts/step) accepts ONLY a number/int, "
        "never an eval/value_ref; an adapter's default eval edge cannot be "
        "overwritten this way, pass a numeric value instead. "
        "Returns {valid, removed, added} — does NOT echo cfg content "
        "(that would force a lowering pass that eagerly evaluates EvalValue). "
        "'valid' is whether the whole draft is currently valid; 'removed'/'added' "
        "list settable paths a ModuleRef key switch ('<path>.ref') dropped/"
        "created so you need not re-list after a variant switch. To read cfg use "
        "tab.get_cfg / editor.get (the nested current-value tree).",
        (
            _str("editor_id"),
            _str("path", "Dotted field path"),
            _json(
                "value",
                "JSON scalar, {__kind:eval, expr}, or {__kind:value_ref, key, type?}",
            ),
        ),
    ),
    "editor.get": MethodSpec(
        5.0,
        "Read an editing session's settable cfg as a NESTED tree of current "
        "values (the read-only view; edit a leaf with editor.set_field using the "
        "leaf's dotted path). Node shape, distinguished by '$'-prefixed reserved "
        "keys: a SCALAR leaf is its bare current value (null = unset); an ENUM "
        "scalar leaf is {'$value': current, '$choices': [...]}; a SWEEP is a "
        "sub-tree of bare edges {start, stop, expts, step} (each edge accepts "
        "ONLY a number/int via editor.set_field — NOT an eval/ref); a REF node "
        "(module/waveform/device selector) is {'$ref': {'current': <chosen>, "
        "'options': [<names>]}, <chosen variant's settable sub-tree>} — only the "
        "CURRENTLY-CHOSEN variant is expanded; 'options' lists bare names while "
        "'current' may be a tagged internal key — switch by passing a bare "
        "'options' name to editor.set_field on the ref's dotted path. "
        "Any other dict is a plain section sub-tree (its keys are child fields). "
        "'prefix' (optional, dotted) returns just the sub-tree rooted at that "
        "node (a prefix at a sweep edge returns the whole sweep node); a prefix "
        "matching nothing returns {}.",
        (
            _str("editor_id"),
            _str_opt(
                "prefix",
                "Return only the sub-tree rooted at this dotted path "
                "(e.g. 'modules.readout'); omit for the whole draft. No match → {}",
            ),
        ),
    ),
    "editor.commit": MethodSpec(
        10.0,
        "Save the editing session (from gui_editor_open) as a ModuleLibrary "
        "module/waveform: lower the session (eval expressions resolved against "
        "MetaDict to concrete numbers) and register it into the ModuleLibrary "
        "under 'name'. This is NOT 'apply a tab cfg edit' — tab cfg edits are "
        "already live (WYSIWYG); this persists the draft as a named ml entry. "
        "Returns {}. On success the session is destroyed; on validation failure "
        "it RAISES and the session is kept so you can fix and retry.",
        (
            _str("editor_id"),
            _str("name", "ml entry name to register under"),
            _expected_versions(),
        ),
        tool_name="gui_editor_save",
    ),
    "editor.discard": MethodSpec(
        5.0,
        "Discard an editing session (from gui_editor_open) without writing to the "
        "ModuleLibrary. Returns {}.",
        (_str("editor_id"),),
    ),
    # Notify prompt — agent-initiated user question (ADR-0025).
    # Two-RPC design: notify.open mints the token + opens the dialog on the
    # main thread; notify.await blocks the off-main worker until the user replies,
    # dismisses, or the dialog's QTimer fires. Both are excluded from
    # auto-generation via _NON_GENERATED_METHODS in the MCP server so that only
    # the hand-written gui_prompt_user tool is exposed to the agent.
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
