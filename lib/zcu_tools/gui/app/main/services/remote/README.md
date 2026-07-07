# `gui.app.main.services.remote` — measure-gui RemoteControlAdapter

**Last updated:** 2026-07-07 - centered sweep cfg paths

This package is the GUI-process side of measure-gui remote control. It exposes a
local NDJSON RPC surface over the live `Controller`, marshals GUI-owned work onto
the Qt main thread, serializes selected events, and enforces measure-gui policy
such as resource-version guards and editor lifecycle tracking.

The agent-facing MCP bridge lives in `zcu_tools.mcp.measure`. This package does
not declare MCP tools and does not own stdio transport.

## Layout

- `service.py`：`RemoteControlAdapter`, a measure-gui subclass of shared
  `RemoteControlServiceBase`; owns client context, guard hooks, diagnostics, and
  editor cleanup.
- `dispatch.py`：runtime method registry projection; keeps the public
  `METHOD_REGISTRY` import path stable.
- `handlers/`：grouped wire method handlers bound to controller, control facets,
  or render-view calls.
- `method_specs.py`：Qt-free public projection for wire method schema, timeouts,
  and MCP generation metadata.
- `method_entries/`：single registration source for method name, handler ref,
  `MethodSpec`, MCP exposure policy, and `ParamSpec` shorthands. Handler refs
  are resolved only by the dispatch projection.
- `events.py`：domain payload type to wire event serializer mapping.
- `dialogs.py`：wire-stable dialog names.
- `path_resolver.py`：dotted-path mutation and settable-tree projection for
  cfg-editor sessions.
- `wire_version.py`：measure-gui wire contract version and GUI code revision.

Shared transport primitives live in `zcu_tools.gui.remote`: NDJSON framing,
typed request/reply/error envelope, socket endpoint, main-thread dispatcher, and
router scaffolding.

## Wire Contract

```text
Request  -> {"id": "...", "method": "tab.run_start", "params": {...}}
Reply    <- {"id": "...", "ok": true, "result": {...}}
Reply    <- {"id": "...", "ok": false, "error": {"code": "...", "message": "...", "reason": "..."}}
Push     <- {"event": "...", "payload": {...}}
```

- One connection has at most one in-flight RPC.
- Request and response roots are JSON objects.
- Line size is bounded by UTF-8 byte length.
- Error codes are closed and typed in `gui.remote.errors`.
- `wire.version` is available before auth; all other methods require auth when a
  token is configured.
- Loopback without token means any same-user local process can control the GUI.

## Dispatch And Threads

Normal handlers run on the Qt main thread through `MainThreadDispatcher`. Handler
exceptions become typed error envelopes.

Only bounded wait handlers run off-main:

- `operation.await`
- `notify.await`

Off-main handlers only consume thread-safe channels. They do not read or mutate
main-thread-owned state, version tables, controller objects, editor sessions, or
Qt widgets.

## Events And Diagnostics

Event push is still available on the wire for GUI/internal consumers. Event
payloads use requery hints for complex objects; live Python objects never cross
the wire.

Diagnostics are separate from EventBus. The controller pushes diagnostics to the
remote adapter sink, which broadcasts diagnostic payloads to clients regardless
of subscription. MCP keeps only diagnostics from the push stream and piggybacks
them on later tool replies.

Agent-visible async completion comes from `gui_op_poll` / `gui_op_wait`, not
from resource-change events.

## Version Handshake

The launch/connect note reports three numbers:

- `WIRE_VERSION`：GUI RPC contract. MCP pins and compares this value.
- `GUI_VERSION`：GUI process code revision. It is displayed, not compared.
- `MCP_VERSION`：MCP bridge code revision. It is displayed by the bridge, not
  owned here.

Current measure-gui values are `WIRE_VERSION = 48` and `GUI_VERSION = 61`.
`MCP_VERSION` is defined in `zcu_tools.mcp.measure.server`.

Only wire-contract changes bump `WIRE_VERSION`. GUI-internal changes that need a
reload signal bump `GUI_VERSION`; MCP-only tool/policy changes bump
`MCP_VERSION`.

## Resource-Version Guard

The GUI maintains a monotonic resource version table for context, SoC, devices,
tabs, results, save paths, and editor sessions. Guarded mutation methods accept
wire-hidden `expected_versions`; the remote adapter compares them atomically on
the main thread before calling the controller.

MCP owns the agent baseline:

- guarded mutations send expected versions derived from policy tables;
- successful writes refresh the baseline;
- pure reads refresh only keys they fully reveal;
- stale rejection is translated into semantic tool errors for the agent.

Version numbers are a bridge concern. Agents see stale-resource descriptions, not
raw counters.

## Method Surface

The wire surface is grouped by ownership:

- `startup.*` / `result_scope.*`：project and result-scope setup.
- `context.*`：MetaDict / ModuleLibrary / active context operations through
  `ContextControlPort`; role-catalog create/list stays on the app controller.
- `soc.*`：mock or remote SoC connection.
- `device.*`：device connect/disconnect/setup/snapshot through `DeviceControlPort`.
- `predictor.*`：Fluxonium predictor load, edit, clear, and predictions through
  `PredictorControlPort`.
- `tab.*`：tab lifecycle, cfg discovery/edit, run, load, save, figures.
- `analyze.*` / `post_analyze.*`：primary and secondary analysis.
- `writeback.*`：analysis result writeback preview and apply.
- `editor.*`：headless cfg-editor session lifecycle.
- `operation.*` / `notify.*`：generic waits, polls, progress, prompt replies.
- `arb_waveform.*`：qubit-scoped arbitrary waveform asset operations.
- `value.*`：read-only session value lookup through `ContextControlPort`.

`method_entries/` is the registration SSOT. Adding an agent-visible method
requires one entry containing the wire method name, handler ref, method spec, MCP
mapping or override, and tests for generation / guard policy. `method_specs.py`
remains the Qt-free public projection used by MCP generation.

## Cfg Editing

`tab.get_cfg` returns the nested settable value tree for discovery. Mutations use
dotted paths through `tab.set_cfg` or `editor.set_field`.

Scalar values may be direct values, tagged eval values, or tagged value refs.
Eval values store a resolved snapshot at set/lower time. Value refs resolve once
through the session value lookup and then become direct scalars.

Sweep nodes appear as editable subtrees, not as lowered `SweepCfg` objects.
`SweepSpec` exposes `start` / `stop` / `expts` / `step`; `CenteredSweepSpec`
exposes `center` / `span` / `expts` / `step`. `editor.set_field` accepts the same
dotted edge paths that `tab.get_cfg` reports.

Headless editor sessions are owned by `CfgEditorService`. Agent-created sessions
are garbage-collected on commit/discard/client drop; UI-owned sessions are tied
to their owner widget or tab.

## Operation Handles

Start methods return operation ids on the wire. MCP captures those ids and
returns opaque handles to the agent. Generic poll/wait reports status, progress,
user feedback, cancellation, timeout, or failure; figures, summaries, and device
snapshots are read through typed getters after completion.

`soc.connect` is synchronous and does not enter the operation-handle table.

## Launch / Shutdown

The GUI launcher starts the remote-control socket after the main window exists.
Shutdown flows through the same MainWindow close path as the UI, persists state,
marks the controller shutting down, stops the remote service, and then closes Qt
resources.

`gui_launch` starts a new GUI process and expects the requested port to be free.
`gui_bridge_connect` attaches to an already-running GUI.
