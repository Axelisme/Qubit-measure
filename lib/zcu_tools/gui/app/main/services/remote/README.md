# `gui.app.main.services.remote` — measure-gui RemoteControlAdapter

**Last updated:** 2026-09-28，writeback and artifact save integration

This package is the GUI-process side of measure-gui remote control. It exposes a
local NDJSON RPC surface over the live `Controller`, marshals State-owned work onto
the injected owner scheduler, serializes selected events, and enforces measure-gui policy
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
- `method_specs.py`：Qt-free projection for wire method schema and timeouts.
- `method_entries/`：single registration source for method name, handler ref,
  `MethodSpec`, agent exposure/guard policy and `ParamSpec` shorthands. The
  `rpc.catalog` projection uses those same entries; handler refs are resolved
  only by dispatch.
- `events.py`：domain payload type to wire event serializer mapping.
- `dialogs.py`：wire-stable dialog names.
- `path_resolver.py`：flat mutation-target projection for cfg-editor sessions.
- `cfg_observation.py`：complete cached cfg observation and prefix projection.
- `wire_version.py`：measure-gui wire contract version and GUI code revision.

Shared transport primitives live in `zcu_tools.gui.remote`: NDJSON framing,
typed request/reply/error envelope, socket endpoint, owner-scheduler dispatch, and
router scaffolding.

## Wire Contract

```text
Request  -> {"id": "...", "method": "tab.run_start", "params": {...}}
Reply    <- {"id": "...", "ok": true, "result": {...}}
Reply    <- {"id": "...", "ok": false, "error": {"code": "...", "message": "...", "reason": "..."}}
Push     <- {"event": "...", "payload": {...}, "seq": 123, "origin": {"kind": "agent", "operation_id": "7"}}
```

- One connection has at most one in-flight RPC.
- Request and response roots are JSON objects.
- Request and response lines allow up to 8 MiB of UTF-8 bytes, excluding the
  newline. There is no chunking protocol. Unencodable replies return a bounded
  `internal` error with reason `response_encoding_failed`. The handler may have
  executed, so callers must inspect state before retrying a mutation. If the
  fallback cannot fit or the reply queue rejects delivery, the connection closes.
  Each client has a 16 MiB encoded-byte budget including its in-flight frame;
  exceeding it closes that client and releases its backlog. MCP rejects oversized
  requests before sending and closes on oversized incoming frames with an explicit error.
- A failed full read does not advance the MCP observation baseline. Large context
  export is not a framing exception or an implicit partial read.
- Error codes are closed and typed in `gui.remote.errors`.
- `wire.version` is available before auth; all other methods require auth when a
  token is configured. MCP `connect(token=...)` authenticates before catalog
  loading and reuses the credential after a GUI restart; failed auth is not a
  wire-version mismatch.
- Loopback without token means any same-user local process can control the GUI.

## Dispatch And Threads

Normal handlers run on the State owner thread through the injected `OwnerScheduler`.
The Qt composition root supplies `QtOwnerScheduler`; headless tests use
`ManualOwnerScheduler`. Handler exceptions become typed error envelopes.

Caller-correctable producer exceptions以remote-independent `ExpectedErrorCategory`分類。shared
dispatch在main/off-main兩條路徑先讓direct `RemoteError`穿透，再以nominal `ExpectedError`作為
唯一generic gate：`INVALID_INPUT`映射`INVALID_PARAMS`，`FAILED_PRECONDITION`映射
`PRECONDITION_FAILED`，message/reason原樣保留且generic data固定為`None`。handler-local
request coercion與structured policies（例如arb waveform data）仍由原handler擁有。

ordinary `RuntimeError`、ProviderError、I/O/persistence與invariant failure進controller-error
branch並記錄traceback。translator projection failure也由同一controller branch收斂，不會留下
empty reply holder（ADR-0047）。

Only bounded wait handlers run off-main:

- `operation.await`
- `notify.await`

Off-main handlers only consume thread-safe channels. They do not read or mutate
owner-thread state, version tables, controller objects, editor sessions, or
Qt widgets.

## Events And Diagnostics

Event push is still available on the wire for GUI/internal consumers. Event
payloads use requery hints for complex objects; live Python objects never cross
the wire.
EventBus pushes add process-wide `seq` and `origin={kind, operation_id}` without
changing the event name or payload shape. The per-connection `client_id` is
RPC-side bookkeeping and is deliberately omitted. Diagnostic and cfg-editor
pushes are not EventBus broadcasts and retain `{event, payload}`.
Event serializers and NDJSON encoding run only when the endpoint finds a matching
live subscriber. Recipient selection is two-phase: payload construction stays on
the State owner thread and outside the endpoint registry lock, then subscribe state
and link liveness are revalidated immediately before enqueue. One event is built
once for any number of matching clients; unsubscribe/disconnect completed first
cannot receive a late push.
Internal tab interaction/content payloads include closed domain facts used by the
Qt reaction matrix. Their serializers deliberately omit those facts and preserve
the existing event names and `{tab_id, requery}` shape.

Cfg-editor change producers pass a payload factory rather than transport state.
Editor versions still bump on every edit; `current_targets()` is materialized and projected once
only when an `editor_id` subscriber exists. `editor_closed` removes a client's
subscription only after its close push is accepted by that client's queue.

Diagnostics are separate from EventBus. The controller pushes diagnostics to the
remote adapter sink, which broadcasts diagnostic payloads to clients regardless
of subscription. Measure MCP does not subscribe, queue or piggyback events.
Agent-visible async completion comes from operation request/reply, not pushes.

## Version Handshake

The launch/connect note reports three numbers:

- `WIRE_VERSION`：GUI RPC contract. MCP pins and compares this value.
- `GUI_VERSION`：GUI process code revision. It is displayed, not compared.
- `MCP_VERSION`：MCP bridge code revision. It is displayed by the bridge, not
  owned here.

Current measure-gui values are `WIRE_VERSION = 70`, `GUI_VERSION = 102`, and
`MCP_VERSION = 93` (defined in `zcu_tools.mcp.measure.server`). WIRE 70 exposes
shared artifact status, batch save operations and guarded close/shutdown replies.
GUI 102 tracks ordered saves and actual output paths; MCP 93 adds save, close and
graceful shutdown tools. WIRE 69 adds
complete writeback previews and identity-preserving batch results. GUI 101 applies
explicit items through shared drafts and follows their pane; MCP 92 forwards the
writeback tool to that owner. WIRE 68 exposes analysis parameters and invalidation
facts; GUI 100 owns analysis validation and explicit pane following; MCP 91 adds
run/analyze tools with bounded waits. WIRE 67 adds
aggregate cfg edits and `context.ml_edit`; GUI 99 owns sequential library
commits through the shared draft model and preserves batch error categories.
MCP 90 exposes cfg/library tools and reports partial commits. WIRE 66 moves
seen guards into the GUI, removes wire expectations and write receipts, exposes
operation state in snapshots, and adds `tab.open_file`. GUI 98 owns new-tab
loading and failure cleanup; MCP 89 forwards once without version bookkeeping.
WIRE 65 adds State-cached device fields to snapshots during setup. WIRE 64 exposes
predictor calibration; GUI 96 routes it through the shared predictor port.
GUI 93 removes
Run's context-content dependency after freezing cfg and device inputs; tab cfg,
tab existence, SoC, devices and hardware exclusion remain protected. WIRE 63
carries complete cached cfg observations; GUI 92 bounds response encoding failures.
WIRE 62 adds
`context.snapshot`, conditional full-read policy and certified resource creation
to the live catalog. GUI 90 declares those policies; MCP 82 consumes them.
GUI 89 reports
an already failed operation as `operation_failed` on cancel, rather than
`finished`; MCP 81 likewise reports failure when the short cancellation wait
observes a failed outcome. GUI 88 made domain cancel RPCs internal in the MCP
catalog and bounded `notify.await` to 600 seconds. GUI 87 corrected no-project
RPC guidance. WIRE 61 adds
`__agent_write_versions` to replies for catalog-declared writes: each changed
resource carries its versions before and after that handler on the owner thread.
WIRE 60 adds `rpc.catalog.reveals_without` for partial reads; MCP samples the
resource version before a full read and records it only after success.

Only wire-contract changes bump `WIRE_VERSION`. GUI-internal changes that need a
reload signal bump `GUI_VERSION`; MCP-only tool/policy changes bump
`MCP_VERSION`.

## Resource-Version Guard

The GUI maintains a monotonic resource version table for context, SoC, devices,
tabs, results, save paths, and editor sessions. Each remote connection starts with
an empty seen map. The adapter compares it with current versions on the State
owner thread before calling the controller. Missing observations, including
version zero, are stale. Wire methods do not accept `expected_versions`.

Run uses the observed cfg and device snapshots, not live md/ml. Its guard does
not require exporting the entire context. Load, editor commit and writeback still
use live context and retain their context guard; Run's change does not authorize
removing those dependencies.

GUI owns observation and write tracking:

- Successful full reads record only the keys named by `reveals`. Optional partial
  parameters, including an explicit empty `prefix`, do not reveal the whole cfg.
  The dispatcher preserves the original request to distinguish omission from defaults.
- Successful writes advance only previously seen resources whose observations
  match the handler's before-versions. Unseen consequential changes stay unseen.
  New-tab identity certifies existence only, not cfg/result/analysis.
- Handler failure or timeout does not establish seen. Reply encoding failure
  rolls back its observation update on the owner thread before the next request.
  This is not rollback of business effects or proof of client receipt.
- `tab.snapshot(tab_id)` reveals existence, result/analysis/post revisions,
  availability and effective paths. It does not serialize raw result arrays or
  claim cfg/writeback contents. The all-tabs index reveals no per-tab state.
- `soc.info(include_cfg=true)` reveals the full SoC cfg. `context.snapshot` reveals
  the active label and complete serializable md/ml contents. Encoding failure
  does not establish a baseline. These replies may be large or sensitive.
- Off-main methods cannot declare guard/reveals or owner-thread write tracking.
  Disconnection discards seen; reconnection requires explicit new reads.

MCP forwards each RPC once and returns observed operation state to the agent.
It does not keep a second seen map, consume write receipts, or perform hidden
pre-reads to unlock a mutation. Snapshot revisions distinguish successive results,
including replacements from the same source file.

A stale guarded mutation has one wire-level recovery contract. The server returns
`PRECONDITION_FAILED` with `data={"stale": [...]}`, where `stale` lists every
resource whose current version no longer matches the caller baseline. The client
must re-snapshot each listed resource through its corresponding read method,
then decide whether to retry. It must not retry against the old snapshots. Stale conflicts
do not add another `ErrorCode`; a future Web adapter may translate this existing
failure into an HTTP-specific status without changing the wire enum (ADR-0052).

`tab.open_file` creates a new tab and reuses the application load operation,
including cfg backfill, cleanup on load failure and focus restoration. It requires
an explicitly observed context, not observations of a tab that does not yet exist.
Backfill failure retains the result and reports `not_applied`; cleanup failure
reports `cleanup_failed`. Existing-tab `tab.load_data` keeps its stricter guards.

## Method Surface

The wire surface is grouped by ownership:

- `startup.*` / `result_scope.*`：project and result-scope setup.
- `context.*`：MetaDict / ModuleLibrary / active context operations through
  `ContextControlPort`; role-catalog create/list stays on the app controller.
- `soc.*`：mock or remote SoC connection.
- `device.*`：device connect/disconnect/setup/snapshot through `DeviceControlPort`.
- `predictor.*`：Fluxonium predictor load, edit, clear, and predictions through
  `PredictorControlPort`.
- `tab.*`：tab lifecycle, cfg discovery/edit, run, load, save (data via `tab_id` only; image via `(tab_id, subtab_id)` with `analysis|post_analysis`) and figures via `(tab_id, subtab_id)` (`run` reads live FigureContainer, `analysis`/`post_analysis` read canonical State figures). `tab.snapshot.save_paths` projects independent `data_path`、`analysis_image_path`與`post_analysis_image_path`; explicit save destinations update the shared GUI drafts. `tab.save_artifacts` submits one application-owned operation for data/analysis/post keys, returns reserved destinations plus an operation id, and guards the observed result, analysis and path resources. Reserved paths are not completion evidence; terminal success and artifact snapshots establish saved results.
- `tab.analyze` / `tab.post_analyze`：primary and secondary analysis (analysis owns `analysis` pane; post owns `post_analysis`).
- `tab.writeback_*`：pane-qualified writeback preview/edit/apply via `(tab_id, subtab_id=analysis|post_analysis)`; draft is opaque, not bound to source context; preview/apply echo `destination_context` (active ExpContext projection at reply time).
- `editor.*`：headless cfg-editor session lifecycle.
- `operation.*` / `notify.*`：live operation indexing, bounded wait, domain-owned cancellation, progress and prompt replies.
- `arb_waveform.*`：qubit-scoped arbitrary waveform asset operations.
- `value.*`：read-only session value lookup through `ContextControlPort`.

Subtab locator is required and closed (`run|analysis|post_analysis`); save_image
only accepts `analysis|post_analysis`. `method_entries/` owns the wire method
name, handler ref, schema, agent exposure and guard/reveal/operation policy.
Adding a wire method requires one entry; MCP receives the projection through
`rpc.catalog` after its version handshake. Descriptions direct the caller to
currently available `rpc_call` methods and to `wait(op)` for asynchronous
terminal status, not removed aliases. No tool inventory is generated from
`MethodSpec`.

## Cfg Editing

`path_resolver.py` projects nominal `SettableTarget` entries for mutations and path
changes. `cfg_observation.py` projects `CfgDraft.observe()` data, never binding
field/editor classes. Setters retain canonical paths and reject legacy aliases.

`tab.get_cfg`, `editor.get`, and `editor.new` return the same typed `tree` format.
Nodes contain kind/path/label/valid. Sections and active references have named
children, including locked literals. Scalar/literal input and sweep inputs retain
mode/raw/resolved/error/validation_error. Reference nodes include their chosen key,
cached shape label, error, override flag, and choices. Reads do not resolve sources.
Unknown objects, non-string object keys, and nonfinite numbers fail serialization;
complex numbers use the shared reversible tag, with no string fallback.

Prefix reads select a node while preserving its full path; sweep edges and reference
keys select their parent node. Unknown prefixes return an empty object. Only a
successful read with no prefix parameter establishes the full cfg observation.
Wire keys children/input/inputs are not mutation-path segments.

Scalar values may be direct values, tagged eval values, or tagged value refs.
Eval values store a resolved snapshot at set/lower time. Value refs resolve once
through the session value lookup and then become direct scalars.

Sweep nodes appear as editable subtrees, not as lowered `SweepCfg` objects.
`SweepSpec` exposes `start` / `stop` / `expts` / `step`; `CenteredSweepSpec`
exposes `center` / `span` / `expts` / `step`. `editor.set_field` accepts the same
dotted edge paths that `tab.get_cfg` reports.

Headless editor sessions are owned by `CfgEditorService`. Agent-created sessions
are garbage-collected on commit/discard/client drop; UI-owned sessions are tied
to their owner widget or tab. Each owner session has a fresh id; Load's successful
Config replacement retires the old id, so clients rediscover the new session
before editing.

## Operation Handles

Start methods return GUI-local operation ids on the wire. The GUI projects active
run, analyze and device ids from their owners, regardless of who started them.
MCP assigns session-local opaque integer handles to both started and discovered
operations; a GUI restart can reuse a wire id but cannot reuse an exposed MCP
handle. `operation.await` reads the shared handle channel off-main and rejects
unknown or evicted GUI ids. `operation.cancel` runs on the owner thread and uses
the domain cancel hook; a non-cancellable operation fails with `not_cancellable`.
MCP uses `cancel(op)` alone. The domain-specific wire cancellation methods remain
available to other socket consumers but are absent from its live catalog.
MCP `wait` reports status, progress, user feedback, timeout or failure as data;
figures, summaries and device snapshots come from typed getters after completion.

`soc.connect` is synchronous and does not enter the operation-handle table.

## Launch / Shutdown

The GUI launcher starts the remote-control socket after the main window exists.
Shutdown flows through the same MainWindow close path as the UI, persists state,
marks the controller shutting down, stops the remote service, and then closes Qt
resources.

The MCP `connect` tool attaches to an existing GUI or explicitly launches one;
exiting MCP disconnects without closing the GUI.
