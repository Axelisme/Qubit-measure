# `gui.app.measure.remote` — measure-gui RemoteControlAdapter

**Last updated:** 2026-10-12，holder admission、ref-free cfg 與完整 device setup

This package is the GUI-process side of measure-gui remote control. It exposes a
local NDJSON RPC surface over the live `Controller`, injects the application's control identity into shared remote dispatch,
serializes selected events, and owns measure-specific editor lifecycle tracking.

The agent-facing MCP bridge lives in `zcu_tools.mcp.measure`. This package does
not declare MCP tools and does not own stdio transport.

## Layout

- `service.py`：`RemoteControlAdapter`, a measure-gui subclass of shared
  `RemoteControlServiceBase`; injects control authority, and owns
  client editor context, writeback key aliases, diagnostics and editor cleanup.
- `dispatch.py`：runtime method registry projection; keeps the public
  `METHOD_REGISTRY` import path stable.
- `handlers/`：grouped wire method handlers bound to controller, control facets,
  or render-view calls.
- `method_specs.py`：Qt-free projection for wire method schema and timeouts.
- `method_entries/`：single registration source for method name, handler ref,
  `MethodSpec`, agent exposure, readonly and operation policy and `ParamSpec` shorthands. The
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

Parameter writeback previews and write receipts use `kind="parameter"`.
Their draft-local IDs use `param-<n>`. Apply receipts list parameter paths in
`written.parameter`.

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
- Large context export is not a framing exception or an implicit partial read.
- Error codes are closed and typed in `gui.remote.errors`.
- `wire.version` is available before auth; all other methods require auth when a
  token is configured. MCP `connect(token=...)` authenticates before catalog
  loading and reuses the credential after a GUI restart; failed auth is not a
  wire-version mismatch.
- Loopback without token means any same-user local process can control the GUI.

## Control identity initialization

After `wire.version` and optional `auth`, send `control.connect` with
`{label: "agent", mode: "owner"}` (both values default as shown). Before this
command a socket is an observer: explicit readonly methods and `rpc.catalog`
are available, business writes return `not_holder`, and acquire/release return
`readonly_connection`. Initialization fixes label and mode for that socket.
An identical repeat is inert; a different label or mode returns `invalid_params`.
Reconnect to change identity. Label is a trusted LAN identity, not a credential.

Owner mode auto-acquires only unlocked Local. Readonly never acquires, even
with the holder's label. `control.status`, `control.connect`, `control.acquire`
and `control.release` return `{holder, locked, online, me}`. Local is always
online; Remote is online while any initialized socket has its label. `me` is
true only for an owner-mode holder. Disconnect does not release authority or
cancel accepted operations. Acquire refuses `control_locked`, `control_held`
(another label online), or `readonly_connection`. Release requires the holder
and returns unlocked Local. GUI reclaim returns locked Local.

Unmarked methods require holder at owner-loop execution, before domain effects and existing domain guards. `tab.interact` remains unmarked, including reads.
A `not_holder` error includes `{holder, locked, online}`. `operation.await` is
readonly; a new holder can cancel accepted work. Subscription methods only
change connection-local observation subscriptions and remain available to
observers. Measure and Fluxdep share this authority; Dispersive stays readonly.
Cfg commands use current inputs without refs. See ADR-0070 for authority and GUI gates.
## Device setup

`device.setup(name, cfg)` accepts a complete Info constructor cfg and returns
an asynchronous operation ID, mapped to an MCP handle. All model-required
fields must be supplied. Defaults follow the Info constructor, not cached values.
The RPC takes the existing Info's model class and validates cfg without merging.
Missing required fields, extra fields and validator failures return INVALID_PARAMS
before setup starts. YOKO requires mode; omitted value_unit is generated by its
Info model from mode. Explicit conflicting units fail that same validator.
DeviceService retains connected admission, leases, busy checks and driver safety.

Raw callers read device.snapshot, change a complete cfg, call device.setup,
wait on the returned operation, and read device.snapshot for cached facts.
The fixed MCP `device_setup(name, cfg)` tool instead accepts a partial patch,
merges it with before Info and waits up to 300 seconds. YOKO value patches must
explicitly include value_unit even when before Info already contains it.
Output/rampstep-only patches do not need a unit. Timeout does not cancel; the
caller can continue with wait(op). The fixed tool has no timeout parameter.

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
empty reply holder（ADR-0068）。

Only bounded wait handlers run off-main:

- `operation.await`

Off-main handlers only consume thread-safe channels. They do not read or mutate
owner-thread state, version tables, controller objects, editor sessions, or
Qt widgets.

## Events And Diagnostics

Event push is still available on the wire for GUI/internal consumers. Event
payloads use requery hints for complex objects; live Python objects never cross
the wire. Parameter／library 變更事件以 `context.snapshot` 提示訂閱者重新讀取目前
工作單位的 native 文件。這次讀取不改歷史 Run 的固定來源。
`aliases_changed` directs subscribers to `tab.get_cfg` after the
committed global table refreshes current cfg publications. It does not change
historical Run sources. MCP completion still uses operation replies, not events.
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
`current_targets()` is materialized and projected once
only when an `editor_id` subscriber exists. `editor_closed` removes a client's
subscription only after its close push is accepted by that client's queue.

Diagnostics are separate from EventBus. The controller pushes diagnostics to the
remote adapter sink, which broadcasts diagnostic payloads to clients regardless
of subscription. Measure MCP does not subscribe, queue or piggyback events.
Agent-visible async completion comes from operation request/reply, not pushes.

## Version Handshake

The launch/connect note reports three numbers:

- `WIRE_VERSION = 95`: GUI RPC contract. MCP pins and compares this value.
- `GUI_VERSION = 127`: GUI process code revision. It is displayed, not compared.
- `MCP_VERSION = 114`: MCP bridge code revision. The bridge displays this value.

The first two constants live in this package's `wire_version.py`.
`MCP_VERSION` lives in `zcu_tools.mcp.measure.server`.

Only wire-contract changes bump `WIRE_VERSION`. GUI-internal changes that need a
reload signal bump `GUI_VERSION`; MCP-only tool/policy changes bump
`MCP_VERSION`. A GUI wire-contract change bumps both `WIRE_VERSION` and
`GUI_VERSION`.

## Current data and provenance

Reads return current owner projections; they do not establish a per-connection
mutation baseline. Holder admission replaces seen maps and resource-version
comparison. Domain guards, result provenance, busy checks and lifetime still
apply. RPCs are forwarded once, without hidden refresh, retry or rollback.
Timeout or response failure may follow an accepted mutation.

Run accepts current Valid cfg and fixes its values and source basis.
Result-bound `run_operation_id` and analysis `operation_id` protect the original
source from replacement, returning `result_superseded` rather than reading a
new result. These tokens are session-local provenance, not control credentials.
Snapshot counters distinguish replacements, including reloads of the same file.
Run fixes ParameterContext at current cfg acceptance, not live md/ml.
Cfg.dev contains role-to-device names, not settings patches. Run does not precheck
whether those names exist. Core lookup of context.devices[name] reports a missing
name when the device is used, and the error names that device. Cfg format and required
unit validation remain. Holder admission does not validate every cfg.dev name or supply RunMetadata.device_snapshot.
After the hardware lease and before adapter.run, the Run framework separately reads
get_info once from every connected device. Any failure rejects Run without hardware
writes or cache fallback; no devices produce `{}`. Save uses that fixed pre-experiment
snapshot and never re-reads hardware.
Run and existing-tab native Load do not require exporting
active context. Load 在 adapter 呼叫前固定 Load-time selection，並獨立捕捉 Save storage。
它不讀 tab birth 或後來的 selection。同 opened entry 且可定位歷史 point 時，
context owner 按 native 檔案的 point 與 resolved alias names 一次讀取持久 point／library。
這份固定 ParameterContext 供 Analyze／Post／Writeback 使用，是 Load 時的持久值，
不是原始量測的參數版本。跨 entry、EMPTY 或無歷史 point 時，analysis source 為 None；
context-free analysis 仍可用，不回退目前 alias。同 entry 的來源讀取失敗在替換結果前傳遞，
舊結果不變。Load preserves the file's original metadata; it guards tab/result/analysis revisions,
not session focus. Editor commit and writeback retain their captured context lifetime.

`tab.snapshot` reports availability, effective paths and published artifact
state, not raw arrays or writeback contents. Its source operation ID identifies
the Run that published raw data; loaded or empty results have null.
Analysis result queries may bind to the original operation and return actual
result params separately from the next-edit draft. Figure reads and image saves
can also bind to that source. Finite summary values remain unchanged; non-finite
values become null with `invalid` paths and reasons. Generic context reads do
not use this analysis-only projection.

`context.snapshot` reads one published parameter capture: target, point,
library and complete ordered alias rows. Alias resolution failures keep their
error and a null resolved value. Optional summaries/path select projections;
they do not grant write authority. Complex values use the reversible wire tag,
while unsupported values, non-finite numbers or encoding errors fail the read.
Context replies may be large or sensitive.

Native Load restores historical metadata, uses the same application operation
and performs cfg backfill without acquired events or Save confirmation.
Backfill failure retains the result and reports not_applied; failed new-tab
cleanup reports cleanup_failed. Existing-tab Load checks its captured owner
and result lifecycle, not a connection observation. Save rejects entry mismatch
before output I/O; within the same entry it uses a loaded Run's historical point.

## Method Surface

The wire surface is grouped by ownership:

- `entry.list/new/open/info`：entry owner discovery、明確建立／開啟與 cached EntryInfo。New／Open 成功發布空 DRAFT，不選 point、不碰硬體。Open 不建立資源；New 安裝失敗保留 entry，使用 Open 恢復。Metadata reads 不授權寫入。
- `point.list/new/use`：saved points in the opened entry. `point.new` rejects unsafe
  destination labels before source reads or writes, for setup seeds and clones
  alike. Empty strings fail the existing required-string wire validation with
  `invalid_params` and message `'label' must be non-empty`. Direct context calls
  reject them with typed `InvalidInputError`. The entry owner's single path
  segment rule is authoritative for context validation. Shared dispatch projects
  the typed input failure for other unsafe labels as `invalid_params`, with reason
  `point_label_invalid` and no structured data. Rejection preserves selection,
  files, ledger and versions. Missing sources and other resource failures retain
  their existing classification. Existing destination labels also return
  `invalid_params`, with the entry owner's exists message and
  no structured data. Setup and clone collision rejection preserve selection,
  filesystem, ledger and versions. Other I/O is not classified as invalid input.
- `context.*`：published parameter capture, snapshot and point-label queries through
  `ContextControlPort`. `context.snapshot` returns native `{target, point, library, aliases}`
  from the current fixed projection, not a historical Run's parameters. Callers do not use
  old flattened md keys or an ml runtime index. Library editing and template create/list use the app controller;
  `editor.*` provides independent library-entry drafts.
- `soc.*`：real SoC connection and current SoC queries. `soc.connect` requires `kind='remote'`, `ip` and `port`; it rejects mock connections.
- `simulation.initialize`：no-argument asynchronous environment initialization through the same `SetupControlPort` coordinator as GUI **Use Simulate Env**. It disconnects real devices, prepares `fake_flux`, binds its live value to MockSoc and installs a predictor if absent. Repeated initialization preserves a valid simulation and its values. Failure may leave partial changes; inspect state before retrying.
- `device.*`：device connect/disconnect/setup/snapshot through `DeviceControlPort`.
- `predictor.*`：Fluxonium predictor load, edit, clear, and predictions through
  `PredictorControlPort`.
- `tab.*`：tab lifecycle、cfg、run、load與save。`tab.save_artifacts`接受 `all` 或唯一的 `{section,name}` refs，保存完整 artifact，不接受 caller paths。`tab.save_data` 與 `tab.save_artifacts` 同步回傳原 Run 的 run_id，即使全部輸出失敗也保留同一 ID，以及本次完整 paths/status/error，不建立 Save operation id。`tab.snapshot` 投影已提交的 sections/member/output 狀態，不計畫新路徑。Run screenshot 讀 live FigureContainer；analysis/post screenshot 讀目前選中的具名圖。
- `tab.analyze` / `tab.post_analyze`：primary and secondary analysis (analysis owns `analysis` pane; post owns `post_analysis`).
- `tab.interact`：以 `tab_id` 讀 active interactive plugin 的 committed `state`、`commands`、`info`、`figure` 和 `preview_active`；可帶 `payload={command, args}` 執行單一經 ParamSpec 驗證的 command。`done` 為保留命令，丟棄 local preview、完成原 analysis operation；agent 透過 `cancel(op)` 對應 GUI `operation.cancel` 請求取消。`include_figure` 預設 true，figure 是 `{png_b64, bytes}` 或無 widget 時的 `null`。False 完全略過 PNG renderer 並回 figure=null，不改變 command 驗證、state 或原 operation_id。MCP done 用此 receipt 先接住原 operation，再觀察完成及保存；其他 command 與純讀保留預設圖像行為。此 method 的純讀與 command 都要求 holder；同一 owner loop 的較晚提交生效。固定 MCP `tab_interact` tool 轉送一次請求：讀取不切焦點，經驗證的 command 跟隨 Analysis pane。
- `tab.writeback_*`：pane-qualified writeback preview/edit/apply via `(tab_id, subtab_id=analysis|post_analysis)`; draft is opaque, not bound to source context; preview/apply echo `destination_context` (active SessionEnv projection at reply time). Preview accepts an optional `operation_id` to require the original analysis result rather than another run's draft.
- `editor.*`：headless cfg-editor session lifecycle.
- `operation.*`：live operation indexing, bounded wait, domain-owned cancellation and progress.
- `arb_waveform.*`：qubit-scoped arbitrary waveform asset operations.
- `value.*`：read-only session value lookup through `ContextControlPort`.

The `tab.get_figure` subtab locator is required and closed
(`run|analysis|post_analysis`). It may bind a Run preview to
`run_operation_id`, or an analysis preview to `operation_id`, never both.
Omitting both retains the current-pane read. `device.snapshot.unit` reads the cached Info's stored `value_unit`: YOKO reports
A or V, and Fake reports native. Missing Info reports none without inferring from
device type or polling hardware; none is not a usable flux coordinate. Recipe
unit selection matches this snapshot exactly, including native, and rejects a
mismatch before editing cfg. Remote cfg updates may include `value_unit`; Info
validation rejects conflicting mode/unit pairs. The GUI panel does not expose
an independent unit editor. `method_entries/` owns the wire method
name, handler ref, schema, agent exposure, readonly and operation policy.
Adding a wire method requires one entry; MCP receives the projection through
`rpc.catalog` after its version handshake. Descriptions direct the caller to
currently available `rpc_call` methods and to `wait(op)` for asynchronous
terminal status, not removed aliases. No tool inventory is generated from
`MethodSpec`.

## Cfg Editing

`path_resolver.py` projects nominal `SettableTarget` entries for mutations and path
changes. `cfg_observation.py` projects `CfgDraft.observe()` data, never binding
field/editor classes. Setters retain canonical paths and reject legacy aliases.

`tab.get_cfg` returns status, tree, source_basis and diagnostics from the
persistent tab resource. It has no prefix or widget prerequisite. Tree paths are
string arrays. `tab.edit_cfg` decodes the common editing codec and applies one
atomic batch to current inputs. Input errors can include path/edit_index.
The adapter does not retry or publish a successful prefix. Plain strings are typed
strings; __text, __expr, __complex and __ref carry the declared editing intents.
Source publications update all affected cfg before notification.

`tab.reset_cfg` obtains current adapter defaults and publishes a complete
observation, even for unchanged inputs. Reset can publish Invalid. It preserves
results and saved files, adds no unsaved guard, and follows the Run pane only
after success.

`tab.run_start` accepts current Valid cfg without an expected ref.
Domain readiness, SoC/device safety and authentication checks remain independent. Optional `aliases`
maps alias names to concrete components for parameter/library reads, cfg preview
and Run acceptance. Omission, None and {} mean no override and use cached cfg.
Nonempty preview resolves current inputs without publication. Run uses
that same mapping for its cfg and fixed metadata capture. An Invalid publication
cannot run without a valid temporary resolution. Global rows and tab state never
retain these overrides. Active Run
blocks manual edits, reset, replacement and close on that tab; source publications
and edits to other tabs remain allowed. Close becomes available after termination.

`editor.get` and `editor.new` retain the independent draft tree with dotted paths.
Nodes contain kind/path/label/valid/readonly. Sections and active references have named
children, including locked literals. Scalar/literal input and sweep inputs retain
mode/raw/resolved/error/validation_error. Reference nodes include their chosen key,
cached shape label, error, mode inline/ref or null when disabled, and choices.
Ref descendants are readonly; library selection creates ref rather than an override.
The local cfg owner exposes explicit inline conversion, without a new RPC endpoint.
Reads do not resolve sources.
Unknown objects, non-string object keys, and nonfinite numbers fail serialization;
complex numbers use the shared reversible tag, with no string fallback.

Independent editor prefix reads select a node while preserving its full path;
sweep edges and reference keys select their parent node. Unknown prefixes return an empty object. Prefix selection does not grant mutation authority.
Wire keys children/input/inputs are not mutation-path segments.

Scalar values may be direct values, tagged eval values, or tagged value refs.
Eval values store a resolved snapshot at set/lower time. Value refs resolve once
through the session value lookup and then become direct scalars.

Sweep nodes appear as editable subtrees, not as lowered `SweepCfg` objects.
`SweepSpec` exposes `start` / `stop` / `expts` / `step`; `CenteredSweepSpec`
exposes `center` / `span` / `expts` / `step`. Independent `editor.set_field`
accepts dotted edge paths from the editor read model. Tab cfg observations and
`tab.edit_cfg` use string-array paths instead.

`TabCfgResources` owns tab cfg lifetime. Tab creation provides the resource before
any view attaches; close revokes old handles. Qt detach only stops watching.
Load backfill publishes on the same resource.
`CfgEditorService` owns separate library, inspect and writeback drafts.
Agent-created sessions are garbage-collected on commit/discard/client drop;
UI-owned independent drafts follow their own owner lifetime.

## Operation Handles

`tab.save_data` and `tab.save_artifacts` complete synchronously on the GUI owner
thread. They return the current complete Save report with the original Run ID,
even when every output fails, and every output path/status/error. Output statuses
and errors determine save success, not the presence of `run_id`. Readonly tab
snapshots retain confirmed identity: null until a persistent output succeeds.
They do not return a Save handle or require
`operation.await` followed by `tab.snapshot`. Optional Run/Analyze source IDs
still bind the result being saved; they are not Save IDs.

Start methods return GUI-local operation ids on the wire. The GUI projects active
run, analyze and device ids from their owners, regardless of who started them.
MCP assigns session-local opaque integer handles to both started and discovered
operations; a GUI restart can reuse a wire id but cannot reuse an exposed MCP
handle. `operation.await` reads the shared handle channel off-main and rejects
unknown or evicted GUI ids. `operation.cancel` runs on the owner thread and uses
the domain cancel hook; a non-cancellable operation fails with `not_cancellable`.
MCP routes cancellation through `operation.cancel`. Its `cancel(op)` and
`cancel(execution)` share the registered analysis execution's cancellation intent.
The domain-specific wire cancellation methods remain available to other socket
consumers but are absent from its live catalog. MCP `wait(op)` reports the GUI
operation outcome; `wait(execution)` also observes MCP-owned result reads, image
saves and preview delivery. Execution identity and completion policy live in the
[measure MCP adapter](../../../../mcp/measure/README.md), not the GUI operation table.

`soc.connect` returns its summary synchronously, without exposing an operation id.
`simulation.initialize` returns a GUI-local `operation_id`; MCP maps it to a handle
for `wait(op=handle)`. Wait for successful completion before starting measurements.

## Launch / Shutdown

The GUI launcher starts the remote-control socket after the main window exists.
Shutdown flows through the same MainWindow close path as the UI, persists state,
marks the controller shutting down, stops the remote service, and then closes Qt
resources.

The MCP `connect` tool attaches to an existing GUI or explicitly launches one;
exiting MCP disconnects without closing the GUI.
