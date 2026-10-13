# `gui.app.measure.remote` — measure-gui RemoteControlAdapter

**Last updated:** 2026-10-13，library 編輯 wire 改用 Form，editor.* 退休

This package is the GUI-process side of measure-gui remote control. It exposes a
local NDJSON RPC surface over the live `Controller`, injects the application's control identity into shared remote dispatch,
serializes selected events, and pushes diagnostics.

The agent-facing MCP bridge lives in `zcu_tools.mcp.measure`. This package does
not declare MCP tools and does not own stdio transport.

## Layout

- `service.py`：`RemoteControlAdapter`, a measure-gui subclass of shared
  `RemoteControlServiceBase`; injects control authority, and owns
  writeback key aliases and diagnostics.
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
- `path_resolver.py`：flat target projection of an old `CfgDraft`; no wire method uses it.
- `form_observation.py`：tab Form schema/input/values/errors projection and
  the shared path-array edit decoder.
- `cfg_observation.py`：old `CfgDraft` tree projection, now used by no wire
  method, and `cfg_error_to_remote`.
- `load_types.py`：result and analysis Load facts, without Form backfill details.
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
`written.parameter`. Sample previews and write receipts use `kind="sample"`,
`sample-<n>` IDs and a concrete component as target. Apply lists components in
`written.samples`; write receipts preserve complete latest JSONL rows in
before/after, including accepted metadata. Preview exposes unavailable_reasons
for omitted sample proposals. The existing preview/set/apply/write methods own
these projections; no sample-specific RPC is added.

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
owner-thread state, version tables, controller objects or Qt widgets.

## Events And Diagnostics

Event push is still available on the wire for GUI/internal consumers. Event
payloads use requery hints for complex objects; live Python objects never cross
the wire. Parameter／library 變更事件以 `context.snapshot` 提示訂閱者重新讀取目前
工作單位的 native 文件。這次讀取不改歷史 Run 的固定來源。
`aliases_changed` reports a global-table change. Existing Form sources and
accepted Run sources stay fixed. Use explicit Reset to capture new aliases. MCP completion still uses operation replies, not events.
EventBus pushes add process-wide `seq` and `origin={kind, operation_id}` without
changing the event name or payload shape. The per-connection `client_id` is
RPC-side bookkeeping and is deliberately omitted. Diagnostic pushes are not
EventBus broadcasts and retain `{event, payload}`.
Event serializers and NDJSON encoding run only when the endpoint finds a matching
live subscriber. Recipient selection is two-phase: payload construction stays on
the State owner thread and outside the endpoint registry lock, then subscribe state
and link liveness are revalidated immediately before enqueue. One event is built
once for any number of matching clients; unsubscribe/disconnect completed first
cannot receive a late push.
Internal tab interaction/content payloads include closed domain facts used by the
Qt reaction matrix. Their serializers deliberately omit those facts and preserve
the existing event names and `{tab_id, requery}` shape.

Diagnostics are separate from EventBus. The controller pushes diagnostics to the
remote adapter sink, which broadcasts diagnostic payloads to clients regardless
of subscription. Measure MCP does not subscribe, queue or piggyback events.
Agent-visible async completion comes from operation request/reply, not pushes.

## Version Handshake

The launch/connect note reports three numbers:

- `WIRE_VERSION = 102`: GUI RPC contract. MCP pins and compares this value.
- `GUI_VERSION = 135`: GUI process code revision. It is displayed, not compared.
- `MCP_VERSION = 118`: MCP bridge code revision. The bridge displays this value.

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
Run retains the tab's Open/Reset ParameterContext, not live parameters/library.
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
not session focus. Library entry commits and writeback retain their captured context lifetime.

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
error and a null resolved value. Summaries preserve dict trees and scalars,
replacing array/list leaves with descriptions. Path selects one complete value;
neither projection grants write authority. Target is the saved_point locator.
Point and library contain native documents, headers, component fields and
provenance. The library retains persistent Specs without materializing references
or opening waveform assets. Complex values use the reversible wire tag.
Unsupported values, non-finite numbers and encoding errors fail the read.
Context replies may be large or sensitive.

Entry new/open publishes metadata without selecting a point. Parameters remain
None and readiness stays EMPTY until point.use or point.new. Entry metadata,
point.list and setup source binding remain readable in EMPTY. Run, writeback and
current-point Inspect Set require a selected point at application admission.
SoC, device and independent predictor operations retain their own rules.
Writeback keeps its captured SavedPoint after selecting another point in the
same entry; a cross-entry destination is rejected.

Native Load restores historical metadata and results, without acquired events
or Save confirmation. It leaves Form input alone. Replies report loaded-result
and analysis preparation facts. Failed new-tab cleanup reports cleanup_failed. Existing-tab Load checks its captured owner
and result lifecycle, not a connection observation. Save rejects entry mismatch
before output I/O; within the same entry it uses a loaded Run's historical point.

## Method Surface

The wire surface is grouped by ownership:

- `entry.list/new/open/info`：entry owner discovery、明確建立／開啟與 cached EntryInfo。New／Open 成功發布 entry metadata，不選 point、不碰硬體。Open 不建立資源；New 安裝失敗保留 entry，使用 Open 恢復。Metadata reads 不授權寫入。
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
  delete/rename accept concrete three-segment keys and preserve owner errors.
- `soc.*`：real SoC connection and current SoC queries. `soc.connect` requires `kind='remote'`, `ip` and `port`; it rejects mock connections.
- `simulation.initialize`：no-argument asynchronous environment initialization through the same `SetupControlPort` coordinator as GUI **Use Simulate Env**. It disconnects real devices, prepares `fake_flux`, binds its live value to MockSoc. Predictor is not part of simulation readiness. Repeated initialization preserves a valid simulation and its values. Failure may leave partial changes; inspect state before retrying.
- `device.*`：device connect/disconnect/setup/snapshot through `DeviceControlPort`.
- `predictor.predict`／`predictor.calibrate`／`predictor.info`：read-only simulator queries with exactly one source, `qubit` or `params`. A concrete `qubit` such as Q2 reads the currently selected point through the injected params_for. It is not an alias or a path. Direct `params` has `model="fluxonium"` with named EJ, EC, EL, or `model="transmon"` with EJmax, EC, d (junction asymmetry), plus flux_half, flux_period, flux_bias fields. Energies use GHz; affine inputs and device_value use the source's native device units. Returned frequencies use MHz. Info returns `{model, params, missing}` and preserves unknown fields as null. Predict and Calibrate require complete valid inputs. Calibrate returns `{flux_bias_before, flux_bias_after, frequency_mhz, clamped}` without changing the component, dialog or source parameters. Both models choose the smallest absolute native bias across all roots. frequency_mhz is the actual attained MHz. Unreachable targets return the globally closest attainable frequency with clamped=true; ties choose the smallest absolute bias. Reachable targets have clamped=false. Invalid parameters or transitions are rejected, not clamped. Queries never read hardware, dialog/cache or an installed model. No target or load/set/clear commands.

Predictor requests with neither source or both sources return `invalid_params` with reason `invalid_predictor_source`. Invalid concrete names return `invalid_predictor_qubit`; malformed direct inputs or invalid physics return `invalid_predictor_params`. Missing computation inputs return `missing_predictor_params`, a message naming the fields and `data.missing`. Info can inspect incomplete inputs without building a model. A missing component provider or context is a failed precondition, not an empty/default model.

For example, `predictor.predict` accepts `{"qubit":"Q2","device_value":1.80}` or `{"params":{"model":"fluxonium","EJ":5,"EC":1,"EL":0.5,"flux_half":0,"flux_period":1,"flux_bias":0},"device_value":0.1}`. The device coordinate is not flux quanta. Both calculation methods accept from_level and to_level, defaulting to 0 and 1. Calibrate additionally requires frequency_mhz and increasing nonnegative levels. Numerical solver failures propagate; a returned bias does not certify a fit.

- `tab.*`：tab lifecycle、cfg、run、load與save。`tab.save_artifacts`接受 `all` 或唯一的 `{section,name}` refs，保存完整 artifact，不接受 caller paths。`tab.save_data` 與 `tab.save_artifacts` 同步回傳原 Run 的 run_id，即使全部輸出失敗也保留同一 ID，以及本次完整 paths/status/error，不建立 Save operation id。`tab.snapshot` 投影已提交的 sections/member/output 狀態，不計畫新路徑。Run screenshot 讀 live FigureContainer；analysis/post screenshot 讀目前選中的具名圖。
- `tab.analyze` / `tab.post_analyze`：primary and secondary analysis (analysis owns `analysis` pane; post owns `post_analysis`).
- `tab.interact`：以 `tab_id` 讀 active interactive plugin 的 committed `state`、`commands`、`info`、`figure` 和 `preview_active`；可帶 `payload={command, args}` 執行單一經 ParamSpec 驗證的 command。`done` 為保留命令，丟棄 local preview、完成原 analysis operation；agent 透過 `cancel(op)` 對應 GUI `operation.cancel` 請求取消。`include_figure` 預設 true，figure 是 `{png_b64, bytes}` 或無 widget 時的 `null`。False 完全略過 PNG renderer 並回 figure=null，不改變 command 驗證、state 或原 operation_id。MCP done 用此 receipt 先接住原 operation，再觀察完成及保存；其他 command 與純讀保留預設圖像行為。此 method 的純讀與 command 都要求 holder；同一 owner loop 的較晚提交生效。固定 MCP `tab_interact` tool 轉送一次請求：讀取不切焦點，經驗證的 command 跟隨 Analysis pane。
- `tab.writeback_*`：pane-qualified writeback preview/edit/apply via `(tab_id, subtab_id=analysis|post_analysis)`; draft is opaque, not bound to source context; preview/apply echo `destination_context` (active SessionEnv projection at reply time). Preview accepts an optional `operation_id` to require the original analysis result rather than another run's draft.
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

`tab.get_cfg` returns Form schema, authored input, ordinary leaf values and
directed errors. It has no prefix or widget prerequisite. Each value has a
model-field path array and a JSON value. Tuples become JSON lists; complex
values use the public reversible `__complex__` tag.

`tab.edit_cfg` atomically writes unique model-field leaf paths. Expressions use
`{expr: text}`; ordinary strings are literal, including library keys. Range
fields are separate leaves such as start, stop and n. Ordinary invalid input
stays editable with errors. Structural/source rejection leaves the entire input
unchanged. No successful prefix, hidden refresh, retry or rollback is added.

`tab.read_module(tab_id, field)` reads one dotted core module field and returns
detached value/mode/source. Key mode reads the fixed library capture and labels
source as library:<key>. Inline mode evaluates that field's input. Other invalid
fields do not block reads; this module's errors return NOT_VALID. Values exclude
full-cfg validation, native to_cfg and extras lowering. Busy and observer reads
are allowed. Reads do not mutate input, publish or refresh sources.

`tab.use_inline(tab_id, field)` requires holder and resource mutation admission.
The Form owner prepares captured library content or Template/static blank
fallback when library/key is absent. Locks remain applied. The resource applies
one public write batch for required or nullable modules and publishes once on
change. Existing inline input is a no-op. Wrong families, structural/source
failures and busy/lifetime/reentrant guards reject before mutation. The reply is
the complete Form observation; blank input may still need leaf edits.

`tab.reset_cfg(aliases=...)` captures sources once and restores fresh adapter
defaults. The observation may contain errors. Reset keeps results and saved
files, adds no unsaved guard and follows Run only after success. Single-use
aliases change neither global rows nor the memento.

`tab.parameter_read` reads requested paths and module refs from
resource.env.parameters. It takes no aliases. Reads and edits use the same fixed
capture as cfg and Run, until the next Reset.

`tab.run_start` accepts only current valid Form values. It rejects nonempty
aliases before focus or operation effects; use Reset with aliases first.
None and {} use the displayed capture. Domain readiness, SoC/device safety and
authentication remain separate guards. Active Run blocks edits, Reset,
replacement and close on that tab; other tabs remain editable.

`context.ml_edit(kind, name, edits, save_as?)` edits one library entry with
the same path arrays and Form values as `tab.edit_cfg`. The controller opens
the entry form at one fixed capture, writes the whole batch once and evaluates
it. Expressions become numbers from that capture; child library keys stay
keys; the root type/style tag is not editable. Only a valid form is written,
at `save_as` (a new name) or at `name`. The reply is `{valid, errors}` with
Form error paths; invalid values leave the library unchanged. Unknown paths
are `invalid_params` with reason `form_errors`, and values carrying the retired
`__kind` tags are rejected with the `{expr: text}` form before any write.

`TabFormResources` owns tab cfg lifetime. Tab creation provides the resource
before any view attaches; close revokes old handles. Qt detach only stops
watching. Library entry forms and writeback drafts never publish tab Form
input.

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

Entry open/new leaves point selection empty. Select or create a working point before Run, WB or current-point Inspect writes. Unselected application commands fail with existing failed-precondition categories; wire rejection retains the existing dispatch policy. Selection admits WB but never replaces its captured SavedPoint destination. Cfg/editor drafts and WB proposal previews remain available under their own contracts.

Local modal dialogs refuse owner acquisition with `local_busy` and "Local 使用中".
Qt app adapters query the active modal on the owner turn; shared control remains
Qt-free. Readonly and existing Remote holder reconnects keep their contracts.
