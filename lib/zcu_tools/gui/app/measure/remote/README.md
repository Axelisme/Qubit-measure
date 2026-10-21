# `gui.app.measure.remote`

**Last updated:** 2026-10-22

**讀者：** framework-dev。量測 GUI 的 RPC 方法、控制權與來源投影契約。

This package is the GUI-process side of measure-gui remote control. It exposes a
local NDJSON RPC surface over the live `Controller`, injects the application's control identity into shared remote dispatch,
serializes selected events, and pushes diagnostics.

The agent-facing MCP bridge lives in `zcu_tools.mcp.measure`. This package does
not declare MCP tools and does not own stdio transport.

## Owners

- [`service.py`](service.py)：`RemoteControlAdapter`, a measure-gui subclass of shared
  `RemoteControlServiceBase`; injects control authority, and owns
  writeback key aliases and diagnostics.
- [`dispatch.py`](dispatch.py)：runtime method registry projection; keeps the public
  `METHOD_REGISTRY` import path stable.
- [`handlers/`](handlers/)：grouped wire method handlers bound to controller, control facets,
  or render-view calls.
- [`method_specs.py`](method_specs.py)：Qt-free projection for wire method schema and timeouts.
- [`method_entries/`](method_entries/)：single registration source for method name, handler ref,
  `MethodSpec`, agent exposure, readonly and operation policy and `ParamSpec` shorthands. The
  `rpc.catalog` projection uses those same entries; handler refs are resolved
  only by dispatch.
- [`events.py`](events.py)：domain payload type to wire event serializer mapping.
- [`form_observation.py`](form_observation.py)：tab Form input/values/errors projection and
  the shared path-array edit decoder.
- [`form_description.py`](form_description.py)：single-layer Form description and field observation。
- [`form_errors.py`](form_errors.py)：將 measure 的 CfgPreconditionError 轉譯為 remote 錯誤；
  來源 provenance 與 precondition 型別由 services/form_contracts.py 擁有。
- [`wire_version.py`](wire_version.py)：measure-gui wire contract version and GUI code revision.

Shared transport primitives live in [`zcu_tools.gui.remote`](../../../remote/README.md): NDJSON framing,
typed request/reply/error envelope, socket endpoint, owner-scheduler dispatch, and
router scaffolding. Typed envelopes live in [`gui.remote.wire`](../../../remote/wire.py).

## Predictor sources

`predictor.list_files` returns `{files: [filename, ...]}` from the current entry's
predictor owner. Filenames include `.json`, are sorted, and are relative to its
`predictors/` directory. No selected point is required. No entry or absent
directory returns `{files: []}`. Listing does not validate file content, create
directories or write files. Other I/O failures retain unexpected-error diagnostics.
Use a returned filename with `predictor.info`, `predictor.predict` or
`predictor.calibrate`.

`predictor.info`, `predictor.predict` and `predictor.calibrate` require exactly one
non-null source, `filename` or inline `params`. Omission and null both mean no
source. A filename includes `.json` and names a file in
the opened entry's `predictors/` directory. Predictor control owns loading and
validation; remote only projects errors and numeric replies. No point, component,
alias, dialog state or cache participates. Inline inputs require no entry.
Queries and calibration candidates never write predictor files.

Invalid filenames/content return `invalid_params/invalid_predictor_file`.
Unreadable files return `precondition_failed/predictor_file_unavailable`; no entry
returns `precondition_failed/no_entry`. Source conflicts and numeric
parameter errors retain their existing predictor reasons in `method_entries`.

## Wire contract

- One connection has at most one in-flight RPC.
- Request and response roots are JSON objects.

## Fixed element paths

`tab.describe_cfg` 與 `tab.edit_cfg` 接受 str 欄位名與 FixedList 的 int index，例如 `["tones", 1, "gain"]`。Description 的 elements 列出原始 index、captured name 與 typed path，非固定 list 時為 null。數字字串不當成 index。Unknown、inactive、越界與 bool index 由 Form 拒絕，describe 的 form_path 保留完整請求路徑。Read 不改 input，也不抓取來源。Wire 的 values、errors、command target／inputs 都保留 typed paths，不把 index 轉成字串。

## Library creation

`context.ml_create_from_role(role_id, name, edits=...)` 使用 `component.partition.name` 名稱。Module partition 接受 pulses、readout、reset；waveform 使用 waveforms。它先將 Form path/value edits 套入 template input，再驗證及寫入。Blank pulse／readout 不填通道、gain 或時長，使用 edits 補齊必填欄位。Pulse waveform 時長的 Form 路徑是 `["waveform", "module", "length"]`，不是 YAML 的 waveform.length。Mux tone 使用 `["tones", index, "ro_ch"]`。Describe 列出主要必填欄位與路徑。路徑、值或整體 Form 無效時回 invalid_params，data.errors 列出 path/message，library 不變。明填值的 generator／ADC capability 相容性由 Run admission 檢查。

帶佔位符的 role 使用固定 capture 的同名 resolved aliases。例如 qub_probe 要先在 Aliases 綁定 qubit。Name 的 component 不決定 alias；缺必要 alias 時回 invalid_params 並指名要綁定的名稱。`context.set_parameter(path="MG12.module.readout", value="MG12.readout.mux_ro")` 可建立缺失的 module slot，也可用 MG12.module 整份 mapping 寫入。回覆的 before 是原始 slot 值，不 materialize module。

## Form commands

`tab.run_cfg_command(tab_id, path, name)` 呼叫目前 active path 公開的 Form command。Path 是既有 canonical model-field 字串陣列。Description 的掛載欄位 commands 帶 name、label、target 與 inputs，sweep 指令掛在 sweep row 而非座標或採樣葉欄位；remote 只投影，不求值或換算。成功回 input／values／errors，失敗不改 input。Unknown command 回 invalid_params/form_errors，inactive path 回 invalid_params/form_path。控制身份、busy、lifetime 與 mutation guard 沿既有 resource.run_command。任意 provider/declaration ValueError 不被重分類為 command rejection。

方法列在 live catalog，由 `rpc_call` 呼叫。to_inline 專用入口保持原契約。

## Recipe 欄位

`tab.read_recipe_field`、`tab.edit_recipe_fields` 與 `tab.select_recipe_module` 明確使用 typed logical path arrays，str 保留欄位名，int 保留完整 list 的原始位置。數字字串不轉 index，名稱不作位置查找。Handler 只解碼 carrier，由 owner 決定路徑是否可達。`form_description.py` 擁有 RecipeFieldObservation、RecipeEditResult 與 RecipeSelectionResult。所有 description、errors 與 source paths 都是 logical。通用 describe_cfg/edit_cfg/get_cfg/reset_cfg 保持 canonical 契約。真實的 modules 容器不省略，例如 mux readout 的 selector 是 `["modules", "readout"]`，與 recipe 相同。Edit 拒絕時列出目前的 canonical editable_paths 與 selection_paths。Ref 的子欄位要先 use_inline 才可編輯。`tab.read_module` 保留明確 materialization，與專用 `tab.use_inline` 的 field 都使用非空 typed array，str 是欄位名，int 是原始 model-list 位置。兩者經共同 Form owner 解析 typed paths。Unknown、越界或非 index 的數字字串保留完整請求路徑，回 form_path。

`tab.read_elements(path)` 在一次 readonly owner turn 回傳全部元素的 index、name、canonical path、authored input、可用 preview values 與定向 errors。Values 使用完整 canonical typed paths，不是 lowered native cfg。完整 Form errors 另列在最外層，包含 participants 錯誤。某個元素 invalid 不會隱藏其他元素。RPC 只序列化 owner 的 aggregate，不逐元素 describe/read 拼接，不改 picker、focus、capture 或草稿。Busy 與 observer reads 仍可用。可選 `fields` 是元素相對 logical typed paths，例如 `[["drive", "gain"], ["sweep", "start"]]`。每個 row 的 `fields` 按請求順序回同一 owner turn 的 RecipeFieldObservation，包含 FixedElements identity、ref source、value_status 與 errors。省略 fields 保留原 aggregate。錯誤 carrier/path 沿既有 Form 錯誤投影。

Read 使用一次 owner turn 的目前 input 與固定 capture。Value_status 區分 available、missing 與 unavailable，input_present/input 只觀察 authored draft。Selector 描述目標本身，source 識別最近 captured ref。`RecipeFieldObservation.readonly` 在 wire 上仍名為 `readonly`，表示穿過 ref 祖先，不表示宣告唯讀。公開 Recipe helper 的屬性命名由 MCP 層負責，wire 不改名。讀取不改 input、base、cfg version 或通知，也不授權寫入。

Edit 保留 ordered list 與原始 index，owner 一次檢查路徑與 admission。結構、source、宣告 readonly、ref ancestor 或 guard 拒絕不提交前綴。一般 invalid draft 可回 changed=true 加 logical preview errors。Select 回 changed 與同 turn 的 fresh field。Use_library 必須帶非空 actual key，其他 action 必須省略 key；explicit null 不等於省略。

Handler 將 LogicalEditError 投影為 invalid_params 與原 reason，保留 path、edit_index、selector_path、available_keys。FormPathError 保留 form_path，LogicalSourceError 回 precondition_failed/library_source，並帶 consuming path 與來源身分。Cfg/control 失敗沿既有投影。ParamSpec primitive/required/enum 失敗維持原生 reason；handler carrier 與 action/key 語義失敗才回 form_errors。Unexpected provider/declaration failures 不重分類。

## Startup scripts

The launch composition selects one trusted `startup(ctx)` Python file. With no
startup option, measure runs the framework's `default_startup.py`; `--script` replaces that default.
`--clean` runs neither and conflicts with `--script`. Custom code must import and
call the default itself if needed. MCP `connect(script=...)` forwards the choice
only when launching a new GUI. Explicit script options are refused on attach,
and attaching never reruns startup. See [ADR-0064](../../../../../../docs/adr/0064-process-startup.md).

Startup scripts own environment construction. The
[FTF startup example](../../../../../../zcu_lab/simulation_envs/FTF/README.md)
builds and installs resources through the same public facets as other scripts.
RPC readiness does not mean background startup has completed; Run remains locked
until its terminal delivery. Unhandled startup failures retain completed work
and report the script path and line.

Device terminal facts and failed handles remain visible for script-origin
operations. Measure omits only their item error dialog so the default script can
log each failure and publish one status summary. Manual GUI device errors still
show dialogs. Normal window close and RPC `app.shutdown` both drain work, run
script on-close callbacks once in reverse order, then close managed drivers.

## Source refresh

`tab.refresh_sources(aliases=...)` recaptures sources while preserving all authored
input and explicit selections, including expressions and prior direct values.
`tab.reset_cfg(aliases=...)` recaptures and computes fresh defaults. Both return
Form input, values and errors, and neither changes global aliases. Source events
do not refresh existing tabs. Recipe aliases continue through Reset, not Run.
The Form owner defines these rules; RPC forwards them without a second evaluator
or implicit refresh. See [ADR-0065](../../../../../../docs/adr/0065-cfg-editing.md).

## Authority and dispatch

After `wire.version` and optional `auth`, send `control.connect` with
`{label: "agent", mode: "owner"}` (both values default as shown). Before this
command a socket is an observer: explicit readonly methods and `rpc.catalog`
are available, business writes return `not_holder`, and acquire/release return
`readonly_connection`. Initialization fixes label and mode for that socket.
An identical repeat is inert; a different label or mode returns `invalid_params`.
Reconnect to change identity. Label is a trusted LAN identity, not a credential.

Unmarked methods require holder at owner-loop execution, before domain effects and existing domain guards. `tab.interact` remains unmarked, including reads.
A `not_holder` error includes `{holder, locked, online}`. `operation.await` is
readonly; a new holder can cancel accepted work. Subscription methods only
change connection-local observation subscriptions and remain available to
observers. Measure and Fluxdep share this authority; Dispersive stays readonly.
Cfg commands use current inputs without refs. See ADR-0070 for authority and GUI gates.

Normal handlers run on the State owner thread through the injected `OwnerScheduler`.
The Qt composition root supplies `QtOwnerScheduler`; headless tests use
`ManualOwnerScheduler`. Handler exceptions become typed error envelopes.

Only bounded wait handlers run off-main:

- `operation.await`

Off-main handlers only consume thread-safe channels. They do not read or mutate
owner-thread state, version tables, controller objects or Qt widgets.

## Completion and provenance

Reads return current owner projections. Holder admission checks current
control identity at dispatch. Domain guards, result provenance, busy checks and lifetime still
apply. RPCs are forwarded once, without hidden refresh, retry or rollback.
Timeout or response failure may follow an accepted mutation.

Diagnostics are separate from EventBus. The controller pushes diagnostics to the
remote adapter sink, which broadcasts diagnostic payloads to clients regardless
of subscription. Measure MCP does not subscribe, queue or piggyback events.
Agent-visible async completion comes from operation request/reply, not pushes.

Run 完成不保存 raw。RPC 呼叫 `tab.save_data` 才保存 raw；recipe worker 的保存也是顯式呼叫。
`tab.snapshot`、`tab.analyze` 的 start reply 與 source-bound analysis result 的 `operation_state`
提供 `raw_data: {run_id, saved}`。這個 run_id 是固定 Run／Load metadata 的身分，保存前就存在。
Saved 只反映 measurement_data 在 owner tracker 是否全部保存，不查檔案是否仍存在。
保存 analysis 不會把 raw 的 saved 改成 true。原本頂層 run_id 仍只在至少一個持久輸出成功後出現。

`tab.save_data` and `tab.save_artifacts` complete synchronously on the GUI owner
thread. They return the current complete Save report with the original Run ID,
even when every output fails, and every output path/status/error. Output statuses
and errors determine save success, not the presence of `run_id`. Readonly tab
snapshots retain confirmed identity: null until a persistent output succeeds.
They do not return a Save handle or require
`operation.await` followed by `tab.snapshot`. Optional Run/Analyze source IDs
still bind the result being saved; they are not Save IDs.

## Contract references

- [Shared GUI](../../../README.md)：frame/queue limits and response-delivery failures。
- [`method_entries/`](method_entries/)：per-method parameters, replies, timeouts and agent policy。
- [`handlers/`](handlers/)：wire payload and error projection。
- [`wire_version.py`](wire_version.py)：wire handshake and version bump rules。
- [ADR-0070](../../../../../../docs/adr/0070-control-identity.md)：control identity and GUI gates。
- [ADR-0068](../../../../../../docs/adr/0068-remote-transport.md)：transport、expected errors、events 與 GUI／MCP 分工。
- [ADR-0065](../../../../../../docs/adr/0065-cfg-editing.md)／[ADR-0008](../../../../../../docs/adr/0008-cfg-editor-session.md)：Form 與 writeback。
- [ADR-0066](../../../../../../docs/adr/0066-operation-lifecycle.md)／[ADR-0063](../../../../../../docs/adr/0063-persistence-ownership.md)：operation、shutdown、provenance 與 persistence。
- [Measure framework](../README.md)／[MCP bridge](../../../../mcp/measure/README.md)：application owners 與 agent-facing policy。
