# `gui.app.measure.remote`

**Last updated:** 2026-10-25

**讀者：** framework-dev。量測 GUI 的 RPC 方法、控制權與來源投影契約。

This package is the GUI-process side of measure-gui remote control. It exposes a
local NDJSON RPC surface over the live `Controller`, injects the application's control identity into shared remote dispatch,
serializes selected events, and pushes diagnostics.

The agent-facing MCP bridge lives in `zcu_tools.mcp.measure`. This package does
not declare MCP tools and does not own stdio transport.

Record writeback uses kind=`record`, IDs `record-<n>` and `written.records`.
The preview's `proposed_value` carries the adapter's JSON object unchanged.
Wire version 123 removes the old sample names without aliases. Component records
live at `records/components/<target>.jsonl`; the event ledger stays at `records/ledger.jsonl`.

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
- [`form_observation.py`](form_observation.py)：tab Form input/values/errors、全表 diagnostics 投影，以及共用 path-array edit decoder。
- [`form_description.py`](form_description.py)：single-layer Form description and field observation。
- [`form_errors.py`](form_errors.py)：將 measure 的 CfgPreconditionError 轉譯為 remote 錯誤；
  來源 provenance 與 precondition 型別由 services/form_contracts.py 擁有。
- [`wire_version.py`](wire_version.py)：measure-gui wire contract version and GUI code revision.

Shared transport primitives live in [`zcu_tools.gui.remote`](../../../remote/README.md): NDJSON framing,
typed request/reply/error envelope, socket endpoint, owner-scheduler dispatch, and
router scaffolding. Typed envelopes live in [`gui.remote.wire`](../../../remote/wire.py).

## Saved results

`results.list()` returns `{runs: [...]}` from a fresh project-wide Database
snapshot through `Controller.query_results() -> ResultsIndex`, the same query
used by the results browser. It needs neither an opened entry nor a selected
point and does not switch either. Missing Database returns an empty list.

Each run includes opaque `key`, `entry`, historical `run_id`, `experiment`,
`point`, captured `roles`, `at`, `registered`, absolute `run_path`,
nullable `data_path`, `figure_paths`, `analysis_paths`, `figures`,
`search_text` and `issues`. Missing historical identity stays null.
`figures` lists exact PNG filename selectors. `search_text` includes tone names
from cfg/analysis even when roles captured only one mux component. Full cfg,
analysis documents and arrays are not returned. The caller filters this list;
the browser's display filters do not change the common source.

`results.figure(run_key, figure)` uses that opaque key and one exact filename
from `figures`. It returns `{mime_type: "image/png", png_b64: "..."}` with
the saved bytes, without rendering or analysis. Arbitrary paths are not accepted.
Unknown runs/selectors return `invalid_params` with `unknown_result_run` or
`unknown_result_figure`. Escaping paths and I/O failures retain shared dispatch's
structured errors. There is no retry.

Both methods are readonly and available to observers in the live catalog through
`rpc_call`; they do not add fixed MCP tools. Every call rescans the disk.
Membership and content can change between list and figure. Neither method
creates files or locks, writes results, or changes entry/point selection.

## Captured writeback

`tab.writeback_preview` 回 owner 當下的 draft 候選，不帶 revision。
`tab.writeback_accept(tab_id, stages)` 接受有序的 `{subtab_id, ids}` 清單，不帶 `expected_revision`。
Owner 在同一次 dispatch 核對 Primary／Post 全部選定 IDs，unknown／duplicate 在第一筆寫入前拒絕。
通過後按 Primary、Post、各 pane 的候選順序寫入 owner 當下 draft 的值與目的地，不還原捕捉時的 proposal。
接受回覆 `{status, written, error}`，written 以 pane 與穩定 item ID 記錄已確認成功前綴。
先呼叫 `tab.writeback_begin(tab_id)` 取得 owner 配置的一次性 `submission_id`，再帶到
`tab.writeback_write` 或 `tab.writeback_accept`。Begin 不代表已寫入，同 ID 重複提交會拒絕。
`tab.writeback_receipt(submission_id)` 唯讀回 `{submission_id, tab_id, status, completed, error}`。
Status 是 pending、completed 或 failed。Pending 包含只 reserve 或仍執行，不能推論沒有副作用。
Completed 的每列是 `{subtab_id, item:{id, kind, target, value, destination}}`。
Destination 是固定的 `{kind: "saved_point", entry_id, point}`，value 是 owner 當次確認的實際值。
查詢不重讀當前 draft 或磁碟，僅保留 GUI 生命週期的記憶體 receipt。
10 秒 handler 逾時不表示寫入失敗。Caller 先查該 ID 的狀態，再決定是否重送。
Recipe receipt.items 是成功 ID 對應的原捕捉候選，實際值與目的地仍取提交當下的 draft。
GUI／MCP 的所有權模型允許問答期間的 draft 編輯與替換，見 ADR-0075 的 2026-10-23 使用者裁定。
第一次寫入失敗就停止，不 retry／rollback。單 pane `tab.writeback_write` 仍編輯當前 draft。

## Workflow namespace

`workflow.open/close/start/status/form/describe_form/edit_form/tunables/pause/resume/stop/cancel/records/figures/figure/answer`
call the same headless `WorkflowControlFacet` as Qt panes. Commands do not require
any GUI mode or selected tab. When the GUI mounts a newly opened owner, it selects
the new tab and switches to Workflow mode. `tab_id` names an opened draft owner;
`run_id` names a native run. Status without run_id lists catalog names and all opened tab
snapshots. Start/Resume operation_id uses the existing measure operation owner;
submission does not prove native acceptance. MCP exposes the same operation as
an opaque handle for wait.

Readonly methods allow observers, including batch previews. Mutations use the
shared holder check and the owner's busy/Form/RUN admission. Preview and
single-layer description use one owner candidate and fixed capture. The remote
only encodes input, ordinary values/errors and metadata. Start omitted/null
batches read shared drafts; explicit batches use fresh captured defaults. Apply
with [] submits current tunables draft, not defaults.

Native run mismatch and lifecycle errors preserve reason/data. Form paths keep
form_path, Form refusal keeps form_input_rejected, and invalid Start/Apply
returns form_errors with path/message rows. Workflow form edits/previews and
`tab.edit_cfg` reject malformed model structures as invalid_params/form_structure;
`data.form_path` identifies the consuming canonical field and the message lists
the expected object/tag shape. Unexpected producer failures retain
controller_error. No retry or rollback follows an error or transport timeout.
Figures and committed records remain readable after terminal delivery. Names
and status do not render; figure returns png_b64 for validated MCP image delivery.
Figures use the pane's same WorkflowPlots projection: workflow names stay
unchanged, current-operation names are operation/<name>. A new attempt replaces
the whole operation group; terminal delivery retains the latest group.
Status includes `pending_question`, a
`{request_id, question_type, question, answer_schema}` envelope or null outside
Ask. Native status also projects pending_request_id, active_path and live_paths.
Active_path is the root-to-first-fork chain; live_paths is a structured list of
all live leaf paths, never a comma-joined string.
`workflow.cancel(run_id, scope, expected)` returns the existing snapshot plus
cancelled/reason/active_path/candidates from the shared resolver. Stop cancels the workflow root. Answer requires run_id,
request_id and answer, and returns the original operation_id. Engine matches the
ID before validating fields. Stale or absent questions return
`precondition_failed/question_mismatch` with current pending_request_id or null.
Invalid fields return `invalid_params/answer_validation`. Neither rejection
consumes the question or records an accepted answer.

Status retains `last_exchange` with the latest question envelope, accepted
answer fields or null, and waiting/submitted/answered/cancelled state. Host
acknowledgement proves answer validation and delivery, not run completion.
Pause, Resume and Answer retain one worker and RUN lease through terminal owner
delivery; RPC/UI code never executes the author. Committed records project their
directory, UTC commit time, read-order revisions, qualified model name and
actual operation-attempt facts alongside the encoded authored record.

## Run scope control

`operation.cancel_scope(operation_id, scope, expected=None)` targets the exact
active GUI Run handle, alongside the unchanged generic `operation.cancel` hook.
`operation.status_scope(operation_id)` returns active_path and structured live_paths.
An inactive Run returns an empty path and an empty list. Cancellation of an inactive handle returns
scope not active and never targets the next Run. The Run owner and workflow host
reuse the experiment Scope resolver and CancelResult. Selectors include kind,
kind#N, operation:<name> and exact full leaf paths. Multiple matches refuse with
reason="ambiguous" and full-path candidates; kind#N only resolves same-parent
sequences. Cross-parent duplicates still require an exact full path.

The GUI path starts with run. Recipe projects only its schedule/acquire tail
onto the local recipe:<name>/operation#N path. Composite live_paths and cancel
ack candidates omit the GUI run root by the same rule. GUI acceptance and refusal
reasons remain unchanged. Handle and sequence checks are process-local; the combined path
is not an atomic cross-process observation. Descendant cancellation marks Run
metadata stopped through partial_result, without changing run Stop intent.
`tab.snapshot.result_state.partial_result` carries the fixed completion fact even
without native metadata. New Run and result replacement clear the old fact.
Recipe RunResult reads this field, not operation terminal status or cancel ack.

Analyze work may explicitly raise AnalysisFailed. Only that work failure adds
failure_kind="analysis" to operation.await; recording/postprocessing errors stay
unclassified. Recipe converts the analysis classification to a failed value,
while arbitrary errors still raise. Root cancellation uses ScriptCancelled after
drain and does not overwrite genuine errors.

## Analysis presentation

Operation-bound `tab.get_analyze_result` and `tab.get_post_analyze_result`
include `presentation: {run_id, stage, figure_names, figures_replaced}`.
Run identity comes from the selected Run/Load metadata, not the live point or
Save receipt. Missing metadata yields null. Stage is primary or post.
Figure names are the native names in that pane's current collection.
`figures_replaced` records whether this successful publication replaced a
nonempty collection in the same Run and pane. It does not report disk writes.
New Run/Load clears this history; Primary replacement also invalidates Post.
An old operation-bound read rejects with `result_superseded` after replacement.
Unbound summary reads retain their summary/invalid shape.
Element names and original positions belong to the experiment's summary;
the remote does not infer element identity from figure names or row offsets.

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

## Shutdown receipt

`app.shutdown` checks idle, unsaved-data policy and view availability before
acceptance. Shared dispatch binds its owner-thread close continuation to the
reply's terminal send. A blocked writer does not start shutdown. A complete
success frame sent to the OS allows close; this is not a peer acknowledgement.
Send, disconnect and enqueue failure also allow the accepted shutdown to close,
and logs distinguish delivered from failed. A missing receipt does not prove
that shutdown was rejected. The wire reply and existing timeout policy stay
unchanged. Qt's deferred close turn avoids reentrancy, not socket delivery races.

## Wire contract

- One connection has at most one in-flight RPC.
- Request and response roots are JSON objects.
- `value.read` reports unregistered keys as `invalid_params` with `reason="unregistered_source"` and `data.path`; other source failures keep their existing error categories.

## Fixed element paths

`tab.describe_cfg` 與 `tab.edit_cfg` 接受 str 欄位名與 FixedList 的 int index，例如 `["tones", 1, "gain"]`。Description 的 elements 列出原始 index、captured name 與 typed path，非固定 list 時為 null。數字字串不當成 index。Unknown、inactive、越界與 bool index 由 Form 拒絕，describe 的 form_path 保留完整請求路徑。Read 不改 input，也不抓取來源。Wire 的 values、errors、command target／inputs 都保留 typed paths，不把 index 轉成字串。

## Library creation

`context.ml_create_from_role(role_id, name, edits=...)` 使用 `component.partition.name` 名稱。Module partition 接受 pulses、readout、reset；waveform 使用 waveforms。它先將 Form path/value edits 套入 template input，再驗證及寫入。Blank pulse／readout 不填通道、gain 或時長，使用 edits 補齊必填欄位。Pulse waveform 時長的 Form 路徑是 `["waveform", "module", "length"]`，不是 YAML 的 waveform.length。Mux tone 使用 `["tones", index, "ro_ch"]`。Describe 列出主要必填欄位與路徑。路徑、值或整體 Form 無效時回 invalid_params，data.errors 列出 path/message，library 不變。明填值的 generator／ADC capability 相容性由 Run admission 檢查。

帶佔位符的 role 使用固定 capture 的同名 resolved aliases。例如 qub_probe 要先在 Aliases 綁定 qubit。Name 的 component 不決定 alias；缺必要 alias 時回 invalid_params 並指名要綁定的名稱。`context.set_parameter(path="MG12.module.readout", value="MG12.readout.mux_ro")` 可建立缺失的 module slot，也可用 MG12.module 整份 mapping 寫入。回覆的 before 是原始 slot 值，不 materialize module。

## Form commands

`tab.get_cfg` 與 `tab.describe_cfg` 的 `diagnostics` 都涵蓋完整表單，包括未參與或未在 picker 顯示的 tone。每列保存 canonical path、目前 message，以及原始 name／index。Seed_source 是初值 expression、slot、template 或作者來源說明。Seed_status 的 missing／available／literal 是 Open／Reset 的事實，Refresh 和手動補值不改寫它。Uncalibrated 只投影作者的 seed metadata，未宣告則為 null，不從缺值或數值推論。全表 diagnostics 合併 preview 與 lowered cfg errors，describe 的局部 errors 保持原契約。Owner 為 `Form.diagnostics` 與 `TabFormResource.diagnostics`，remote 不另建缺值規則。

`tab.run_cfg_command(tab_id, path, name)` 呼叫目前 active path 公開的 Form command。Path 是既有 canonical model-field 字串陣列。Description 的掛載欄位 commands 帶 name、label、target 與 inputs，sweep 指令掛在 sweep row 而非座標或採樣葉欄位；remote 只投影，不求值或換算。成功回 input／values／errors，失敗不改 input。Unknown command 回 invalid_params/form_errors，inactive path 回 invalid_params/form_path。控制身份、busy、lifetime 與 mutation guard 沿既有 resource.run_command。任意 provider/declaration ValueError 不被重分類為 command rejection。

方法列在 live catalog，由 `rpc_call` 呼叫。to_inline 專用入口保持原契約。

## Recipe 欄位

`tab.select_recipe_option(tab_id, path, option)` 讓 owner 依精確 Form option label 或 captured module key 選擇。option 必填，null 只可選 nullable 欄位。Inline labels 走既有 converter。`tab.run_recipe_command(tab_id, path, name)` 在 logical path 執行可用命令，to_inline 明確複製固定來源。兩者回 changed，合法不完整草稿仍可保留；readonly、來源、ref 祖先、holder 與 busy 限制仍生效。未知或歧義選項不 fallback。

`tab.check_recipe_ready(tab_id)` 是 readonly 呼叫，以固定來源驗證完整 Form 與 lowered cfg。它回全部 logical path/message errors，不 mutation、不 Run、不重捕捉來源。Recipe 層把 errors 轉為 needs_parameters，不把這個檢查做成空 mutation batch。


`tab.read_recipe_field` 與 `tab.edit_recipe_fields` 明確使用 typed logical path arrays，str 保留欄位名，int 保留完整 list 的原始位置。數字字串不轉 index，名稱不作位置查找。Handler 只解碼 carrier，由 owner 決定路徑是否可達。`form_description.py` 擁有 RecipeFieldObservation、RecipeEditResult 與 RecipeSelectionResult。所有 description、errors 與 source paths 都是 logical。通用 describe_cfg/edit_cfg/get_cfg/reset_cfg 保持 canonical 契約。真實的 modules 容器不省略，例如 mux readout 的 selector 是 `["modules", "readout"]`，與 recipe 相同。Edit 拒絕時列出目前的 canonical editable_paths 與 selection_paths。Ref 的子欄位要先用 `tab.run_recipe_command` 的 to_inline 才可編輯。

`tab.read_elements(path)` 在一次 readonly owner turn 回傳全部元素的 index、name、canonical path、authored input、可用 preview values 與定向 errors。Values 使用完整 canonical typed paths，不是 lowered native cfg。完整 Form errors 另列在最外層，包含 participants 錯誤。某個元素 invalid 不會隱藏其他元素。RPC 只序列化 owner 的 aggregate，不逐元素 describe/read 拼接，不改 picker、focus、capture 或草稿。Busy 與 observer reads 仍可用。可選 `fields` 是元素相對 logical typed paths，例如 `[["drive", "gain"], ["sweep", "start"]]`。每個 row 的 `fields` 按請求順序回同一 owner turn 的 RecipeFieldObservation，包含 FixedElements identity、ref source、value_status 與 errors。省略 fields 保留原 aggregate。錯誤 carrier/path 沿既有 Form 錯誤投影。

Read 使用一次 owner turn 的目前 input 與固定 capture。Value_status 區分 available、missing 與 unavailable，input_present/input 只觀察 authored draft。Selector 描述目標本身，source 識別最近 captured ref。`RecipeFieldObservation.readonly` 在 wire 上仍名為 `readonly`，表示穿過 ref 祖先，不表示宣告唯讀。公開 Recipe helper 的屬性命名由 MCP 層負責，wire 不改名。讀取不改 input、base、cfg version 或通知，也不授權寫入。

Edit 保留 ordered list 與原始 index，owner 一次檢查路徑與 admission。結構、source、宣告 readonly、ref ancestor 或 guard 拒絕不提交前綴。一般 invalid draft 可回 changed=true 加 logical preview errors。`require_valid=true` 先以固定來源驗證 detached candidate 與 native cfg，錯誤回 changed=false 與完整 logical errors，不安裝或發布候選。省略或 false 保留一般 invalid draft 行為。空 batch 也可要求驗證目前 cfg。Select 回 changed 與同 turn 的 fresh field。

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

## Point-file reload and guarded writes

`entry.info.files` lists the selected point's absolute parameters, library and
aliases paths and their installed SHA-256 revisions. Read this base before
editing a file. Edit that path, then call `context.reload` with its path and the
previously installed `expected_revision`, not the edited disk hash. Supply the
key explicitly. Use null only when the aliases base has no file.

```json
{"method": "context.reload", "params": {"path": "/project/results/entry/points/base/aliases.yaml", "expected_revision": null}}
```

The owner validates and installs the candidate, publishes source changes and
returns `{document, path, revision}`. It never rewrites or repairs the edited
file. Invalid paths or contents return `invalid_params` with the path and
reason. A stale installed base returns `precondition_failed` with
`reason=revision_mismatch`. A non-holder receives the existing `not_holder`
rejection before domain effects. Failures preserve the installed state.
Existing tabs keep their fixed captures until an explicit Refresh or Reset.

`context.set_parameter`, `context.aliases_set`, `context.ml_edit` and
`context.ml_create_from_role` accept optional `expected_revision`.
If supplied, the session owner compares it with that document's current disk hash
before writing. Explicit null checks for
a missing aliases file; parameters and library reject null. Omission keeps the
original unguarded writer contract. These checks are optimistic, without a
cross-process lock or a multi-file transaction.

Recipe and startup code still use the context owner facets. Startup must select
or create a point before `ctx.context.apply_alias_rows(rows)`. That command
validates and atomically saves the selected point's aliases before installing
or publishing them. No extra save or subsequent reload is needed.

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
readonly; a new holder can cancel accepted work. Measure and Fluxdep share this authority; Dispersive stays readonly.
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

Post start receipts include `result_read_timeout_seconds` and a source-bound
`recovery` RPC envelope. The owner budgets 5 seconds plus 5 per measured tone,
capped at 60 seconds. The Post result handler ceiling is 60 seconds.
After a result-read timeout, read `tab.snapshot` and compare
`post_analysis_state.source_operation_id` with the recovery operation ID.
Use the supplied `operation_id` with `tab.get_post_analyze_result`; a replaced
pane returns `result_superseded`, including replacement after the snapshot.
These reads never rerun analysis. Both analysis pane snapshots expose their
native `source_operation_id`; MCP operation handles are separate identities.

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
