# `zcu_tools.gui.app.measure`

**Last updated:** 2026-11-01, neutral run metadata and GUI run stop terminal

**讀者：** framework-dev。量測 GUI 的組裝、分頁、tab admission 與 Run／Load／Save 分工。

`gui.app.measure` 是 measure-gui 的 app framework。它負責 tab lifecycle、cfg
editing、context/SoC/device/session wiring、run/analyze/save/writeback workflow、Qt
view 與 GUI-side remote handler。使用者套件的 adapter 擁有實驗領域知識；
framework 接收 caller 注入的 `Adapter` 值。
RunAnalyzeControl 對 MCP 結果讀取使用固定 60 秒 budget，不猜 opaque result 結構。
Adapter 的 `make_form` 從固定 `FormContext` 產生 fresh input。
分析與 Post 的結果／參數槽位預設 None，表示沒有數值或輸入。
`AnalysisPaneSnapshot.has_result` 與 `PostAnalysisPaneSnapshot.has_result` 交付 State 的結果存在事實。
Figures-only 與已發布的空 FigureCollection 都算有結果，不靠 numeric payload 判斷。
UI 與 remote 讀取這兩個欄位，不從 admission 推回結果是否存在。
None 結果產生空摘要與零 writeback 候選，圖與空摘要沿用既有保存格式。

## Immutable definition values

[`adapter/value.py`](adapter/value.py) defines four frozen, keyword-only values.
`Adapter(spec, make_form, run, guide, analysis=None, post_analysis=None)` connects
native persistence, captured Form construction and adapter-owned acquisition.
`Analysis(fn, options=None, writeback=None)` and
`Interactive(plugin, frontend, options=None, writeback=None)` occupy the same
primary slot. `PostAnalysis(fn, options=None, writeback=None)` declares the
independent second stage. Authors do not declare capability flags or artifact hooks.

Adapter's seven generic slots connect experiment cfg, raw Result, primary
options/result, post options/result and adapter input. The input slot defaults
to the experiment cfg type. The four analysis slots default to None. Keyword-aware
Protocols in [`adapter/hooks.py`](adapter/hooks.py) retain `context=` and `plots=`.
Analysis and writeback callbacks retain their existing typed request envelopes.

Primary options have signature `(source, env) -> O`; post options also receive
the adopted primary result. The returned object's type supplies the form and
JSON schema. None means no options form, not a missing analysis capability.
The host generates once when preparing a new source or primary, retains the
object for schema queries, and passes edited options to analysis. Existing lab
wrappers may still read historical parameters during analysis. Their cleanup is
a separate task.

The private primary-stage base hides only Interactive's state type S.
Interactive passes its typed plugin to the host's generic SessionFactory, then
closes over that matching session in a frontend mount callback. The host owns
session lifetime and operation settlement. The UI still validates, mounts and
cleans up the returned widget. The contract imports no widget or controller.

Registry stores values and performs its one heterogeneous generic erasure at
registration. Capabilities derive from the primary value and post declaration.
Every value requires SoC for Run and permits a Load attempt; restore compatibility
is decided by the declared spec. `restore`, `data_item` and `analysis_item`
provide framework projections without I/O. Guide reads its declared Markdown
only when called. Record proposals contain adapter-owned JSON, with no coordinate declaration.

Each adapter declares `run(inp, *, context) -> RunRecord`, converts its input
to the actual experiment cfg and invokes the experiment. Even without a
conversion, the adapter declares a local run returning that record. RunService
accepts the record and attaches the JSON dump of RunMetadata. It does not wrap a second
record or replace the returned cfg, including cfg=None, with Form input.
Save and analysis consume the record's experiment cfg and result. Load restores
that record through spec and never fills Form from historical cfg. Analysis and
writeback do not read current Form input. No adapter class compatibility path
is provided.

Analysis results are dataclass instances, pydantic models or string-keyed mappings.
They need no serializer or result base class. `summary.summarize_analysis(result)`
uses declared field names and converts arrays to lists and complex values to
`{real, imag}` objects. Non-finite values become null. Unsupported fields or root
values raise TypeError with their summary path.

Save, live RPC and MCP use this converter without translating keys. Live RPC
reports non-finite paths in its existing `invalid` envelope. Historical analysis
documents retain only pane, parameters and the object summary; reopening saved
results shows null without the live invalid list. A completed figures-only None
result has an empty object summary. Before completion, RPC returns summary=null.

## Spectrum range selection

`range_find_plugin.RangeFindPlugin` captures numeric MHz/magnitude arrays and
caller-supplied component labels. It uses `analysis.dips.find_spectrum_dips`
for both the initial guess and parameter changes. Manual selection snaps to
measured samples. Assigning a component to another dip clears its old assignment.
Changing parameters replaces manual selections and retains assignments only at
surviving frequencies. `max_dips` limits automatic selection, not manual edits.

`RangeFindState.options` contains the final numerical controls.
`RangeFindState.selected` contains frequency-ordered points with optional
`component` and `estimate`. A manual point outside the current candidates has
`estimate=None`. The adapter's terminal result builder must retain this state
in its result if the analysis must be reproducible. The host saves that result
under the historical analysis document's `summary`; its `parameters` field
continues to describe the initial Analyze form. No live form or lab config is read.

The Qt frontend and `tab_interact` use the same actions. Plugin commands are
`set_parameters`, `add_dip`, `remove_dip` and `assign_dip`; the host owns `done`.
The command schemas describe their arguments. Dip indices are zero based in
current frequency order. Component null clears an assignment. No selections
rejects Done without closing input.

## Owners

- [`adapter/`](adapter/value.py)：四種定義值、typed callback hooks、request/result 與分析參數。
  `spec.tag` 提供 experiment tag。`restore` 將已讀 Container 與 spec 交給 `restore_run`，不建構 core。
  Writeback 型別由 `writeback/` 擁有，adapter 公開入口再匯出。ModuleWriteback proposal 直接使用持久 ModuleSpec。
- [`services/`](services/app_services.py)：app service layer。Run、Analyze、PostAnalyze、Load 與 Tab 將 pane 提交及退役資源清理交給 `pane_commit`。
  GuardService 統一 domain admission，啟動時的 hardware exclusion 由 gate 裁決。
  各 operation 決定終局與觸發清理的時機。Run 與 Load 不依賴 WritebackService 來清理 draft。
  Analyze 回傳就發布結果，拋錯就 failed 並顯示原始訊息，不辨識例外型別。
  Service 透過 ports 與其他 services 合作；package `__init__` 只做 lazy public re-export。
  `services.results_index.scan_results(project_root)` 供結果瀏覽器與 RPC 共用唯讀 `ResultsIndex`。它掃描各 Database entry 的 run 目錄，從 datafile.read_doc 讀保存 metadata，再以同名 Result entry 的 acquired 事件提供歷史身分，缺 ledger 仍列為未登錄。瀏覽投影容許 metadata 缺欄與額外欄位，不以目前 RunMetadata 型別擋住歷史內容；壞文件保留 run 與 issues。Cfg JSON 原樣讀取，不查 catalog 或驗證實驗 schema。Cfg／analysis 搜尋包含未出現在 roles 的 tone 名稱。文件與圖檔讀取只接受索引中的 run key 與成員檔名，讀取前重驗根目錄與 symlink 邊界。索引不建立檔案或 lock，也不改 opened entry。
  `services.results_comparison.build_comparison(index, run_keys)` 將保存的 summary JSON 投影為逐 run 的比較列。身分四欄後接葉節點點路徑，欄位依釘選與 JSON 順序取 union。物件 list 以索引多行呈現，不辨識實驗名稱或特定 key；parameters 不進表。檔案與 JSON 讀取仍委派 `ResultsIndex`，不換算單位或重新分析。
- [`state.py`](state.py)：保存 tab/device/pane/path/version-table SSOT，限定在 owner thread 修改。
  每種 pane 只有一個被動交換入口，安裝備妥的 pane 與 artifacts，回傳退役資源。
  `preview_pane_transition` 與 Analysis/Post swap 共用退役選擇，回傳具名失效事實。
  查詢不改版本、artifacts、draft 或 plots，service/control 只轉接，remote 只投影 wire 字串與順序。
  State 不規劃 artifacts、不執行退役清理，也不在交換時遞增版本。
  Analysis 與 PostAnalysis 各自保存 `WritebackDraft | None`，None 表示沒有 draft。
- [`services/pane_commit.py`](services/pane_commit.py)：提交 Run、Analysis、PostAnalysis pane。
  它先規劃 artifacts，再交 State 交換，遞增版本，最後關閉退役 draft 並釋放 plots。
  規劃失敗保留舊 pane。清理失敗逐項記錄，不回滾已提交結果，也不跳過其他資源。
  Run 提交清空 Analysis 與 PostAnalysis，Analysis 提交清空 PostAnalysis，PostAnalysis 提交保留 Analysis。
  Tab 移除後也由此 module 清理退役資源。未發布的 plots 由 `discard_unpublished_plots` 結束產生作業，不呈現最後一幀，並保證釋放 attachments。
- [`artifacts/`](artifacts/handover.py)：tracker、SaveService、SaveControl、保存中心與 `ArtifactHandover`。
  保存中心在全部 artifact 都是 SAVED 時停用 Save All 與各列 Save。
  任一 artifact 不是 SAVED 時恢復啟用，但仍須符合 idle、可寫與 active context 條件。
- [`writeback/`](writeback/service.py)：parameter／module／record proposals、draft、service、control 與 Qt review widget。
- [`ui/`](ui/main_window.py)：Qt widgets、MainWindow top-level façade、capability-driven `ExpTabWidget`、
  feedback widgets；writeback view 由 `writeback/` 擁有，generic cfg form不屬於app package。
  Footer 訊息在可用寬度內換行，不以長錯誤文字撐大視窗；tooltip 與 view snapshot 保留完整文字。
- [`remote/`](remote/README.md)：與 `ui/` 平級的 GUI-process driving adapter，將 RPC 意圖交給
  application commands/queries。`remote.method_specs` public import path 不載入
  Qt-bound service code；MCP bridge 不在本 package。

resources.entry executor 與 remote handler／wire types 留在各自 owner。
App composition 協調 Artifact 與 Writeback。
`measure-artifacts-no-writeback` 與 `measure-writeback-no-artifacts` contracts 約束兩個方向，
允許共用 State／events／context。

## Composition root

`MeasureGuiBehavior` 組裝後啟用同一個已安裝文件 watcher。它監看作用中三檔，選 point 時另監看仍由 Inspect 編輯的 entry 層文件。Controller 從目前 holder 注入 admission：Local 僅發布 pending，Remote 才自動重載。Authority 改變時重查 pending 檔案。Controller.close 先解除 watcher 與 authority listener，再 drain 背景工作，避免關閉後重載。

`MeasureGuiBehavior` 明確接收 lab 與 keyword-only 程序設定。
它依本 package 的 `lab.MeasureGuiLab` Protocol 消費 lab，不 import 使用者套件。
建構時先呼叫 `lab.register_components`，再建立含 `lab.templates` 的 Registry，
交給 `lab.register_experiments` 填入 adapters。
`lab.catalog_reload_policy` 提供 source packages 與 reload／preserved 模組邊界。
Frozen value `CatalogReloadPolicy` 由 `gui.app.measure.lab` 擁有，lab 與合成測試物件共用此型別。
Behavior 在掛載 widgets 前，以 `lab.workflow_attachments` 設定同一 workflow facet。

Behavior 組裝 `State`、`Controller`、`MainWindow`、persistence caretaker 與 app-local `RemoteControlAdapter`。
Logging、rendering、`QApplication`、control options 與 exit code 留在 shared `gui.runtime`。
`apps.measure` 選定程序參數，以 `partial` 建立 zero-arg behavior factory。
`launch_gui_runtime` 先初始化 logging，再呼叫 factory，讓 catalog 註冊與建構錯誤寫入 log。
根直接傳入 lab 的 logging namespaces；不另建 logging Protocol。
組裝邊界見 [ADR-0072](../../../../../docs/adr/0072-module-boundaries.md#lab-定義與組合根)。

根選擇一份可信的 `startup(ctx)` 腳本。`--clean` 略過全部腳本且與 `--script` 互斥。
顯式腳本取代框架的 `default_startup.py`；自訂腳本需要預設行為時，自行 import 並呼叫
`zcu_tools.gui.app.measure.default_startup.startup`。
Framework 不 import lab 來選腳本。視窗顯示後，`after_show` 在 Qt loop 排入背景執行。
沒有選定腳本時開啟 shared Setup dialog。

`build_app_services()` constructs the app-local services and injects their driven
ports. UI and remote code submit tab, run/analyze, save, writeback and session
commands through the exposed control facets. `Controller` owns app composition,
lifecycle, experiment reload, template creation and library-edit orchestration.
It does not duplicate the existing facet commands.

`AppServices.session` and read-only `Controller.session_services` expose that
same session bundle to embedded borrowers. `Controller.workflow_control()`
provides the headless workflow tab registry, captured forms, native run control
and retained figure/record queries. GUI panes and remote callers use this same
owner. Workflow runs borrow the session RUN lease, runner and progress. Closing
an idle workflow tab releases only its presentation. Panes dispose their own
subscriptions, editors and timers before deleting children; they never drain
or disconnect the shared session. Native Ask answers and cancellation drain
continue on the retained operation worker and lease, never on the UI loop.

The toolbar keeps New Tab ▾ on the left and Setup…, SoC…, Devices…, Inspect…,
Predictor…, Results browser…, and More ▾ on the right. When space is short,
Predictor and Results browser move into More. More lists Reload experiments,
Retry skipped tabs (only when recovery has skipped tabs), and Workflow tabs /
Experiment tabs before any folded tools. The tab switch names the family
it will reveal. Catalog recovery changes Reload experiments to Retry reload.
Remote control disables reload and retry while leaving the inspection and tab
navigation entries available.

The toolbar opens Setup… and SoC… as separate non-modal windows. Setup manages
result entries and working points; SoC connects the board and displays soccfg
through `Controller.soc_control`. The named `soc` window supports remote screenshots.
Session dialogs remain available for inspection during Run or remote control.
Run disables Setup and SoC submissions. Inspect, Devices and ARB retain their
existing Run policy. Remote control makes all five shared dialogs read-only;
Local restores editing subject to each dialog's Run and validation gates.

`Controller.session_services.soc_connection.install_soc` / `uninstall_soc` publish
a caller-built board on the owner thread. `register_device` adopts a built driver through DeviceService.
`close()` only drains work, leaving resources available for caller callbacks.
`finish_close()` drains, runs startup callbacks in reverse order, then calls
`close_hardware()` to close registered drivers and unpublish the board.
MainWindow close, programmatic shutdown and the runtime fallback use this same
ordering. Drain timeout preserves resources and raises for retry. Callback and
driver failures are logged without skipping the remaining cleanup. Boards remain
caller-owned and have no required close method.

The runtime's `MeasureGuiBehavior.close()` delegates to its owner-thread
`dispose()`. After successful `finish_close()`, it stops the remote adapter and
calls `MainWindow.dispose()` to release view subscriptions and destroy the Qt
tree before QApplication/Python exit. Partial assembly retains the window for
this same cleanup. A drain timeout leaves the window and adapter intact for
retry. View disposal finishes hidden named dialogs through `QDialog.done()` so
their owners release subscriptions, detaches tabs, then explicitly delivers the
window's DeferredDelete even when the main event loop has already returned.
Successful disposal is idempotent; closing/hiding a window alone is not disposal.

[`StartupContext`](startup_context.py) exposes typed soc, device, setup, context
and tab commands. Public signatures, thread restrictions and failure semantics
live with these facets rather than a separate script API list.
Each facet calls the existing owner through `OwnerScheduler`. Live `current()`
queries project remote endpoints, device rows, entry/point and authored tab inputs
without hardware I/O. Blocking SoC/device connections run through the existing
async services and wait only on script workers. Owner calls reject before
admission. `devices.remember` publishes an idle row for adapter Reconnect.
`tabs.new_tab(adapter, input=TabInput(...))` lets the form owner restore opaque
`FormInput.to_json()` input on fresh sources. `TabInput(form_json: str)` is an
opaque authored-input carrier, not a disk session schema; it includes invalid
drafts and selections, not sources or results. Only the Form owner decodes it.
Incompatible paths are discarded and logged; malformed JSON and unknown adapters
raise before tab publication.
On-close callbacks run on-owner and use nonblocking facets without self-queuing.
`cache_root` locates the user cache without defining a session schema. Scripts
own their files and close-time capture. `summary()` sets the successful terminal
status line; unhandled script errors override it.
`StartupService` shares the operation runner, handles and background drain.
Its app-local startup lease blocks Run while allowing script hardware commands.
Progress and terminal facts update the status bar and Run controls. Owner calls
carry an app-local system startup origin through existing operation attribution.
Script-initiated device failures retain their terminal fact, failed handle and
logs, without a duplicate item dialog. Manual GUI device failures still notify.
Unhandled script errors report the script path and line while retaining completed work.
The framework does not load lab environments or own script cache formats.

The framework-owned [default script](default_startup.py) owns
`measure-session.json` under `ctx.cache_root`. It attempts SoC, devices,
entry/point, then tabs with input. A saved point=None needs only a successful
entry open, which selects Global. A saved point label must select successfully;
failure skips tabs without falling back to Global. Entry-open failure also skips
tabs.
Each failure is logged; failed devices are remembered when possible, and skipped
tabs are not retried after later source selection. Its on-close callback captures
then-current state. Aliases are not in that cache. Opening an entry loads its
global aliases; selecting a point loads `points/<label>/aliases.yaml`.
Startup code opens an entry before `ctx.context.apply_alias_rows(rows)`.
That command saves active aliases, global or point, without a separate reload.
Without an opened entry it raises no_context and retains no pending rows.
Templates use caller-authored `Template.bindings`.

`Adapter.save_stem(context)` delegates to the adapter's spec.save_stem callback.
SaveContext carries the full adapter path, run_id, historical point/roles, available cfg values and saved-at time.
The default stem joins the dotted adapter path, hyphen-joined role values and full run_id, for example `twotone.time_domain.t1_Q1-R1_20261027T120000Z-a1b2c3`.
An adapter can pass its own spec with a different callback. Save uses the returned stem verbatim for `runs/<stem>/` and the Labber date tree's `<stem>.hdf5`.
Only an existing Labber name triggers numeric suffix selection. Native same-path replacement stays unchanged.
Labber comments carry snapshot.point and snapshot.roles beside cfg. Figure-only result browsing links successful saved output paths to acquired facts, not directory-name suffixes.

User experiment adapters import `gui.app.measure` contracts. The framework
receives their catalogs by injection and never imports the user package.

## Figure presentation

`services/run_analyze_control.py` 的 `MeasurePlotHost` 在 Qt canvas 掛載後通知 pane，
沿用 `QtPlotHost` 的 owner-thread 繪圖與 release 契約。

Run、Analysis、Post-analysis、Workflow 與 Data 的具名圖預覽，四張以內在同一頁平鋪，
超過四張時每頁最多四張。上一頁／下一頁按鈕與 `i / N` 顯示目前頁次。
換頁會將該頁第一張設為「目前圖」，點選 canvas tile 會改選該張。
Run／Analysis／Post-analysis 的 RPC screenshot 與 `get_current_figure_for_pane`
讀取這張原生 Figure。Workflow 的 `get_current_figure()` 提供同樣的唯讀查詢。
同一 FigureCollection 重送時保留頁次與目前圖，清空時隱藏分頁列。
Data 預覽只保存圖名與 raster cache，分頁不移動原始 canvas。
Workflow 共用這組平鋪與分頁列，不顯示圖名下拉選單。

## Workflow hosting

`Controller.session_services` exposes the existing session bundle. Workflow
attachments import from `services.workflow_attachment`; the caller supplies the
registration list to `WorkflowControlFacet.configure`. Framework code never
imports the lab registry.

`Controller.workflow_control()` returns the headless facet. Each `open(name)`
creates independent plan/tunables drafts, including repeated names. `tab_id`
addresses a draft owner; `run_id` addresses its native run. Start, Stop and Apply
use the same runner, operation and RUN lease as experiment tabs.
Busy runs, including Ask waits, refuse tab close. Closing the last idle workflow tab
does not drain services or disconnect hardware.

Qt workflow panes display these owners and forward commands. A busy workflow's
local Stop remains available under Remote control. It reclaims locked Local before
requesting Stop through the same facet; other mutations still require the holder.
Tunables Discard restores the latest accepted input and clears local invalid text.
Before Start, it restores the captured form base. It is enabled only for a changed
draft while editing is allowed. Running and waiting runs allow tunables edits
and Apply; stopping locks edits, Apply and Discard. A new Ask selects its
question page. Waiting means an Ask is awaiting its answer.
Measure remote calls the same facet without a pane, toggle or selected-tab
dependency. Form
`observe_form` prepares one call-local candidate using the owner's fixed env;
metadata and values come from that candidate, with no native lowering or commit.
None Start batches use parsed shared drafts. Explicit batches use fresh captured
defaults. Tunables Apply with an empty batch submits the current shared draft.

`WorkflowAttachment(script=...)` pairs an async workflow with its native plan,
live tunables and optional summary types. One Runner worker calls Engine.execute
through terminal completion. Ask retains that worker, operation, progress and
RUN lease. Answer signals it without another submission.
Running and waiting Apply both publish a complete tunables model and a new revision.
The next tun.current read sees it; already captured operation cfg stays unchanged.
Each operation calls its callable once. Cooperative child cancellation delivers
its returned value. Root Stop records completion facts, then raises ScriptCancelled.

`pending_question` returns the detached
`{request_id, question_type, question, answer_schema}` envelope while that worker
waits for an answer. Authored fields stay inside `question`. Answer requires the
original request_id. Engine matches it before schema validation; stale or absent
IDs raise QuestionMismatch with the current pending ID. Rejection preserves the
question and records no accepted answer. Wrong native models or fields raise
TypeError/ValidationError without terminating the run.

The headless owner retains `last_exchange` with the latest question, accepted
answer and waiting/submitted/answered/cancelled presentation state. Qt panes and
RPC/MCP status read the same facts. GUI Answer sends the rendered question ID,
not whichever question is pending at click time. Submitted is not acceptance.
Stop releases Ask with None on its existing worker. Control I/O failure retains
its original cause and fails after worker cleanup. Only terminal owner delivery
settles the operation and releases RUN.
The tab injects `WorkflowPlots`. Each native `env.run` attempt receives a fresh
same-host operation Plots and finishes before returning or being discarded.
`WorkflowFigures` tiles workflow figures and the current operation together,
with up to four figures per page and page controls for larger sets.
Follow latest starts checked and selects each newly presented or live-refreshed
figure's page and tile. Clicking a canvas unchecks it and pins the selection;
final flushes do not move focus.
Pane and MCP read `WorkflowPlots.figures()`: workflow names stay unchanged and
operation names are `operation/<name>`. Workflow registration
rejects that prefix. A new attempt releases and replaces the whole operation
group. Terminal runs retain the latest group; replacement Start and close
release both groups without destroying retained Figures.

Hosting decisions and their reasons are in ADR-0067 and ADR-0068.

## Sources and workflow

`Session.cfg` owns one `TabFormResource`. The resource owns one `FormState`
and captures parameters, library and cached device infos on Open, Refresh and Reset. Preview, result and Run use that capture until the next explicit source action. Refresh preserves all authored data and explicit selections, including expression drafts and previously resolved direct defaults. Reset recaptures sources and computes fresh defaults. Capture and factory failures preserve the previous form.
Only `open_tab_from_file` isolates adapter `make_form` failure so native history
can still load without a Form. Template binding and default construction still
fail before publication. The form area and disabled Run show the factory reason.
Reset uses the normal source capture and factory: success installs a real Form;
any preparation failure propagates unchanged, leaving no Form and updating the
visible reason, including capture, template binding and defaults failures. Native cfg
never supplies form input. See ADR-0073 and ADR-0067.
Live experiment reload uses detached `WorkspaceTab`／`WorkspaceSnapshot` RAM carriers and the shared `{data,selections}` codec. Rebuilding uses fresh captured defaults and Core recursive restoration, with dropped paths reported to the host. `TabSnapshot.cfg_input` remains live raw data. Framework disk persistence stores only window dimensions and left-panel width in `measure_layout_v1.json`; startup scripts own session save/restore policy. Entry and each point own independent alias files. Open/Refresh/Reset capture the active document set and its table. RPC and startup Apply save that table; an opened entry is required, a point is not. Missing files are empty without fallback. Inspect offers Active documents and Entry template on all three pages. Completed valid edits write through to the selected destination. Same-target source notifications preserve dirty input and its original revision. Switching entries or closing discards unsubmitted input. Existing tabs keep their captures until explicit Refresh or Reset.
Source events do not refresh existing forms. Device capture copies cached infos;
it never queries a live driver.

Recipe 的 logical read/edit/selector 命令也由 `TabFormResource` 擁有。Adapter 解析路徑並產生隔離候選，resource 在同一 mutation guard 內安裝，保留 base。Changed batch 只通知與發布一次，inline no-op 仍先檢查 guards。一般 invalid draft 保留，結構或來源拒絕不提交成功前綴。Logical readonly read 只投影固定 captured Spec，不建立可寫草稿，不更動 input、base 或 publication。通用 Form RPC 與 widget 仍使用 canonical paths。

RunService 在 run 終局後請 DeviceService 背景讀回已連線儀器。成功台更新觀察 cache，失敗台保留既有值。這項讀回不改變 run 終局、settlement、lease 或 run 前固定的 acquisition snapshot。

RunInputs 固定 admission 的 ParameterContext，RunPaneState 隨 result 保存同一引用。
`stop_requested` 表示前端在這次 Run 發出過 Stop 或成功的 selector cancel，包含只取消子層。
Core 自己設 stop 不改這個欄位。Snapshot 與 wire 的 result_state 交付同一 bool。
New Run 或非 acquisition replacement 重設為 False。
Core 正常回傳就發布 record，GUI 顯示 finished，operation handle 也回 finished。
Stop、root 與子孫 scope 的取消都不改這個 terminal。
Core 拋錯就標 failed、不發布 record，Stop 後的錯誤也顯示原訊息。
RunMetadata.completion 的 stopped 表示 core 回傳前前端發出過停止請求，不判定資料是否完整。
晚開 FIT、interactive factory、Post 與 WB 讀這份 result source，不讀後來的 tab binding。
Native Load 保留檔案 metadata，從 parameter_snapshot 還原當次 point、library 與 alias。
Load 接受任意位置的檔案，不讀 live point，也不依 entry 身分選來源。
缺快照時使用沒有 general、target 或 storage 的空內容，分析可以使用自己的預設值。
Run／Load 的輸出預覽與 Save 共用 `SaveService.plan_layout`，依目前 roots、記錄的 entry／point 與 adapter stem 計畫 paths。
正常 Load 後立即有預覽，Analyze／PostAnalyze 後也有預覽，不要求 storage，狀態仍為 Not saved。
Metadata 缺 native schema 必填 key 時只有 artifact/member declarations，沒有輸出預覽。
預覽只隔離 `MetadataFieldError`，保留 Load 的原始內容；Save 仍完整驗證 metadata。
缺 entry 名稱使用 unnamed_entry。Point 為 null 的 global 或歷史結果將圖片副本放在 entry 層 figures/，不建立 unnamed_point。Save report 發布實際 paths、成功與失敗。
Load 不新增 Run token 或量測歷史。
純計算來源為 None，不回退到 session；新 Run 與 clear 清除舊來源。Analyze／Post-analyze 不攜帶 predictor。Predictor 是獨立查詢工具，不自動參與實驗。

Entry New／Open 安裝 GLOBAL 參數、library、aliases 與 storage，不自選既有 point。
Run、WB 與 Inspect 編輯可直接使用 GLOBAL，不要求選 point。
重新 Open 同 entry 回到 GLOBAL。Preview 與 Apply 的目的地是命令當下作用中的文件組。
選了 point 時，Inspect 的 Entry template 只改 entry 文件，不改作用中 point 或既有 tab capture。
Preview 的 destination_context 與 receipt destination 只報 entry_name、point，不含 kind。
GLOBAL 的 point 為 null，POINT 為 label。未開 entry 仍可預覽合法 draft，兩個位置欄位與 current 都為 null；非空 Apply 仍以 no_context 拒絕。
已提交 receipt 的 entry_name 必為字串。來源 capture 與 opaque actual_path 不變，preview 不鎖定 Apply 的目的地。
Cfg/editor drafts 與 WB 候選 preview 保持各自的生命週期。

`RecordWriteback` 只提出 adapter 組裝的 JSON object。
WritebackService 不解析欄位、不補座標、不查裝置或 alias。
格式由 adapter 間的 policy 決定，見 [Record policy](../../../../../zcu_lab/v2/measure-authoring.md#record-policy)。
元件記錄放在 `records/components/<target>.jsonl`，wire kind 與 ID prefix 使用 `record`。

`AdapterCapabilities` 由定義值推導 analysis 與 post-analysis。
每個 Run 都需要 SoC，guard 檢查 context、cfg 與連線。實驗的輸入驗證在 run 內執行。
每個 adapter 都可嘗試 Load，LoadService 以檔案與 spec 判斷相容性，失敗保留舊成果。
Qt tab 依 analysis／post-analysis 建立控制項，remote writeback subtab params 也檢查對應宣告。

## Tab admission

`GuardService.admit(tab_id, action) -> AdmissionDecision` 在 owner thread 讀取 domain 前提。
Decision 包含 `allowed`、第一個拒絕的 `reason` 與共用 `message`，放行時 reason 為 None、message 為空字串。
所有 action 先查 `experiment_unavailable`，再查 `unknown_tab`，其餘檢查依下表由左至右執行。

| Action | 其餘檢查順序 |
| --- | --- |
| RUN | active context → cfg → SoC 連線 → tab busy |
| LOAD | tab busy |
| ANALYZE | Run result → analysis capability → tab busy |
| POST_ANALYZE | Primary result → post-analysis capability → tab busy |
| WRITEBACK | opened entry → Primary result → tab busy |

Run、Load 不要求 storage，Load 與 Analyze 不要求目前 context。
Analyze 缺能力時回 `no_analysis`，不會把 `AnalysisMode.NONE` 送入 FIT worker。
Post-analyze 缺能力時回 `no_post_analysis`。
Writeback owner 另行驗證目前選定的 destination；Post writeback UI 還要求 Post result，只有 Post result 不能略過 Primary result 前提。
Save 不使用這份 admission，保存前提仍由 SaveService 檢查。

`check_load/analyze/writeback` 是 admit 的薄包裝，拒絕時以 decision 的 reason/message raise `GuardError`。
Run facet 只呼叫 `capture_run_inputs`，由 capture 呼叫一次 admit，invalid cfg 也以相同 reason/message raise `GuardError`。
放行後 capture 只呼叫一次 `accept()`，執行時的 resource／reentrancy 拒絕保留 `CfgPreconditionError`。
`TabFormResource.run_invalid_reason()` 先讀 Form unavailable 原因，再讀 model／lowering 錯誤。
多錯誤時首因附 `(+N more)`，N 是其餘錯誤數。
這個唯讀查詢不接受或發布 cfg、不捕捉來源，可在 notification 中呼叫。
Resource revocation 與 author lower-hook 例外仍傳出，不偽裝成放行。

UI 的 domain 拒絕停用按鈕並顯示 decision.message。
Run admission 放行後，UI 仍依序顯示尚未寫入 resource 的 editor parse 錯誤、startup、其他 tab Run 的阻擋訊息。
Admission 不取得 lease，hardware exclusion 仍由 operation gate 在啟動時裁決，放行不保證啟動成功。
Remote 也讀取 decision；只有 `unknown_tab` 投影成既有 `INVALID_PARAMS` 與空 reason，其餘拒絕保留 decision 的 reason/message。
Run aliases 檢查仍在 admission 前，admission 則先於 source provenance 檢查。
沒有 Run result 時，帶 run_operation_id 仍先回 `no_run_result`；已有結果但來源過期或未知時仍由 source owner 回 `result_superseded`。
Remote Run 保留 admission 後 `accept()` 執行時 `CfgPreconditionError` 的 wire 轉譯。
責任分工與裁決見 ADR-0066、ADR-0067。

## Experiment reload

`ExperimentReloadService` 擁有 RAM-only snapshot 與重載／恢復狀態，透過注入的
`ExperimentCatalogLoader` 載入新 registry，透過 Workspace 的正常 close/apply seam
重建全部 tabs。Controller 只轉接，工具列與 MainWindow 負責整批 destructive confirmation
和 failure/recovery presentation。Registry 的持久範本、hardware context 與 framework 不重建。

## Shared owners and decisions

- [`gui.session`](../../session/README.md)：context、SoC、device、project settings、predictor、operation
  handles、operation runner、progress/shutdown service、shared dialogs。
- [`gui.form`](../../form/README.md)：Qt-free input、FormState、按 model-field path 查詢的單層 description、preview 與 result。
- [`gui.form_adapter`](../../form_adapter/README.md)：core cfg／持久 Spec 推導、FieldMeta 呈現投影與 key-or-inline 轉換。
- [`gui.plotting`](../../plotting/README.md)：explicit figure host、container 與 export。
- [`mcp.measure`](../../../mcp/measure/README.md)：agent-facing MCP policy layer and tool surface。
- [ADR-0067](../../../../../docs/adr/0067-gui-application.md)：GUI owners、pane lifecycle、adapter 與 domain facts。
- [ADR-0065](../../../../../docs/adr/0065-cfg-editing.md)／[ADR-0008](../../../../../docs/adr/0008-cfg-editor-session.md)：Form、Library 與 writeback。
- [ADR-0066](../../../../../docs/adr/0066-operation-lifecycle.md)／[ADR-0063](../../../../../docs/adr/0063-persistence-ownership.md)：operation、shutdown 與 persistence。
