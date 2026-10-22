# `zcu_tools.gui` — GUI framework cheat-sheet

**Last updated:** 2026-10-23

**讀者：** framework-dev。共用 GUI 層、app 邊界與生命週期速查。

High-level map of the shared GUI layer. App-specific detail lives in each app's
own README under `app/<name>/`; cross-cutting subpackages (`event_bus`,
`plotting`, `remote`, `session`, `widgets`) are shared by every app.

## Source catalogs (`catalog_loader.py`)

`SourceCatalogLoader[CatalogT]` 共用 bounded prepare/load、namespace inventory、preserved/fixed source 與 restart 契約。App 注入零參數 typed builder，負責 catalog validation 和發布；loader 不認識 measure Registry 或具體 workflow。SourcePackage、PreparedCatalogReload 與 CatalogReloadError 位於同一 shared owner。Measure 使用 registration adapter，workflows 使用 typed callable/form/panel pairing。Import 並非任意 Python 副作用交易；fixed-source integrity 失敗要求重啟。App 仍擁有 active admission、draft recovery 與呈現政策（ADR-0067）。

## Owner-loop execution (`session/adapters/`)

Core state belongs to one owner thread. `OwnerScheduler` is the marshal seam：Qt
composition使用queued-signal adapter，headless/test composition使用explicitly pumped manual
adapter。`BackgroundExecutor`與scheduler分離：Qt runtime使用QThread/QThreadPool adapter；
headless runtime使用pure `ThreadPoolBackgroundExecutor`，並在worker完成後透過scheduler
投遞唯一terminal callback。Headless `quiesce()` 是 terminal close，會停止新工作並等待worker與owner delivery
ack。Qt `quiesce()` 保持可重複 drain；terminal shutdown 先呼叫 `stop_accepting()`，再取消與drain。
關閉入口後兩個executor的`submit`都立即拋`BackgroundAdmissionClosedError`，不排worker或callback。
Session services只依賴`submit` port，不持有另一份admission狀態。

## Remote response delivery (`remote/`)

Shared framing 的 request/response 上限為 8 MiB UTF-8 bytes，不含換行。
仍採單則 NDJSON 訊息，不分批；完整 context read 失敗不回部分資料。
Per-client writer 在 socket backpressure 時保存尚未送出的 byte cursor，等待可寫後接續；
不重送整則訊息，停止或斷線會退出等待並釋放連線。
每個 client 的待送預算為16 MiB，包含 queue 與 writer 持有的完整 encoded frame；
完成交付才釋放預算，超出則中止該連線，IO owner 清空 queue 並釋放 app context。
Shared endpoint 無法編碼 RPC 回覆時送有界的 `internal` error，reason 為
`response_encoding_failed`。Handler 可能已執行，caller 不可因回覆失敗而盲目重送 mutation。
若 correlated fallback 仍無法編碼，或 reply queue 拒收，就中止該連線，交 IO owner
釋放其 app context。Push 的 drop policy 不變；shared transport 不解讀 method、guard 或 operation。

## Resource version lifecycle

`VersionTable` owns the shared counter mechanism; apps own keys and semantic
writes on their owner thread. Exact `retire(key)` makes an absent resource read
as 0 while retaining its last counter. Same-key recreation advances beyond that
counter, so owner projections can distinguish replacements after name reuse.
`drop_prefix` instead forgets both live and retired counters, and later bumps
restart at 1. Owners choose whether replacement history must survive name reuse.
These counters are not RPC mutation baselines. History lasts only for the table lifetime.

## Control identity

Measure and Fluxdep inject application-owned `ControlIdentity` into shared remote
and Qt projections. Business RPCs require holder unless explicitly readonly.
The owner loop checks current authority before domain effects. Reads provide
current data, not per-connection version baselines. Authentication, domain busy,
result provenance and operation lifetimes remain separate checks.

The main window alone observes control identity. Its `set_writable` updates
write controls and pushes availability into existing and new tabs/views. Domain
busy/valid gates and pure viewing remain separate. Local modal dialogs refuse
owner acquisition with `local_busy` and "Local 使用中" on the Qt owner turn.
Choosers use non-native presentation so Qt can detect them. Dialogs do not
subscribe to authority or recheck holder after choices. Measure Setup, Inspect,
Device and ARB windows receive read-only state on open and on control transfers.
Local restores editing without overriding busy/valid gates.
Only Measure experiment, embedded workflow and device Stop reclaim
locked Local before requesting cancellation. Main windows dispose their own
watch on final teardown; domain sessions and background drains retain their
existing lifetimes. See [ADR-0070](../../../docs/adr/0070-control-identity.md).

## Remote parameter declarations

`gui.remote.ParamSpec` owns optional string enum declarations, schema projection
and request membership validation. Callers without an enum keep their existing
required/null/default behavior. ARRAY declarations also own the schema element
kind, with string as the default and number available for native fit bounds.
Element shape/type/finite validation remains with each domain handler. MCP preserves
scalar JSON types for GUI admission instead of converting malformed inputs.
Domain plugins supply their own authoritative choices; MCP does not maintain
another allowlist.

## Expected Errors (`expected_error.py`)

Caller-correctable failure使用Qt-free、remote-independent的nominal `ExpectedError`
seam。closed category只有`INVALID_INPUT`與`FAILED_PRECONDITION`；producer以concrete
exception或raise site明列分類，optional `reason_code`提供stable machine tag。普通
`RuntimeError`、provider/persistence/async terminal與invariant failures不自動opt in。
RuntimeError-compatible fixed leaves讓既有view/handler catches維持相容（ADR-0068）。
`gui.remote`的shared dispatch在main/off-main兩條路徑以同一translator投影nominal
`ExpectedError`；generic mapping保留message/reason且不攜帶structured data。direct
`RemoteError`仍供request coercion與domain-special payload使用，ordinary exception則保留
controller error與traceback。

## Form generator (`form_adapter/`)

`form_adapter` 是 Qt-free 前端框架，從 core cfg 與作者樹生成 `gui.form` 的 FormModel 並轉回 core cfg。各 app 與表單定義從它的公開 root import `Form`；它不依賴任何 app。細節見 [form_adapter README](form_adapter/README.md)。

## 共用表單與持久 Spec

`gui.form` 提供 Qt-free FormState、單層 describe、preview、result 與錯誤，`widgets/form` 的 FormEditor 顯示同一份 caller-owned state。Measure、Library／Inspect、writeback 與 workflows 共用這套工具，宿主各自決定 lifetime、capture 與提交時機。跨宿主契約見 ADR-0065、ADR-0008。

`gui.form_adapter` 集中 core cfg／持久 Spec 的生成與轉換。FieldMeta 的呈現資訊、合併的 None／keys／Inline 選項與屏蔽欄位政策都由生成器處理。Library 寫入直接提交 ModuleSpec／WaveformSpec，保留 child key，不先展開 runtime cfg。表單 expression 使用固定 env，Library 保存求得的數字。

Complex 表示轉換直接使用 `core.complex_codec`。Measure 的來源 provenance 與 precondition 型別由 `app/measure/services/form_contracts.py` 擁有，remote 錯誤轉譯由 `app/measure/remote/form_errors.py` 擁有。編輯入口由 `gui.form`、`gui.form_adapter` 與 `widgets/form` 提供。

## Process Runtime (`runtime.py`)

`gui.runtime` owns process-level startup mechanics for launchable GUI apps:
logging, `QApplication` creation, rendering initialization, shared plot-host
lifecycle, remote-control option construction, adapter start/stop, and integer
exit-code handling. Apps expose a fixed `GuiRuntimeBehavior.spec` class variable
for static process contract and implement `assemble(control)` for app-local
controller/window/adapter wiring. Launch-time CLI values stay in
`GuiLaunchOptions` and behavior constructor arguments. App modules expose
behavior classes; standalone `scripts/run_*_gui.py` launchers are the process
entrypoints and call `launch_gui_runtime(...)` directly.

The runtime seam deliberately stops above remote method/domain/session policy:
`gui.remote` still owns transport, each app owns dispatch/domain handlers, and
`gui.session` owns measurement-session primitives. The three standalone GUI launchers (`measure`, `fluxdep`, `dispersive`) use this
runtime directly. Measure hosts workflow tabs in the same session as experiment
tabs. The launcher injects workflow registrations; `WorkflowControlFacet` owns
headless drafts and native run routing. Qt panes and remote commands drive that
same facet. Workflow tab close never drains the session or disconnects hardware.

`gui.launcher` is the import-light CLI edge companion for runtime launchers. It
declares shared logging/control flags and converts parser output into
`GuiLaunchOptions`. Analysis entry flags and `AnalysisTarget` opening belong to
`app.analysis_project.launcher`. The shared `gui.launcher` avoids importing Qt,
matplotlib or app modules at module import time. Runtime initializes the explicit
plot host, shutdown callback and mathtext support after QApplication exists;
it does not select a process-wide Matplotlib backend.

When runtime gets or creates the `QApplication`, it calls
`gui.widgets.selectable_text.install_selectable_message_boxes(app)`. The call
installs one idempotent application event filter. Every `QMessageBox` shown in
the process then lets users select its main text, informative text and detailed
text with the mouse or keyboard. Call sites keep using plain `QMessageBox`; all
three launchers get the filter because runtime is their only `QApplication`
factory.

A `QLabel` that shows a data value (device address, name, path, computed value,
status or error message) calls `make_label_selectable(label)` from the same
module, so users can select and copy it with the mouse. Labels for field names,
hints, placeholders and decoration stay unselectable.

## Entry selection

Measure 的 shared SetupDialog 使用 `ProjectSettingsService` 開啟或建立 Result Entry，
再透過 ContextService 選取或建立 Working Point。Entry discovery 由 resources owner
列出含 setup.yaml 的條目，普通查詢重用 cache，明確 refresh 才重讀。開啟 entry
不自選 point，不連硬體，也不從 chip／qubit／resonator 名稱推導路徑。

Fluxdep／dispersive 的 `AnalysisEntry`、`AnalysisTarget`、`EntryDialog`、
entry CLI 與 RPC 投影由 [`app.analysis_project`](app/analysis_project/README.md)
擁有。它開啟既有 entry，從 setup 選 fluxonium component，不使用 chip／qubit
兩級目錄或 params.json。分析 app 不執行 Database Save。
Measure 的 entry binding 仍由 session 擁有。共用 GUI 不依賴此分析 owner。

## Shared session dialogs (`session/ui/`)

`SetupDialog`、`InspectDialog`、`ArbWaveformDialog`、`DeviceDialog` 與
`PredictorDialog` 由 shared session UI 擁有，依賴各 domain 的窄 control facet。
`session/ui/alias_panel.py` 的 AliasPanel 是共用 Inspect 的 Aliases 頁，編輯元件與 device
共用的有序 alias 表；沒有獨立的 Aliases 對話框或入口。
Apply 要求已選 point，先驗表，再由 entry owner 原子保存該 point 的 `aliases.yaml`，成功後才安裝 memory 並發布 fact。
空表也會保存。失敗保留舊表、舊檔與 draft。切 point 載入該 point 的表，缺檔就是空表，不回退 setup 或 caller preset。
Inspect 的 clean draft 隨切換換表。Dirty draft 固定來源 entry／point，跨 target 拒絕 Apply；Revert 載入目前 point 的表。
未存 draft 在關閉 Inspect 時走 Save／Discard／Cancel。
App 保留 dialog lifecycle 與呈現政策，shared dialog 不反向 import app。

## Dialog presentation and lifetime

Hosts choose dialog presentation separately from lifetime. Measure named windows
use NonModal plus `show()`; `open()` remains window-modal and is not a synonym
for non-modal. The owner retains dialogs until accepted/finished cleanup.
Short-lived modal choices may use a nested event loop, including measure's
single RunLoadDialog interaction. Measure exception reports retain a non-modal
message box with traceback. Modal owner admission protects Local
input, including file choosers and confirmations, without dialog control wiring.

RunLoadDialog discards further input on finished. MainWindow rejects it before
teardown, and a late file chooser cannot accept a new selection. Persistent
dialogs that hide and reopen keep their existing owner lifetime.

Short-lived modal confirmations and text prompts are the
deliberate exception. A synchronous `QMessageBox.question` / `.warning` /
`.critical` or `QInputDialog.getText` may block: it is raised by a direct,
synchronous user action while no operation is in flight, carries no
long-running flow, and never marshals a worker result back to the main
thread. Its nested modal loop still services the control socket's queued
notifier, so it neither stalls startup nor deadlocks cross-thread
marshalling. The `open()` rule still governs every dialog that hosts a
long-running flow or can surface while an operation is running.

Widgets that need information messages, warnings, critical errors,
confirmations, or explicit destructive confirmations receive the shared
`DialogPresenter` port. Production uses `QtDialogPresenter`; when a
`DialogRefStore` is supplied, information, warning, and critical boxes stay
non-blocking and retained through the shared lifecycle helper, but remain
window-modal by default. Measure explicitly selects non-modal notifications;
confirmation policy is unchanged. Close paths or operation-running prompts use callback-style `confirm_async` /
`destructive_confirm`, so the Qt event loop keeps pumping while the decision is
pending. `destructive_confirm` takes a `DestructiveConfirmation` with the title,
message, action label, safe-alternative label and default choice; the parent and
decision callback remain explicit inputs. Tests inject a recording presenter
instead of monkeypatching `QMessageBox`, so dialog decisions are scripted at the
object boundary.

Non-modal dialogs shown with `show()` use the shared dialog lifecycle helper,
or an equivalent named registry when remote screenshot/list semantics require
stable names. Either path keeps a Python reference until `finished` / `destroyed`
cleanup runs.

## 共用 Qt Form editor

`widgets/form` 擁有 FormEditor 與 `value_widgets.py` 的共用 scalar 控制項 helpers，analysis forms 也使用這些 helpers。Editor 透過 `FormState.describe(path, env)` 讀取單層 description，首次展開才查下一層。Editor 讀取 preview 與 errors，不求值、不複製 Library，也不擁有 Apply。

文字在 Enter 或失去焦點時提交，未解析文字留在 view，宿主提交前檢查 local_parse_errors。唯讀時仍能複製文字、折疊與捲動。選項與 Expression 的呈現、訂閱生命週期見 [FormEditor README](widgets/form/README.md)。

## EventBus Lifecycle

`BaseEventBus` stays Qt-free and payload-type keyed. `subscribe()` returns an
idempotent cleanup handle, and long-lived windows / bridges group those handles
so close/stop tears down callbacks explicitly. Event delivery rules are
unchanged: callbacks for the concrete payload type run in order, and one
subscriber raising is logged without aborting later subscribers.
Every emit also receives a process-wide monotonic `EventMeta(seq, origin)` stamp.
Legacy `subscribe()` callbacks still receive only the payload; attribution-aware
boundaries use `subscribe_with_meta()`. The per-bus origin is a nested
`ContextVar` defaulting to `user`, but it does not cross worker threads:
`OperationHandles` captures the origin at creation and `OperationRunner` restores
it with the string operation id around terminal delivery. Remote dispatch declares
an opaque per-connection `agent` origin; persistence restore declares `system`.
Remote bridges subscribe their EventBus push callbacks before opening the socket;
startup subscription failure is fatal and rolls back partial subscriptions, while
runtime subscriber exceptions remain isolated by the EventBus.
Remote payload materialization and encoding are lazy: the shared endpoint first
selects matching live subscribers, calls the main-thread factory once outside its
registry lock, then revalidates before using the existing per-client queues. Zero
matches perform no serializer/encode work; multiple matches share one encoded
line while preserving independent ordering, backpressure, and drop isolation.
EventBus wire pushes carry the unchanged `event` / `payload` plus `seq` and
`origin={kind, operation_id}`; RPC-only `client_id` never crosses the wire.
Diagnostics and editor pushes are separate channels and retain their existing
two-field envelope.

Views hold render caches only and never apply optimistic domain updates. Pending
state is broadcast by core so every attached view renders the same authoritative
state.

## Logging (`logging_setup.py`)

`logging_setup.setup_gui_logging` is the single place that decides *how* every
GUI entry point configures logging. All four `scripts/run_*_gui.py` launchers and
the measure MCP server (`mcp/measure/server.py:main`) call it instead of each
rolling their own handler set.

Key invariants:

- **The file handler is attached at the whole `zcu_tools.gui` namespace** (plus
  any `extra_namespaces` an entry point injects for its experiment package;
  the MCP server adds `zcu_tools.mcp`). Attaching
  at the package root, not an app sub-namespace, is deliberate: cross-cutting
  subpackages (`event_bus`, `plotting`, `remote`, `session`) are siblings of the
  app namespace, and a handler scoped to one app would silently miss them. That
  missed-sibling gap is the bug this scheme exists to prevent; the regression
  test is `tests/gui/test_logging_setup.py`.
- **Per-session timestamped files** under `<repo>/logs/<group>/<app>/` (`group`
  = `gui` for launchers, `mcp` for the server). Each launch writes its own file,
  so a previous session's evidence is never overwritten. On startup the helper
  purges all but the newest `retain` (default 10) files in that directory.
- **Levels:** file handler at DEBUG, stderr handler at WARNING. High-frequency
  bookkeeping (operation create, background submit) logs at DEBUG; lifecycle
  events (operation settle, connect/device result, persistence flush/restore,
  writeback apply) at INFO; failures at WARNING/ERROR (worker exceptions carry
  `exc_info` so the real traceback survives the cross-thread marshal).
- A `--log-file` CLI override (when a launcher exposes one) wins over the
  per-session scheme: the explicit path is used verbatim and no purge runs.
- Repeated setup in the same process replaces handlers previously installed by
  this helper instead of stacking duplicate stderr/file handlers. User-installed
  handlers are left untouched.

The MCP server uses stdout for its JSON-RPC transport, so its logging never
touches stdout — the helper only adds a stderr handler and the DEBUG file
handler.

## Qt Typing Invariants

Qt override signatures must match the binding stubs exactly, including nullable
event parameters and stub parameter names. `closeEvent` implementations use
`a0: QCloseEvent | None`; `eventFilter` implementations use
`a0: QObject | None, a1: QEvent | None` and immediately delegate nullable inputs
to `super()`.

Qt table header accessors are treated as optional by the stubs. Code that
configures `horizontalHeader()` must assert the header is not `None` before
calling resize APIs; the assert records the widget invariant and keeps the
runtime failure explicit if a binding ever violates it.
