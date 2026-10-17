# `zcu_tools.gui.app.measure`

**Last updated:** 2026-10-18，run 終局後 device cache 讀回

`gui.app.measure` 是 measure-gui 的 app framework。它負責 tab lifecycle、cfg
editing、context/SoC/device/session wiring、run/analyze/save/writeback workflow、Qt
view 與 GUI-side remote handler。使用者套件的 adapter 擁有實驗領域知識；
framework 只看 `ExpAdapterProtocol`。共用 `BaseAdapter` 住在 `adapter/base.py`。它宣告 `Form`，分頁從固定 `FormEnv` 產生 fresh input。

## Owners

- [`adapter/`](adapter/protocol.py)：framework-facing contract、request/result/analyze params、
  adapter validation 與 protocol signature 所需的 session vocabulary。Writeback 型別由
  `writeback/` 擁有，adapter 公開入口再匯出。ModuleWriteback proposal 直接使用持久
  ModuleSpec，不帶 editor identity。
- [`services/`](services/app_services.py)：app service layer。Service 依賴 ports，不直接 import sibling service
  implementation；package `__init__` 只做 lazy public re-export。
- [`state.py`](state.py)：tab/device/pane/path/version-table SSOT 與主線程 mutators；固定的
  Run、Analysis、Post-Analysis、Save pane 各自擁有自己的 resource。
- [`artifacts/`](artifacts/handover.py)：tracker、SaveService、SaveControl、保存中心與 `ArtifactHandover`。
- [`writeback/`](writeback/service.py)：parameter／module／sample proposals、draft、service、control 與 Qt review widget。
- [`ui/`](ui/main_window.py)：Qt widgets、MainWindow top-level façade、capability-driven `ExpTabWidget`、
  feedback widgets；writeback view 由 `writeback/` 擁有，generic cfg form不屬於app package。
- [`remote/`](remote/README.md)：與 `ui/` 平級的 GUI-process driving adapter，將 RPC 意圖交給
  application commands/queries。`remote.method_specs` public import path 不載入
  Qt-bound service code；MCP bridge 不在本 package。
- `driven/`：measure app-local Qt/liveplot driven adapters；與 `adapter/` 的 experiment
  framework contract 分開命名。

resources.entry executor 與 remote handler／wire types 留在各自 owner。
Artifact 與 Writeback 不直接互相 import，由 app composition 協調。
`measure-artifacts-no-writeback` 與 `measure-writeback-no-artifacts` contracts 約束兩個方向，
允許共用 State／events／context。

## Composition root

`MeasureGuiBehavior` is the process-runtime behavior for the shared
`gui.runtime` launcher seam. It assembles `State`, `Controller`, `MainWindow`,
persistence caretaker, and the app-local `RemoteControlAdapter`
without owning process policy such as logging, rendering initialization,
`QApplication`, control option construction, or exit-code handling. The
standalone launcher is the process entrypoint; this module does not expose a
second `run_app` path. After the window is shown, `after_show` opens the same
Setup dialog through `MainWindow.open_dialog`, so a toolbar click focuses that
instance instead of building another.

`build_app_services()` constructs the app-local services and injects their driven
ports. `Controller` is a facade over the service bundle; UI and remote code use
the controller for app-specific workflow and the exposed session control facets
for setup/context/device/predictor/progress domains.

User experiment adapters import `gui.app.measure` contracts. The framework
receives their catalogs by injection and never imports the user package.

## Sources and workflow

`Session.cfg` owns one `TabFormResource`. The resource owns one `FormState`
and captures parameters, library and cached device infos on Open and Reset. Preview, result and Run use that capture until the next Reset.
Source events do not refresh existing forms. Device capture copies cached infos;
it never queries a live driver.

Recipe 的 logical read/edit/selector 命令也由 `TabFormResource` 擁有。Adapter 解析路徑並產生隔離候選，resource 在同一 mutation guard 內安裝，保留 base。Changed batch 只通知與發布一次，inline no-op 仍先檢查 guards。一般 invalid draft 保留，結構或來源拒絕不提交成功前綴。Logical readonly read 只投影固定 captured Spec，不建立可寫草稿，不更動 input、base 或 publication。通用 Form RPC 與 widget 仍使用 canonical paths。

RunService 在 run 終局後請 DeviceService 背景讀回已連線儀器。成功台更新觀察 cache，失敗台保留既有值。這項讀回不改變 run 終局、settlement、lease 或 run 前固定的 acquisition snapshot。

RunInputs 固定 admission 的 ParameterContext，RunPaneState 隨 result 保存同一引用。
晚開 FIT、interactive factory、Post 與 WB 讀這份 result source，不讀後來的 tab binding。
Native Load 保留檔案 metadata。在相同 opened entry，Load 一次讀回歷史 point，
固定檔案的 alias 名稱供分析與 WB。值是 Load 時的持久值，不重建未保存的原始參數版本。
跨 entry、EMPTY 或無 point 時，Load 不提供分析來源，也不回退全域 alias。
Context-free analysis 仍可用。Save storage 獨立固定在 Load request 時。
Load 不新增 Run token 或量測歷史。
純計算來源為 None，不回退到 session；新 Run 與 clear 清除舊來源。Analyze／Post-analyze 不攜帶 predictor。Predictor 是獨立查詢工具，不自動參與實驗。

Entry New／Open 不建立參數環境，也不自選既有 point。
Run、WB 與目前 point 的 Inspect Set 要先選取或建立工作點；GUI 停用操作並提示，application 也拒絕未選取狀態。
重新 Open 同 entry 清除 selection，WB 必須再次選取任一 point 才能提交，
但目的地仍是候選捕捉的 SavedPoint，不跟隨畫面上的 point。
Cfg/editor drafts 與 WB 候選 preview 保持各自的生命週期。

`ExpAdapterProtocol` 是 framework 呼叫 adapter 的契約。`AdapterCapabilities`
宣告 SoC 需求、analysis、post-analysis、load 的支援範圍；`requires_soc` 不代表
SoC 已連線。Run guard 檢查 context、cfg、SoC 與 preflight；operation owner 另
檢查動態 gate。Load permit 和 `LoadService` 檢查 `load_data`，Qt tab 依 analysis／
post-analysis 宣告建立控制項，remote writeback subtab params 也檢查對應宣告。
但 analyze permit 未查 analysis，非 interactive 的分析會進 FIT 路徑；
post-analyze 入口未查 post-analysis。不可將 UI／remote 的局部檢查視為
所有 application 入口的保證；補齊目標見
[GUI capability draft](../../../../../docs/adr/draft/gui-adapter-capability-guards.md)。

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
