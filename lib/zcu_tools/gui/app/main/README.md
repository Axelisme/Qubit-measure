**Last updated:** 2026-06-26（ArbWaveform autoscaled preview）

# `zcu_tools/gui/app/main/` — measure-gui Framework AI Note

> **MCP 搬遷（2026-06-08, c8eb1a03）**：measure 的 MCP server entry（原 `services/remote/mcp_server.py`）已搬到 `zcu_tools/mcp/measure/server.py`、共用 `McpBridge`→`zcu_tools/mcp/core/bridge`；`MCPBridgeConfig` 拆出基底 `McpServerConfig`（measure 仍用子類 `MCPBridgeConfig`，含 guard/operation/diagnostic policy）。`mcp` 是 `gui.remote` 使用方（非 leaf）。本筆記內樹狀圖與 `services/remote/mcp_server.py`、`gui/remote/mcp_bridge`、`parents[6/7]`（現 `parents[3/4]`）等舊位置/深度按此對映。

> **位置**：此套件 2026-06-04 從 `zcu_tools/gui/` 純搬遷到 `zcu_tools/gui/app/main/`（Phase 133 步驟 A，只搬不抽共用層）。import 一律 `zcu_tools.gui.app.main.X`；同子目錄內走相對 import、跨子目錄走絕對。`zcu_tools.gui` 父套件只是 namespace（不 re-export、import-clean）。本筆記與 ADR/memory 中殘留的 `zcu_tools.gui.X` 舊路徑請按此對映理解。

> **session 共用層（2026-06-09, 3b89bacf, S1）**：量測 session-core 值型別 + 事件 + async-operation handle 已提到 app-agnostic 共用層 `zcu_tools/gui/session/`（供 measure + 開發中 autofluxdep 複用，計劃見 `.agent_state/plans/gui/session_core_extraction.md`）：`session/types.py`（`ExpContext`/`ContextReadiness`/`SocHandle`/`SocCfgHandle`/`SocProtocol`/`SocCfgProtocol`——adapter 從此 re-export，因為是 `ExpAdapterProtocol` 契約詞彙）、`session/events.py`（session 事件：`SessionEvent` enum + `SessionPayload` base + md/ml/context/soc/predictor/device payloads）、`session/operation_handles.py`（`OperationHandles`）。三模組 import-clean（登記於 `tests/gui/test_shared_layer.py`）。S2–S5（抽 session services/SessionState/dialog、重塑 autofluxdep）待做。

## Module Purpose

通用 GUI 框架層，不含任何實驗領域知識。`experiment/v2_gui/` 依賴此層，反向不成立。

## Architecture Overview

```
gui/
├── adapter/         — ExpAdapterProtocol (framework 契約, generic-free), request validation, ExpContext（+ContextReadiness/Soc handles 自 `gui/session/types` re-export，契約詞彙）, Spec/Value 型別樹, DirectValue | EvalValue, CfgSchema.to_raw_dict(), dataclass-based analyze params (ParamMeta)。共用實作 BaseAdapter 在 experiment/v2_gui/adapters/base.py
├── cfg_schemas.py   — module_cfg_to_value(cfg) / waveform_cfg_to_value(cfg)：live cfg → (spec, value)
├── events/          — measure-gui experiment 事件（domain 各自分模塊，ADR-0021）：`events/tab.py`（`TabEvent`+4 payloads：TAB_ADDED/CLOSED/CONTENT_CHANGED/INTERACTION_CHANGED）、`events/run.py`（`RunEvent`+2 payloads：RUN_STARTED/FINISHED）；session 事件（md/ml/context/soc/predictor/device）在共用層 `gui/session/events.py`（SessionEvent，wire 字串不變）；**`event_bus.py` 已刪**，各 service 直接 import BaseEventBus + domain payload types；subscribe(PayloadType,cb) 自動推型別、emit(payload) 吞+log
├── specs/           — fresh spec factory functions（對應 program/v2/modules/ 結構）
├── state.py         — State (被動容器 + 主線 mutator/version bump) + aggregate roots TabState/DeviceState (帶自己的查詢謂詞 is_busy/has_*/is_connected/is_live/effective_save_paths)
├── registry.py      — name → Adapter class 映射
│   （off-main 執行 = 共用 `gui/background.py` `BackgroundRunner`：**純 off-main 執行器**（ADR-0026 §2）`submit(work, *, run_in_pool, on_done, on_error[, enter])`——只把無參 thunk 丟 worker/pool、結果 marshal 回主線程，**無 scopes 參數、不認 figure/stop facet**（舊 `OffMainScopes`/`_entered` 已退役）。scope wiring 改由 op policy 的 work thunk 自己用 closure 套（`figure_ambient`/`progress_ambient`/`ActiveTask`）。生命週期編排移到 `OperationRunner`（見「Hardware Operation Gate」/ADR-0026），runner 只剩執行。**measure 與 autofluxdep owner 直接持具體 `BackgroundRunner`**（才能呼 anti-segfault 的 `quiesce()`）；session service 只認 `BackgroundExecutor` port 的 `submit`（薄 `BackgroundRunner(QObject)` 殼已刪）。三 app 共用同一 runner）
├── controller.py    — 薄 Façade，委託邏輯給 services/
├── live_model.py    — LiveField / LiveModel：反應式資料層 (Observable Proxy)
├── sweep_model.py   — SweepEditor：無 Qt dependency 的 sweep canonical transform
│   （plot substrate 已抽到共用層 zcu_tools.gui.plotting — backend/host/routing/container/setup）
│   （progress primitives 已下放共用 session：ProgressBar/ProgressBarModel 在 `gui/session/pbar_host.py`，ProgressEvent/Kind/Transport port 在 `gui/session/ports.py`，ProgressStack widget 在 `gui/session/ui/progress_stack.py`）
├── adapters/        — driven adapters（實作 service port 的 Qt/OS/HW 具體層）：qt_progress_transport.py(QtProgressTransport 實作 session ProgressTransport port，queued-connection marshal worker→主線程)
├── services/progress.py — ProgressService(Qt-free 唯一 progress owner，dict[operation_id, ProgressContainer]，owner_id→live op；經 session ProgressTransport 收事件、無鎖；import session pbar_host primitives) + ProgressContainer(一 operation 一個，dict[handle_id, ProgressBarModel]) + BoundProgressFactory(綁 operation 的 worker factory)。**concrete service 留 measure（approach B）**；run + device-setup 共用
├── services/        — 拆分出的純邏輯層
│   ├── app_services.py   — AppServices frozen bundle + build_app_services()：集中 service 建構/接線
│   ├── guard.py          — GuardService + 型別化 Permit (Run/Save/Analyze/Writeback)：domain guard 單一所有者
│   ├── operation_gate.py — OperationGate（純 Exclusion，ADR-0019）：run/SoC connect/device mutation 的 hardware 互斥（ensure_can_start/register/release/has_active/is_device_mutating），keyed by token
│   （OperationHandles + OperationRunner + OperationChannel 在共用層 `gui/session/`：`operation_handles.py` = async Handle/Cancel facet（mint token + `create(cancel_hook=)`/settle/await_outcome/poll/cancel/message/stop/cancel_all/live_count/has_cancel_hook）+ per-op OperationChannel（單一有序事件 FIFO Settled/Message/Stop，取代舊 FeedbackInbox + poll-loop，ADR-0025）；`operation_runner.py` = OperationRunner（唯一 kind-agnostic 生命週期機制，ADR-0026 §1）+ OperationSpec policy。run/device/connect/analyze 共用、零 kind）
│   ├── shutdown.py       — ShutdownCoordinator（Qt-free：cancel-all + 輪詢等停 + timeout，begin/tick→state，ADR-0003）
│   （SoC connect / predictor / device 等 session-core service 住共用層 `gui/session/services/`：`connection.py` `SoCConnectionService`（SoC connect op、OperationRunner client、`_ConnectWorker` QThread 已刪，ADR-0026 §5）+ `predictor.py` `PredictorService`（純計算，predictor load/predict/批次曲線）。measure 經 `get_soccfg()`/`has_soc()` 讀；硬體摘要 `get_soc_info()`（describe_soc + `soc.info` RPC）是 Controller façade method（controller.py）讓 agent 讀）
│   ├── scopes.py         — `figure_ambient`（app 層 ambient scope helper，ADR-0026 §2）：把 matplotlib routing ContextVar + `QtLivePlotBackend` 一起裝進 worker thunk（co-dependent facet，Qt-specific 故留 app 層；session 層的 `progress_ambient` 在 `gui/session/scopes.py`）
│   ├── context.py        — ContextService (專案與 md/ml 狀態；from_raw 反序列化；coerce_md_value)
│   （DeviceService 住共用層 `gui/session/services/device.py`：connect/disconnect/setup 經 OperationRunner client、driver factory、cached DeviceSnapshot、registry boundary（DeviceRegistryPort）；領域 policy（rollback/簿記/snapshot）保留為領域 service，ADR-0026 §6）
│   ├── workspace.py      — WorkspaceService (tab lifecycle + capture_session/apply_session + RestoreReport；session_codec 為其內部 raw↔live 實作)
│   ├── caretaker.py      — PersistenceCaretaker (Driven Adapter：單檔 gui_state_v1.json disk I/O + load/flush 時機；ADR-0015)
│   ├── persistence_types.py — AppPersistedState memento (pydantic v2 frozen) + APP_STATE_VERSION
│   ├── session_codec.py  — tabs raw↔live codec (Workspace 內部，Caretaker 只見不透明 cfg_raw)
│   ├── startup.py        — StartupService (typed startup requests + capture_startup/restore_startup；無狀態)
│   ├── run.py            — RunService (執行與進度條；OperationRunner client：組 OperationSpec、cancel-partial 判讀在 on_terminal、寫 State 經 TabResultWritePort)
│   ├── load.py           — LoadService (Analysis tab / remote 載入 canonical result：呼 adapter.load、替換 run_result、清 analyze/post/writeback；不碰 SoC、不反填 cfg_snapshot)
│   ├── staged_analyze.py — _StagedAnalyzeService (analyze/post-analyze 共用基底：OperationRunner client（exclusion=None，handle-only）+ 主線程 record result/figure 經 TabAnalyzeWritePort + 失敗路徑)
│   ├── analyze.py        — AnalyzeService (主分析層：FIT worker + INTERACTIVE finish；算 writeback items)
│   ├── post_analyze.py   — PostAnalyzeService (第二分析層，鏡像 AnalyzeService：FIT-only、在 primary analyze 結果之上重算、gate on primary 已存在；State 平行 post_* 欄位)
│   ├── tab.py            — TabService (分頁狀態與 tab-local query/update)
│   ├── tab_view.py       — TabViewService / TabViewSnapshot (pure tab render read model)
│   ├── save.py           — SaveService (資料/圖片儲存 pipeline)
│   ├── writeback.py      — WritebackService (分析結果寫回 md/ml)
│   ├── cfg_editor.py     — CfgEditorSession (aggregate root：set_field/commit_schema 行為上身，只到 CfgSchema 快照) + CfgEditorService (Repository：lifecycle/LRU/變更流)；commit 把 CfgSchema 交 ContextWritePort 寫 (ADR-0006，session 不再 lower/register)
│   ├── arb_waveform.py   — ArbWaveformService：qubit-scoped arbitrary waveform asset adapter；共用 `meta_tool.ArbWaveformDatabase`，負責 preview PNG、resource version bump 與 GUI/MCP 邊界
│   ├── ports.py          — driven-adapter / sibling-service ports (Protocol)：PersistOriginator(Caretaker↔Controller 窄介面)/ProjectIO/DriverFactory/ContextRead/ContextWrite(+ContextWrites batch)/WritebackQuery/TabLifecycle/StartupContext/RememberedDevice/**TabResultWritePort**(run policy 對 State 的窄 write：clear_tab_results/set_tab_running/update_tab_result)/**TabAnalyzeWritePort**(analyze/post policy 的窄 write：set_tab_analyzing/update_tab_analyze/update_tab_post_analyze)。app service / op policy 依賴 port 而非具體 infra/sibling/`State` (ADR-0005/0006/0026 §3，`State` 是唯一 implementer)
│   └── remote/           — RemoteControlAdapter：第二個 driving View，是共用 NdjsonRpcEndpoint 上的 router；socket/NDJSON-over-TCP transport 住在共用層 zcu_tools.gui.remote（見下「共用 transport 層」）
│       ├── method_specs.py — METHOD_SPECS 契約表 (wire 參數型別 SSOT)；MethodSpec 型別來自 gui.remote
│       └── dispatch.py     — BoundMethod 綁 handler→METHOD_SPECS；METHOD_REGISTRY
│       （measure-gui MCP entrypoint 已搬出本套件，住在 zcu_tools/mcp/measure/server.py：guard bridge + override tools + 自己的 stdio loop，建在 gui.remote McpBridge 上）
└── ui/
    ├── cfg_form.py         — CfgFormWidget：LiveModel 反應式容器
    ├── fields/             — 渲染邏輯：registry.py / common.py / containers.py
    ├── inspect_dialog.py   — InspectDialog(InspectDialogBase 子類)：補 Arb Waveforms top-toolbar 入口與 ml create/modify（_MlCreateDialog/_MlModifyDialog 拖 CfgEditor）；md tab + ml view/rename/del 在 base（session）
    ├── arb_waveform_dialog.py — ArbWaveformDialog：管理 qubit-scoped arbitrary waveform `.npz` asset；支援 formula segment insert/delete、normalize toggle、保存、rename/delete、debounced I/Q/Abs preview（Normalize 只決定是否 peak-normalized；y 軸一律依目前畫出的資料自動縮放）；新建 draft 預設為兩側 half-Gaussian 的 flat-top recipe；ML waveform 建立仍走 Inspect 的正常 create/modify 流程
    ├── main_window.py      — MainWindow(QMainWindow) 實作 ViewProtocol；toolbar 保留 Setup / Devices / Predictor / Inspect（Arb Waveforms 入口在 Inspect 內）；named dialog registry 對 Predictor 採 persistent hide-on-close（重開不重算曲線），其餘 dialog 仍 close 後釋放；持 FeedbackPanel（`refresh_feedback_widget` 依 (live op 數 且 `ctrl.has_agent_connected()`) 把單一 app-level panel mount 進 target tab 的 plot_layout / unmount；target tab = running tab，無則 active tab；tab 變則 re-mount；`ExpTabWidget.mount_feedback_panel`/`unmount_feedback_panel` 為 host API）+ `open_notify_prompt` 開 NotifyUserDialog
    ├── feedback_widget.py  — FeedbackPanel(_CollapsibleSection)（docked 在 figure 下方、可摺疊的「Send to agent」section，預設展開；非 overlay）；user→agent nudge / Send & Stop；Stop 鈕依 active op 是否有 cancel hook gating，`Controller.can_cancel_active_operation`→`OperationHandles.has_cancel_hook`→`OperationChannel.can_cancel`，無 op-kind 知識，ADR-0025 §Stop-gating；unmount 時 clear_input
    ├── notify_dialog.py    — NotifyUserDialog（`gui_prompt_user` 的 non-modal prompt；dialog 是 timeout SSOT，QTimer fire→`ctrl.timeout_notify`；Reply/Dismiss/window-X/timeout 各呼一次 NotifyChannel producer，ADR-0025）
    └── analyze_form.py     — AnalyzeFormWidget：扁平 analysis 參數表單
（共用件已下放 session：setup_dialog/device_dialog/predictor_dialog/inspect_base 在 `gui/session/ui/`（吃 `SessionControllerPort`）、ProgressService/IOManager 在 `gui/session/services/`、QtProgressTransport 在 `gui/session/adapters/`、TrimDoubleSpinBox 在 `gui/widgets/spinbox.py`。measure 保留 app-local OperationGate（policy）+ 直接持共用 BackgroundRunner（executor）+ 自己的 cfg-editor/role-catalog/inspect ml-edit）
```

## Key Design Decisions

### 雙 Client 對稱與 Guard（Permit / Lease）

- View（Qt UI）與 RemoteControlAdapter（NDJSON RPC）是兩個平級 client，受保護操作必經同一帶 guard 路徑，行為規範一致。
- `GuardService` 是 domain guard 的單一所有者，發放型別化 **Permit**（`RunPermit` / `SavePermit` / `AnalyzePermit` / `WritebackPermit`）。受保護 service 方法（`RunService.start_run` 等）只收 Permit，不自行重查前置條件；pyright 在編譯期擋住「拿錯 / 沒拿 permit」。
- **Permit = 靜態前置**（context readiness、committed-cfg validity、SoC capability、adapter 可選 `validate_run_request` 純 preflight）；純憑證、無需釋放。`RunPermit` 攜 `RunRequest` + committed `CfgSchema` + adapter，validity 在 acquire 時 lower 一次並執行 adapter preflight（若 adapter 提供），fail-fast 後才開 async operation handle。
- **Operation = token + opt-in facets（ADR-0019）= 動態互斥/handle 拆兩個 sibling leaf**（取代舊 `OperationGate` 統合 façade + `OperationLease`）：
  - **`OperationGate` = 純 Exclusion**（hardware 排斥；`is_tab_busy` 仍歸此語意但在 service 查）：`ensure_can_start(kind)`（fail-fast guard，conflict raise，**在開 handle 前**呼叫，故衝突不留半成品）+ `register(token, kind, owner_id, resource_id)` + `release(token)` + `has_active`/`is_device_mutating`，keyed by token。只 run/device/connect 用。
  - **`OperationHandles` = async Handle/Cancel facet（per-op `OperationChannel`，ADR-0025）**：`create(cancel_hook=None) -> token`（mint operation_id + 開該 op 的 channel）+ `settle(token, OperationOutcome)` + 動詞 `await_outcome`（off-main 阻塞、消費 channel）/`poll`（非阻塞）/`cancel`/`message`（nudge）/`stop(reason)`（Send & Stop）/`cancel_all() -> list[token]`/`live_count()`/`has_cancel_hook()`。run/device/connect **和** analyze/interactive 共用（**analyze 只拿 handle、不拿 exclusion**）。
  - 終端：domain → `handles.settle(token, outcome)`（喚醒 awaiter）→ `gate.release(token)`（放 hardware，若有）。token = 上 wire 的 `operation_id`。
  - **跨線程互動走單一有序 channel（ADR-0025，取代舊「多 channel + 時序敏感 combine」）**：一個 op 一條 `OperationChannel`（thread-safe FIFO，承載 typed 事件 Settled/Message/Stop，consumer 依到達序消費，race-free + deadlock-free by construction）。**cancel 經 channel 的 cancel hook 觸發**（`create(cancel_hook=)` 時註冊，封裝 op 間差異：run/device = `stop_event.set`、interactive = 直接 settle、connect/FIT-analyze = `None`）；`stop(reason)` 是「enqueue Stop + 觸發 hook」一次原子操作。無 cancel hook 的 op（connect）→ stop no-op（靠 shutdown timeout 兜底）。`OperationChannel.can_cancel` 讓 floating widget 的 Stop 鈕依 cancel-hook gating，**不看 `stop_event.is_set()`**（互動 analyze 的結構洞消失）。見 `docs/adr/0025`/`0019`（承 `0002`/`0003`）。
- Permit 模式只涵蓋 run / save / analyze / writeback；device mutation 前置幾乎全是動態 hardware 互斥，續用 OperationGate，不套空殼 Permit。
- 決策脈絡見 `docs/adr/0001-permit-lease-typed-guard.md`、`docs/adr/0002-version-table-async-handle-off-main.md`；術語見 `CONTEXT.md`。

### Service 接線與 Remote View 投影

- `build_app_services()` 集中建構/接線**所有** domain service（含 `CfgEditorService`），回傳 frozen `AppServices` bundle；Controller 持 bundle 並 alias 到 `self._xxx_svc` 供 façade 呼叫。單一 `OperationGate`（Exclusion）+ 單一 `OperationHandles`（Handle）+ 單一 `OperationRunner`（kind-agnostic 生命週期機制，ADR-0026 §1）共享（ADR-0019/0026）；session-core service 經 `build_session_services(..., runner=..., device_registry=...)` 組（soc_connection/predictor/context/device/startup）。是唯一的 composition root：在此把具體 driven adapter / sibling service 注入給依賴 **port** 的 app service（structural Protocol，零 runtime 包裝）。`CfgEditorService(cfg_editor_ctrl=self, read_port=self, write_port=self, version_bump=self.bump_editor_version)`——`CfgEditorHost` 組合 facet（reactive env + `ContextReadPort` + `ContextWritePort` + bump），Controller 全部實作。`WritebackService` 也注入 `write_port=self`（ADR-0006 batch apply）。

### Service 角色（DDD/Hexagonal，見 `docs/adr/0005`）

按**角色**而非話題聚合。**App Service**（被動編排，依賴 port 非具體 sibling/infra，無 app-service→app-service import，由 `test_app_service_decoupling` AST gate 守）；**Aggregate Root**（`CfgEditorSession` / `TabState` / `DeviceState` / `ExpContext` 帶**自己的行為**：set_field/commit、is_busy/has_run_result/effective_save_paths、is_connected/is_live、has_context/is_active）；**Repository**（`CfgEditorService` 管 session lifecycle）；**Driving Adapter**（`MainWindow` + `RemoteControlAdapter` = 兩個 user-facing client，共用 Controller façade）；**Driven Adapter**（persistence/driver/IOManager，只經 port 被呼叫）。State 容器的 mutator + version bump 仍是 State 職責（主線寫入不變式），aggregate 謂詞是 query-only。

- **View 三渠道（ADR-0013）**：Controller 把 `ViewProtocol` 拆成 `DiagnosticSink`（多個，fan-out，`notify_diagnostic(severity,title,message)`，severity∈error/info）+ `RenderHost`（單一可選，pbar/container，start_run/analyze 用；headless 為 None 容忍）+ `RenderView`（snapshot/screenshot/dialog）。`set_view`→`add_view`（持 `list[DiagnosticSink]`+`Optional[RenderHost]`）。診斷 fan-out **不經 EventBus**（報告故障的通道不該是故障系統）。render/snapshot/dialog 查詢由 `RemoteControlAdapter` 持 `render_view`（=MainWindow，app.py 注入）**直接拉、不經 Controller**——`ViewQueryService` 已刪。cfg 欄位編輯走 tab 的 `CfgEditorService` session（`editor.set_field` on editor_id from `tab.snapshot`，ADR-0013 F11）。
- tab cfg 編輯的 run-guard 在 `_h_editor_set_field`：`owner_of_editor(editor_id)` 反查 owner（tab cfg session 的 owner=tab_id），若該 tab 正在 run → `precondition_failed`，與人靠 `cfg_form.setEnabled(idle)` 同義；ml-entry/owner-less session 不受擋。

### Wire Schema SSOT 與 MCP 生成

- `services/remote/method_specs.py`（Qt-free）持 `METHOD_SPECS` 契約表，是 wire 參數型別的單一來源；`MethodSpec`（timeout/description/`params`/`tool_name`，不含 handler）型別本身與 `BoundMethod`/`build_method_registry` 住在共用層 `gui.remote.method_spec`（見下「共用 transport 層」）。
- `dispatch.py` 以 `BoundMethod` 將 handler 綁到 `METHOD_SPECS`；import 時 fail-fast 檢查 handler/spec key 集合一致。
- `ParamSpec`（共用層 `gui.remote.param_spec`）同時驅動執行期 per-param 驗證（service 在 dispatch 前驗）與 MCP `inputSchema` 生成（`build_input_schema`）；兩者不漂移。handler 收到已驗證的 typed params，不再呼叫 `_require_*`。
- `zcu_tools/mcp/measure/server.py` 從 `METHOD_SPECS` 生成 1:1 RPC tool 的 schema + forwarder；手寫 override 僅留 lifecycle / fan-out / 檔案寫入 / 多欄位 coercion（connect/device/startup）。`coerce_*`（multi-param → frozen request）仍由 handler 顯式呼叫。MCP 端的 socket transport（連線/送 RPC/reader thread/RID 路由）由共用層 `McpBridge` 持有，measure-gui server 在其上加 guard bridge、operation tracking、override tools 與自己的 stdio loop。
- 該 server 以 script 啟動，import `method_specs` 會經 `gui/__init__` 拉進 Qt；bridge 容忍此（不建 QApplication），換取與 dispatcher 共用單一 SSOT。
- **lazy auto-connect**（measure-only policy，在 `send_gui_rpc`、非共用 bridge）：首次 `gui_*` 工具呼叫時若未連線，自動經 session-discovery 解析 control-socket port 並 attach（連不到才 raise）。讓 `gui_bridge_connect` 對 agent 與人類使用者皆變可選；外部終端 agent 透過 loopback mcp.json 啟動後，第一個 gui_* 呼叫即自動連上 GUI；**只 attach 控制通道、不碰 SoC `soc.connect`**（SoC 連線仍是使用者決定）。

### 共用 transport 層（`zcu_tools.gui.remote`，三 GUI app 共用；domain 仍各自）

measure / fluxdep / dispersive 三個 GUI app 共用 transport + wire 機制，domain 邏輯仍各 app 自有：

- **GUI 端**：`NdjsonRpcEndpoint` 持 socket 生命週期 + NDJSON framing + per-client writer/outbound queue + 內建 wire.version/auth handshake + reply 編碼 + `broadcast(line, predicate)` push fan-out + main-thread marshal primitive。**router scaffolding 也共用**：`RemoteControlServiceBase`（`gui.remote.control_service`）擁有 `route` 骨架 + events.* handlers + `_dispatch_on_main`（含 off-main + `_guard`/`_after_success` 兩 seam）+ EventBus subscribe/serialize/broadcast。各 app 的 `RemoteControlAdapter` **subclass 它**：fluxdep/dispersive 零覆寫（read-only），measure-gui 覆寫 policy seams（guard / editor lifecycle / diagnostic / `_get_bus`）——**adapter 是 router 不是 socket owner**。
- **MCP 端**：`McpBridge`（class，socket state 是 instance 屬性）持連線/送 RPC/reader thread/RID 路由/pending map/pid files，注入 config + `on_event` hook。read-only apps 的 `zcu_tools/mcp/<app>/server.py` 是 thin entrypoint：填 config + instructions，body 全交 `core.readonly_server.build_readonly_server`（共用 `send_gui_rpc`/lifecycle tools/cleanup/`run_stdio_loop`）。measure-gui 在 bridge 上額外疊 guard bridge / operation tracking / override tools，但 stdio loop **也走共用 `run_stdio_loop`**——measure-only 行為（logging / diagnostic piggyback / reason-tag error / serverInfo version）經 `on_start` / `on_each_reply` / `on_error` / `server_version` hook 注入。
- **wire 契約**：`MethodSpec` / `BoundMethod` / `build_method_registry`（`gui.remote.method_spec`）與 `ParamSpec`（`gui.remote.param_spec`）是共用型別；各 app 自填 `METHOD_SPECS` 契約表。
- 版本 guard、operation-handle 簿記等 measure-gui 專屬 policy **不**下放到共用層；共用層只提供 transport mechanism（三層分工：RPC=mechanism、mcp=簿記+翻譯、agent 只收語義）。

### Agent 體驗（事件 / 錯誤 / 發現）

#### Agent launch UI 已退役（ADR-0024）

measure-gui 不再提供 toolbar「Agent」按鈕、`AgentLaunchDialog` 或 `services/agent_launcher.py`。Agent session 由外部 CLI/MCP 入口自行啟動並透過既有 measure-gui control socket 連線；GUI 端只保留 runtime control surface，不再負責 spawn terminal、記錄 resumable session 或注入 bootstrap prompt。
- **跨線程互動（user↔agent，ADR-0025）**：cooperative-interrupt 走 per-op `OperationChannel`（取代舊 ADR-0023 `FeedbackInbox` + poll-loop）。GUI 端的 `FeedbackPanel`（docked 在 target tab figure 下方的可摺疊 section，**只在有 live op 且 MCP control client 連線時 mount**＝agent 在驅動時才出現，C3）讓 user 在 op 進行中對 active op 注入 nudge（`Message`，op 續跑、agent 收 `user_feedback`）或 Send & Stop（`Stop(reason)`，依 cancel-hook gating）；`await_outcome` 消費 channel 依到達序折疊成 AwaitResult。client 連線/斷線經 `NdjsonRpcEndpoint.has_live_client()` + queued dispatch 在主線程刷新 widget。反向（agent→user）走獨立的 `NotifyChannel`（`gui_prompt_user` → notify.open/await wire → `NotifyUserDialog` prompt，事件 Reply/Dismiss/Timeout）。

- **Run / Device-setup 生命週期（對齊，wire v10）**：`RUN_STARTED{tab_id}`+`RUN_FINISHED{tab_id,outcome,error_message}` 與 `DEVICE_SETUP_STARTED{name}`+`DEVICE_SETUP_FINISHED{name,outcome,error_message}` 同構（一事件一語義+outcome，拆自舊 RUN_LOCK_CHANGED / DEVICE_SETUP_CHANGED）。進度走 `operation.progress(operation_id)`（run+device 通用，按 id 查；`{active,bars}` 形狀，主線程 `ProgressBarModel` 實時算 format/percent + raw n/total；折進 `gui_*_poll` running 回傳）；進度純拉不發 event（高頻），完成才發 *_finished。**progress 由唯一 `ProgressService` 持有**（Phase 111，見下「Progress 子系統」）：container 生死綁 operation（discard_operation 於終局），View 經 owner_id（tab_id/device_name）attach 一次、跟隨 operation 輪替 → 關 dialog 不銷毀進度（container 在 service 仍活，重開 re-attach 即見）。
- **可讀 id**：`tab_id` = `<adapter-slug>-<hash>`（如 `twotone-freq-1a2b3c4d`），owner-keyed `editor_id` = `<tab_id>-ed`（agent 不必查 snapshot 即知 editor 屬哪 tab）；仍是不透明唯一 string key。
- **錯誤原因**：precondition 失敗的 `RemoteError` 帶 `reason`（machine tag，如 `no_run_result`/`no_active_context`），來源是 `GuardError.reason_code`；`ErrorCode` enum 不變，agent 以 `reason` 分支決定修復動作。
- **cfg path 發現**：`tab.get_cfg` 讀該 tab 的 `CfgEditorService` session（經 `editor_id_for_owner(tab_id)` → `cfg_editor_get`），回 `build_settable_tree` 產出的**巢狀 value tree**（而非舊的 flat `{path, kind, value, type, choices?}` 列表）。葉節點以 `$`-前綴鍵區分型別：scalar 葉 = 裸值（`null`=未設）；enum 葉 = `{"$value": v, "$choices": [...]}`；sweep 節點 = `{start, stop, expts, step}` 裸 edge 值；ref 節點 = `{"$ref": {"current": "key", "options": [...]}, ...當前 variant 子樹...}`（只展開 current variant；`$ref.options` 是 bare 名列表，`set_field` 接受 bare 名）。`prefix`（dotted）回該節點子樹（不命中回 `{}`；指向 sweep edge 回整個 sweep 節點）。**`verbosity` 與 `under` 參數已移除**。每個 tree 中的葉路徑保證可被 `editor.set_field`（dotted path，語法不變）設定；form 未 populate（無 session）→ `precondition_failed`。
- **adapter 規格查詢（Phase 169 移除）**：`adapter.cfg_spec` / `adapter.analyze_spec` 兩個 wire method 已連同 `build_spec_tree` 系列 helper 移除（冗餘 agent 工具收斂）。要讀某 adapter 的 cfg 形狀或 analyze 參數，build 一個 tab 後走 `tab.get_cfg`（巢狀 value tree）/ `tab.get_analyze_params`（當前 analyze 參數）。adapter 的 `cfg_spec()` classmethod 仍是 spec 層內部 API（`make_default_cfg` 用），不受影響。
- **writeback workflow（persistent draft，ADR-0008）**：analyze 完成時一次算出 items 存 `TabState.writeback_items`，preview/UI/agent/apply 全讀/改同一份（**不重算**，重算會丟編輯）。三種 item 共有 `target_name`（套用目標名，可改）＋ `description` ＋ `session_id`（`<kind>-<n>` 識別碼，service stamp，與 target_name 解耦）。metadict 帶 `proposed_value`；module/waveform 帶 `editor_id`——指向一棵 gc=False CfgEditorService model（種子=edit_schema），agent 經 `editor.set_field(editor_id,…)` 改、user 點 Edit 時 widget attach 同一 model（WYSIWYG）。`WritebackWidget` 的 Edit dialog 兩種都有「Apply as」欄位寫 `item.target_name`（與 session_id 解耦的可改套用目標名）：metadict dialog Save 時驗證非空，module/waveform dialog `editingFinished` 即時提交、空白回退原名。`writeback.preview(tab_id)` 回 items（id/target_name/kind/selected/description；metadict 加 proposed_value；module/waveform 加 editor_id+has_edit_schema）；`writeback.set(tab_id, id, …)` 改 selected/target_name/proposed_value（cfg 走 editor.*）；`writeback.apply(tab_id)` 讀持久草稿、apply 用 target_name、snapshot 各 item 的 model→lower、**不收 selections**、回 applied_ids。rerun/reanalyze 先 `teardown_tab_items` 再重算。`applied_session_ids` 記已套用。adapter 只給 target_name/description/值；session_id/editor_id 由 service stamp。
- **連線/專案拆分**：`soc.connect`（kind+ip?/port?，**同步** RPC）與 `startup.apply`（chip/qub/res+可選 result_dir/database_path）皆 ParamSpec 生成；agent 端的 mock 連線走 `gui_soc_connect(kind='mock')`。省略 result_dir → DRAFT context。
- **共享 cfg 編輯（CfgEditor session，ADR-0008，取代舊 delegated 設計）**：`editor.new/set_field/get/commit/discard` + `editor.subscribe/unsubscribe`。**model 永遠 service-owned**；`CfgFormWidget` 是可插拔 viewer——`attach(model)` 顯示+反映、`detach()` 走但不 teardown。生命週期按 `gc`:`open(gc=True)`(agent 自開 ml entry,commit lower→ml,LRU+斷線回收)與 `open_seeded(seed, gc=False)`(tab cfg/inspect/writeback 草稿,無 item_kind→teardown-only,owner 顯式 `teardown(editor_id)`,不受 LRU/斷線)。widget 先 detach service 再 teardown。**external refresh 歸 service**(ADR-0004 Reaction):service 訂 MD/ML/CONTEXT/DEVICE_CHANGED 對每棵 owned model `refresh_external` 刷 EvalValue,widget 經 model on_change 免費重畫,widget **不**碰 EventBus。`set_field` 回被改 path 子樹;scalar 可填 `{__kind:eval,expr}`,commit lower 成 concrete。**editor 專屬變更流**(獨立 EventBus):任一 client 改 → 訂閱者收 `editor_changed`/`editor_closed`(reason)。tab 的 editor_id 經 `tab.snapshot` 暴露。見 `docs/adr/0008`。
- **並發感知(資源版本表 + 版本 guard,Phase 94/95;Phase 100 補完盲區)**:`State.version`(`VersionTable`)per-resource 單調遞增,owner service 在主線 bump。run/save/writeback.apply/editor.commit 帶 optional `expected_versions`(mcp 由 `_GUARD_DEPS` 填,wire-only/MCP-hidden),`_guard_versions` 原子比對不符即 `PRECONDITION_FAILED`。**覆蓋面**:md/ml 語義寫入 bump `context`(ContextService 8 method、三 context-switch、**以及 writeback.apply 直接寫 md/ml 處**——故 run/commit/writeback 偵測得到 md/ml 被改);run/analyze result 各有 `tab:<id>:result`/`tab:<id>:analyze` key;writeback.apply 依賴 result+analyze+context。`State.set_context` 是**純欄位替換**(soc/predictor 也用),**不** bump context(避免 SoC 連接/predictor 載入誤封依賴 context 的 op);soc 有獨立 `soc` key。editor session teardown `drop_prefix("editor:<id>")`,對齊 tab/device。device 集合用 `devices:__set__` 基數 key（成員新增/移除才 bump,狀態/info/remember 編輯不動）補 `device:*` glob 對「集合新增」的盲區（Phase 102）。通知面正交走 EventBus subscribe/poll(不帶版本號)。三層:RPC=mechanism、mcp=簿記+翻譯、agent 只收語義。細節見 `services/remote/README.md`、`docs/adr/0002`。

### Spec / Value 與 Lowering

- `gui/specs/` 只暴露 fresh factory function，不暴露共享 `CfgSectionSpec` 實例
- **Spec fluent 覆寫（spec 層，ADR-0009）**：`CfgSectionSpec`/`ModuleRefSpec`/`WaveformRefSpec` 皆有 `lock_literal(path, value)`（換 `LiteralSpec`，鎖定）回**同型新 frozen spec**（spec never-mutated）；path duck-type 穿透 `ModuleRefSpec`/`WaveformRefSpec.allowed`（含則套、全不含才 raise），**並可下鑽巢狀 ref**（`_with_override`/`_path_exists` 三型互遞，能鎖 `qub_pulse.waveform.length` 這種波形內 leaf）。**鏈式起點可從根或子樹**：對 helper 回的子樹直接鎖（`make_pulse_readout_module_spec().lock_literal("pulse_cfg.freq", 0.0)`，path 較短、內聚）優於從根走長 path。**必須在 `cfg_spec()` 內呼叫且結果被 `return`**（鎖是 spec 契約，cfg_spec 是唯一所有者；外部鏈式違反責任分離）。鎖定屬 spec 層、非 value 的 0.0+editable=False。`LiteralSpec` 由 widget 一律不畫（`containers.py`，spec 不帶 hidden 旗標）。`readonly`/`hidden` 無使用者已移除（要 editable=False 直接 `ScalarSpec(editable=False)`）。
- **Value OO 覆寫（value 層）**：`CfgSectionValue`/`ModuleRefValue.with_field(path, value)` **in-place 改回 self**（取代 default factory 長參數列）；刻意不對稱於 spec 層（value 樹本就可變）。fluent 方法（`with_field`/`lock_literal`/`_with_override`）回傳 `Self`（綁定呼叫者具體型別，非硬編字串）；`with_field` 的 value 強型別為 `ScalarLeafInput`（`int|float|str|bool|DirectValue|EvalValue`）、`lock_literal` 的 value 為 `object`（進 `LiteralSpec.value`）。
- **每角色 default factory（`shared/defaults/` 每角色一檔）**：每角色暴露 `make_<role>_default`（blank,md 衍生預設、不查庫）+ `make_<role>_ref_default`（查庫 preferred → fallback blank;**`optional=True` 查無回 `None`**——ADR-0010 的停用態（裸 `None`，取代舊 `DisabledRefValue`），adapter 直接放進 `modules` fields，無 `if x is not None` guard，與 optional spec 對稱；`fields` 型別為 `dict[str, Optional[CfgNodeValue]]`）。adapter 直接呼叫,逐欄位選 ref（校準後引用庫）或 blank（spectroscopy 新脈衝）。共用工具在 `defaults/helpers.py`（`patch_pulse_fields`/`patch_ro_cfg_fields`/`make_trig_offset`/`make_default_value`/`select_named_module_value`/`select_named_waveform_value`）。**零鎖定**（鎖歸 adapter cfg_spec）。fake/demo adapter 在 `adapters/fake/`（registry key `fake/*`，與真實驗選單分開）。
  - **md 衍生欄位走 Eval 模式**：factory 讀 md 一律經 `md_scalar_float`/`md_scalar_int`（ctx_helpers）——md 有 key → `EvalValue(expr=key)` 保留表達式（resolved 交給 lowering）、無 → `DirectValue(fallback)`。**不**用裸 `md_get_*`+無條件 DirectValue（那會寫死數值、丟掉表達式）。`gain`/`length`/`nqz` 等常數仍 DirectValue。
  - **形狀明確 vs role 預設**：role 有多形狀者（readout: pulse/direct;reset: none/pulse/two_pulse/bath）各形狀有**獨立具名 factory**（`make_pulse_readout_default`/`make_direct_readout_default`、`make_pulse_reset_default` …），`make_<role>_default` 是 thin alias → 該 role 的預設形狀（readout/reset 皆 = pulse）。spec 只允許單一形狀的 adapter（如 onetone 用 `make_pulse_readout_module_spec`）**必須呼叫形狀明確 factory**，不可靠 role 預設「正好是該形狀」（最小驚訝）。
  - **角色清單**：`qub_probe`（qub_ch/q_f pulse）/`res_probe`（res_ch/r_f pulse,**無 ro_cfg**,CKP/AC-Stark res 側）/`pi_pulse`（ref 查 pi_amp/pi_len,blank=qub pulse）/`pi2_pulse`（ref 查 pi2_*→pi_*）/`readout`（blank=inline pulse+ro_cfg,含 ro_waveform 庫邏輯;ref 查 readout_dpm/rf）/`reset`（含 none/pulse/two_pulse/bath 子型 blank;ref 查 reset_bath/reset_10/reset_120）/`qub_waveform`（ref 查 qub_flat/qub_cos,blank cosine）/`res_waveform`（ref 查 res_flat/res_const,blank const）。
  - **ref vs blank 慣例**：spectroscopy 用 blank（`make_qub_probe_default`,探測是新脈衝、freq 常被 sweep 鎖 0.0）;校準後 t1/t2/rabi 用 ref（`make_pi_pulse_ref_default`/`make_readout_ref_default`）。preferred_names 按角色名慣例,使用者存庫時用該名。
  - （舊三層 `module_value_defaults`/`module_ref_defaults`/`role_defaults` 的 `default_*` wrapper 已刪,收斂為單層每角色檔。）
- **role catalog（憑空建 ml entry「from template」，user+agent 對稱，唯一 create 入口）**：`gui/role_catalog.py`（`RoleEntry`/`RoleCatalog`,只 import gui.adapter）是介面；`experiment/v2_gui/registry.py::register_all_roles` 啟動填充（與 `register_all` 同檔,鏡像；entry script `run_gui.py` 建空 catalog 填充後傳給 `run_app`,gui 框架自身不 import 實驗層）。**兩類 role**：(1)**md-aware**（res_probe/bath_reset…，`make_<role>_default`,eval 預設）;(2)**`:blank`**（`<disc>:blank`,如 `pulse:blank`/`reset/bath:blank`/`drag:blank`,結構零值 = `make_default_value(spec)` 包成 Ref）——每 discriminator 一個(7 module + 6 waveform),涵蓋無 md-aware role 的形狀(裸 pulse、drag/flat_top/gauss/arb waveform)。`ALL_ROLE_ENTRIES` = md-aware 先、blank 後。`Controller.create_from_role(item_kind, role_id, name)`：跑 factory → 讀 value 的 type/style disc → 查 spec → `set_ml_*_from_schema(CfgSchema(spec, value))`（ADR-0006：lowering 收進 ContextService，create 端不再自己 lower）。**一次性 create**,不開 editor session、本輪不 edit-before-commit；改內容走正常 modify 路徑。**撞名 fail-fast**：`_require_new_ml_name`（create=新建;`editor.commit` 仍 upsert）。雙端：user 經 inspect「Create…」按鈕 + `_MlCreateDialog`（選 role+名→直接建）；agent 經 `context.ml_list_roles`（發現）+ `context.ml_create_from_role`（建）。**`editor.new` 已 from_name-only**（modify 既有 entry）——憑空建 RPC 只走 `context.ml_create_from_role(role_id="<disc>:blank")` 再 `editor.new(from_name)`；`discriminator` 從 RPC 移除（內部 `service.open(discriminator=)`/`_initial_schema` 分支保留作 seed,僅內部可達）。
- **inspect ml 入口拆分（create vs modify 彻底分開）**：`_MlConfigDialog`(舊 add/modify 用 `mode` 糊在一起)拆成 `_MlCreateDialog`(role picker,唯一 create)＋ `_MlModifyDialog`(編輯既有 entry,**name/type 唯讀=固定形狀,不換 type**;要換形狀=刪了重建)。toolbar 3 按鈕(Add Module/Add Waveform/From template)收成單一「Create…」。
- writeback 採 typed items + `WritebackService`；module / waveform writeback editor 直接重用 `CfgFormWidget`
- spec/value builder 必須與 domain config 欄位名一致；漂移會在載入時直接 invalid
- **`CfgSchema.validate(ml)` = 靜態合法性檢查邊界**（`lowering.validate_section`，復用 `find_allowed_spec`）：結構完整（每 spec key 有 entry）+ **LiteralSpec value==spec.value（不一致 raise，含 None → raise「is None but not a disabled optional ref」）** + DirectValue scalar 型別（**int→float widen OK、float→int reject、bool/str 嚴格**）+ choices + None 只能在 optional ref。**EvalValue 跳過**（型別 resolve 時才定）。在「成品邊界」顯式呼叫（`make_default_cfg` 產出、`to_raw_dict` lower 前），**不在 `__post_init__`**（會誤傷編輯中間態：cfg_form/editor draft/codec restore 都建 CfgSchema 且值可暫不合法）。**LiteralSpec 一致性的檢查點只在這裡（validate）**：lowering 本體（`_section_to_dict_inner`）對 LiteralSpec **直接用 `spec.value` 產出、不看 value 樹、不比對**（信任輸入已過 validate）——所以「value 樹的 literal 寫錯」由 validate fast-fail，lower 不重複查。三道防線互補：CfgBuilder.build 事前對齊（消除 adapter 重複宣告）／ validate 事後 raise（守手拼、codec restore、editor draft 等非-builder 路徑）／ lowering 信任 spec.value 產出。
- **`CfgSchema.validate_dynamic(md, ml)` = 動態合法性檢查邊界**（`lowering.validate_dynamic_section`）：每個 scalar 必須有值（`DirectValue(None)` → raise）、每個 `EvalValue` 必須能用 md resolve（`evaluate_numeric_expr` + `coerce_eval_result`）、`SweepValue` 的 `EvalValue` edge 同理、每個 `DeviceRefSpec` 必須已選（非空字串）、ref 遞迴。`md` 是必要參數（非 Optional）。由 `to_raw_dict` 在靜態 validate 後、lowering 前呼叫（`md is None` 時跳過）。**不在 `make_default_cfg`**（新建 cfg 的 scalar 本就 unset）。lowering 本身保有重複的動態檢查作為安全網。`_validate_eval` 不復用 `_resolve_eval`（後者有 snapshot 優先/stale cross-check 語義，validate 只確認可 resolve）。
- `CfgSchema.to_raw_dict(md, ml)` 是**唯一** lowering 入口（free function `schema_to_dict` / `_section_to_dict` 已折進此 method 並刪除，Phase 120a）；**lower 前先 `self.validate(ml)`（靜態）再 `self.validate_dynamic(md, ml)`（動態，md 非 None 時）**（不合法 cfg 在產 exp cfg 前 fast-fail）。簽名 `(md, ml)`：md 在前（解 EvalValue），ml 次之（查 library reference）；兩者皆 optional（全 resolved 時可傳 None）。未知 library reference、unsupported cfg value type 直接 fail fast。lowering 是唯一的 EvalValue 解析點：`EvalValue.resolved` 有 snapshot 就用，無則拿 `md` 對 `expr` 求值；無 snapshot 又無 md 才 fail fast。**drift 交叉檢查**：snapshot 與 md 同時存在時，lowering 會拿 md 重新求值與 snapshot 比對（coerce 到 spec type 後比，避免 5 vs 5.0 假陽），不一致 `logger.warning`（snapshot 仍勝，語義不變）。**ADR-0006：所有 ml/md 內容寫入唯一經 `ContextService`（lowering 在此，呼叫端只交未-lower CfgSchema）**。`set_ml_*_from_schema` / `apply_writes`（writeback batch，1-bump/1-emit）內部 `schema.to_raw_dict(md, ml)` lower（傳 md → 呼叫端不可能漏 md）+ `Factory.from_raw` + register + bump + emit。`ContextReadPort`（get_current_ml，editor seed）/ `ContextWritePort`（from_schema×2 + set_md_attr + apply_writes）對稱拆分。editor.commit / writeback.apply / inspect save / create_from_role 四來源端同一條 port。無 public `set_ml_*_from_raw`（agent raw RPC 已移除，見 remote README）。
- discriminator (`type` / `style`) 使用 `LiteralSpec`（widget 一律不畫；所有 `LiteralSpec` 同此待遇）
- `SectionLiveField.__init__` 對 hint 中缺少的 key fallback 到 `make_default_value(spec)`（解決 Combo 切換時新欄位 invalid 問題）

### Adapter Contract

- **框架契約 = `gui.adapter.ExpAdapterProtocol`（generic-free Protocol）**：列出 framework 真正呼叫的必備成員（`cfg_spec`/`guide`/`make_default_cfg`/`make_save_paths`/`run`/`analyze`/`save`/`get_analyze_params`/`analyze_params_cls`/`get_writeback_items`/`capabilities` 等），無實作、無 generic。GUI 一律以不帶 generic 的 `ExpAdapterProtocol` 持 adapter（run/analyze/writeback result 過 Qt `Signal(object)` 即成 `object`，gui 從不 narrow）。`BaseAdapter.validate_run_request` 是可選純 run preflight hook，讓 SoC-dependent 但可預測的 cfg 錯誤在 GuardService 階段同步拒絕，不建立 operation handle。
- **`guide()` → `AdapterGuide`（靜態行為導覽，開跑前讀）**：五欄 prose（behavior/expects_md/expects_ml/typical_writeback/recommended），**導覽非契約**——讓 agent/user 概觀「測什麼、讀哪些 md/ml key、寫回什麼、推薦設定」，實際怎麼用是其自由（現在式、含具體 key name + 建議範圍 + 標 optional）。BaseAdapter 給「(no guide written yet)」誠實預設；每個 registered adapter 應覆寫（測試 `test_every_registered_adapter_has_a_written_guide` 守此）。雙端出口：agent 走 `adapter.guide` RPC（→ `gui_adapter_guide`）、user 看 `ExpTabWidget` 左側 Config/Analysis 之後的唯讀第三分頁 "Guide"
- **共用實作 = `experiment/v2_gui/adapters/base.py::BaseAdapter[T_Cfg, T_Result, T_AnalyzeResult=NoAnalysisResult, T_AnalyzeParams=NoAnalyzeParams]`**（PEP 696 default；pyright 1.1.410 + typing_extensions 支援）。四 generic 強型別連動（run→analyze→writeback）活在**單一 concrete adapter 內部**，由 BaseAdapter generic 保證。adapter 繼承 BaseAdapter，structural 滿足 Protocol（非 nominal）
- `exp_cls` 透過 structural `ExperimentProtocol[T_Cfg, T_Result]` 約束；`run(soc, soccfg, cfg, **kwargs)` 與 `save(filepath, result, **kwargs)` 採 `**kwargs` 容納 experiment-specific 擴充（例 `earlystop_snr`）
- `run()` 固定為 `run(req, schema)`；runtime user params 不屬於 adapter API，必須進入 `CfgSchema` 或 adapter constructor
- analyze result 直接攜帶 `figure`；analyze params 用 adapter-owned dataclass instance + `Annotated[..., ParamMeta(...)]`，`Literal[...]` 表示 choices，**`Optional[T]` 表示可留空欄位**（`_resolve_field_info` strip None+flag `optional`；form 渲染**既有 optional-scalar QLineEdit**「(none)」空態+numeric validator，**同 cfg form 的 ADR-0010 optional scalar**；`describe_analyze_params` 標 `optional:true` → agent 傳 `null`，`dataclasses.replace(ap, **updates)` 本就吃 None，**WIRE 不變**——mcp verbatim 轉發 spec/updates 不解讀 optional）
- **無 analysis 的 adapter**：只填兩格 generic（`BaseAdapter[Cfg, Result]`，PEP 696 補 `No*`），宣告 `capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)`，**不覆寫** `analyze`/`get_analyze_params`。BaseAdapter 的預設 body **raise NotImplementedError**（Fast Fail）——`analysis=NONE` 時 framework gate（`tab.py`）擋掉不會走到；`analysis=FIT` 卻忘覆寫時才觸發 raise。(舊 `NoAnalysisAdapterMixin` 已刪)
- `RunRequest`、`AnalyzeRequest[T_Result, T_AnalyzeParams]`、`SaveDataRequest[T_Result]`、`WritebackRequest[T_Result, T_AnalyzeResult]` 是跨 worker 的 immutable 邊界
- 真實 experiment adapter run result 必須攜帶 `cfg_snapshot`；adapter 在 save 邊界以 snapshot + raw result 重建 `last_cfg/last_result`，委派既有 experiment `save()`
- 預設 save path policy 由 `BaseAdapter` 提供；一般 adapter 只需定義 filename stem；`make_save_paths()` 是 pure suggestion，不建立目錄。`SaveService` 是 GUI save flow 的 path reservation boundary：在提交 worker 前用 `reserve_labber_filepath` 決定 final data path，再把 exact path 傳給 adapter/experiment save；低層 persistence 不自動 suffix 或覆寫。路徑結構對齊 notebook（single_qubit.md）：data=`{database_path}/{stem}@{active_label}`、image=`result/chip/qub/exps/{label}/image/{stem}.png`。**日期段（`YYYY/MM/Data_MMDD`）在 `database_path` 裡**（`derive_project_paths` 用共享純函數 `get_datafolder_path` 算出 `Database/chip/qub/YYYY/MM/Data_MMDD`，與 notebook 的 `create_datafolder` 回傳同款，唯不建目錄）——`make_default_save_paths` 直接 join `ctx.database_path`，**不再重複補日期**。`get_datafolder_path`（純）↔ `create_datafolder`（= 純 + makedirs）是 utils 層同一抽出；GUI 走純版本。chip/qub + 日期 scoping 全在 `derive_project_paths` 一處；`apply_project` 不 re-scope。setup dialog prefill 時對持久化的 chip/qub **re-derive**（取今天日期，避免隔天恢復寫進昨天資料夾）
- `build_exp_cfg()` 慣例：flat GUI raw cfg → nested exp cfg dict 重組，交由 `ctx.ml.make_cfg(...)` 統一驗證與 `modules`/`sweep` 轉換。**不屬 Protocol**（只被 `run()` 內部呼叫）。**有預設實作**：設 `ExpCfg_cls: ClassVar` 即繼承 `return req.ml.make_cfg(raw_cfg, ExpCfg_cls)`;`ExpCfg_cls` 未設（None）時預設 body **raise**（Fast Fail，與 analyze no-op 同慣例）。bespoke 映射（fake 自建 cfg、fakefreq 多帶 kwarg、flux_dep pop `dev` 重塑）覆寫 `build_exp_cfg`、不設 `ExpCfg_cls`
- **cfg 靜態/動態拆分**：`make_default_cfg(ctx)` 是 BaseAdapter 的 concrete = `CfgSchema(cfg_spec(), make_default_value(ctx))` + **`schema.validate(ctx.ml)`**。adapter 實作兩個 abstract：`cfg_spec() -> CfgSectionSpec`（**classmethod**、純結構、**不讀 ctx**，供 agent 未建 tab 即查 spec）與 `make_default_value(ctx) -> CfgSectionValue`（讀 md/ml/device 算預設值，**不屬 Protocol**）。`analyze_params_cls()` classmethod 由 `get_analyze_params` 的 return annotation 反射取得 analyze params dataclass（靜態）。
- **adapter 必回傳完整合法 value 樹**：`make_default_value(ctx)` 產的 value 樹**每個 spec key 都要有 entry**（含 `LiteralSpec` = `DirectValue(spec.value)`、停用 optional ref = `None` entry）；`make_default_cfg` 用 `validate` 強制（漏/不符當場 raise，責任指向該 adapter，**框架不補齊**）。`lock_literal` 鎖的欄位 value 也要服從鎖定值（如 onetone/freq `.with_field("pulse_cfg.freq", 0.0)`，因 freq 被 sweep 軸接管；value 帶 EvalValue("r_f") 是與鎖定矛盾的誤導值，validate 抓它）。
- `Controller/State` 是 live SSOT；`ExpContext` 不應視為全域 frozen snapshot

### Adapter Capabilities

- `AdapterCapabilities(requires_soc, analysis: AnalysisMode)` 是 SoC / analyze 需求單一聲明，由 `ExpAdapterProtocol.capabilities` / `BaseAdapter.capabilities` ClassVar 表達。**覆寫時必須帶 `ClassVar[...]` 註解**，否則 pyright 視為 instance attr、Protocol 結構比對失敗。`analysis` 判別式 `NONE | FIT | INTERACTIVE`（取代舊 `supports_analysis: bool`，Phase 145）：NONE=無分析（power_dep 2D 掃描）、FIT=worker 擬合、**INTERACTIVE=人在圖上拖線選點、結果延遲**（onetone/twotone flux_dep）。
  - **INTERACTIVE 契約**：adapter 多 `setup_interactive_analysis(req, host: InteractiveHost) -> InteractiveSession`（兩 Protocol 在 `adapter/types`，Qt-free）。`InteractiveHost`(figure/redraw/run_background) gui 實作（`ui/interactive_analysis.InteractiveAnalysisWidget`=canvas+從 `actions()` 渲染 generic 鈕+Done；`run_background` **委派注入的窄 `InteractiveHostEnv` port=Controller**，背後走 `BackgroundRunner` pool，widget 不持自己的 pool —— passive host 收窄注入，ADR-0019），`InteractiveSession`(on_press/move/release/actions/invoke_action/info_text/finish) adapter 寫（flux-pick 在 `shared/interactive_flux_pick`，包 `TwoLinePicker`：toolkit-agnostic，host-driven mutation（拖線/swap）由 host repaint，但 **live mirror-loss 由 picker 自有 matplotlib `canvas.new_timer` throttle 重算 + 自 `draw_idle`**（取代原 `InteractiveLines` 的 `FuncAnimation`，host 無 hook）；**只 off-main auto-align 走 `host.run_background`**）。
  - **生命週期**：`Controller.analyze` 分派 FIT→`AnalyzeService.start_analyze`(worker)／INTERACTIVE→`_start_interactive_analyze`（acquire lease + `RenderHost.mount_interactive_analysis` 主線程掛 canvas，**不起 worker**）；用戶 Done → `AnalyzeService.finish_interactive(tab_id, session)` → `session.finish()` → 走**和 FIT 同一條** `_on_analyze_finished`（writeback compute + `update_tab_analyze` + lease release）。agent 端 `gui_tab_analyze_start` degrade（pending）+ 泛型 `gui_op_poll(handle)` 等用戶（MCP 26）。互動全主線程、只 heavy step 經 `host.run_background`→`InteractiveHostEnv`(Controller)→`BackgroundRunner` pool off-main（ADR-0017 Case B / ADR-0019）
- `RunService.start_run()` 在 acquire `RUN` lease 前依 `capabilities.requires_soc` 呼叫 `require_soc_handles(req)` Fast Fail；違規不佔 lease、不啟 worker
- `TabViewSnapshot.capabilities` 為 UI 唯一來源；`MainWindow` 依 `requires_soc` 決定 Run 按鈕是否需要 SoC、依 `analysis is (not) AnalysisMode.NONE` 顯示 / 隱藏 Analyze tab
- `Controller.start_run()` 不再 hard-code `has_soc()` 檢查；SoC 需求屬於 capability domain
- Fake / Lookback / FakeFreq 三個 simulation adapter 宣告 `requires_soc=False`，可在無 SoC 連線時執行

### LiveModel 反應式架構

- `Value` 是快照、`LiveField` 是代理；Model 層具備信號與 Resolver，UI 僅作 Observer
- `ui/fields/registry.py` 提供 `register_widget` 裝飾器；每種 `LiveField` 對應獨立 Widget 模組
- **Partial Re-binding**：`ModuleRefWidget` 僅在切換選單時銷毀並重建子樹，其他部分保持焦點
- **Module/Waveform 引用三態（`LibraryBindingState`）**：`LINKED`（chosen_key 為庫名、value=庫快照）/ `MODIFIED`（庫名 + 使用者改過 value，由 `_on_sub_change` 在 LINKED 時自動翻）/ `CUSTOM`（`<Custom:label>` 自建）。對應 notebook 的 `ml.get_module(name)` vs `ml.get_module(name, override)` vs inline dict。**override 已支援**：lowering 永遠輸出 `value` 子樹（不回讀庫），改 value 任一欄位即 override。`ModuleRefValue/WaveformRefValue.is_overridden` 把 MODIFIED 態**持久化**（`get_value()` 填 `is_modified()`、`__init__`/`set_value` 從 `is_overridden` 還原 MODIFIED、persistence 編解碼帶此欄位）—— 否則 reload 後 override 標籤丟失（庫名被當純 LINKED）。`<Custom:>` ref 的 `is_overridden` 無意義、永遠 False。
  - **懸空引用處理（按 binding state 分流，刻意不對稱）**：ref 的 chosen_key 指向已刪/改名的庫 entry 時（`ML_CHANGED`→`refresh_external`→`_rebuild_sub_field`，或 persisted load）：
    - **LINKED**（純庫引用、value=快照、無 override 可丟）→ **保留 chosen_key、標 missing+invalid、紅字 badge**（`_missing_library_ref`/`has_missing_library_ref()`/`cfg_form` 驗證訊息/`containers` 紅字 hint）。**可復原**：之後 register 同名 entry，`_refresh_library_binding` 下個 ML_CHANGED 偵測 key 回來 → 重建 valid LINKED（delete+re-add re-link）。
    - **MODIFIED**（user 改過 value）→ **降級 `<Custom:label>` 保留編輯**（已偏離庫、不復原，但 override 必須保住）。label 取自舊 sub_field spec label / value 的 type/style disc / `allowed[0]`。
    - **rename 不遷移引用名**：rename = `delete+register+emit ML_CHANGED 一次`，引用方靠此分流（LINKED→可復原 invalid、MODIFIED→Custom 保值）。md-side EvalValue 懸空**維持 invalid 不 fallback**（無上次值可保、等同輸錯名）。
- **集中 refresh 邊界**：`CfgFormWidget` 擁有 context/md/inspect/device EventBus subscriptions；`LiveField` 透過 `refresh_external()` 接收外部刷新
- **Eval scalar value**：scalar value 是 `DirectValue | EvalValue`；`ScalarSpec` 只宣告物理型別；`EvalValue` 可只帶 `expr`（不帶 `resolved`），由 lowering 拿 md 解析。`ScalarLiveField`（及 `SweepLiveField` 經其 edge 子欄位）bind 時 auto-resolve 填 snapshot，故 `= ?` ghost 只在 expr 真的無法解析時顯示。lowering 解析時依**擁有的 `ScalarSpec.type`** 做 `coerce_eval_result`（int spec → int、float spec → float），所以 `EvalValue("ro_ch")` 這種 int channel lower 成 int 而非 float；sweep edge 無 per-edge type、一律 float
- **Channel as scalar**：`ch` / `ro_ch` 使用 `ScalarSpec(type=int)`
- **Eval UI**：`ScalarWidget` eval mode 限可編輯 numeric scalar 且無 choices；右鍵 menu 切換 direct/eval；expression evaluator 只接受安全 numeric AST 與 MetaDict direct variable names
- **Invalid state**：validity 屬於 model state，向 parent bubbling
- **ScalarSpec.required**：標記欄位產生 `DirectValue(value=None)`，表單在填值前永遠 invalid（**未填 = `value is None`**，ADR-0010；無 `is_unset` flag——scalar 型別 int/float/bool/str 合法值永不為 None，故 None 一義表未填，但 entry 仍包在 `DirectValue` 內以保 direct/eval 模式身份）
- **ScalarSpec.optional**（`required` 的反面，兩者互斥、`__post_init__` 守）：欄位**可留空（None）且留空時 valid**；lowering 對 optional unset **省略該 key**（→ config-model 預設值，典型 None，如 `PulseCfg.mixer_freq`），validate_dynamic 同放行。widget：optional 數值/字串 direct-mode 用 `QLineEdit`（spinbox 無法表達「空」）、空字串=None、placeholder `(none)`、numeric 帶 validator；`optional + choices/bool` 不支援（widget fast-fail）。value 樹仍完整（entry 在、值 None）——不破 ADR-0010。
- **ScalarSpec.group**（純呈現提示）：同名非空 group 的欄位由 `SectionWidget` 收進**預設摺疊**的子 `_CollapsibleSection`（如 "Advanced"），排在無 group 欄位之後。**不改 value 樹形狀**（grouped 欄位仍是該 section 的扁平 leaf、頂層 lower）；lowering/validate 完全無視 group。用於把少用欄位（如 pulse `mixer_freq`）移出主表單降雜訊。

### State 邊界

- `State.cfg_schema` 為 committed SSOT；`LiveModel`（`SectionLiveField` tree）為 runtime draft SSOT。tab 模式 auto-commit：每次 `LiveModel.on_change` → `CfgFormWidget.schema_changed` → `Controller.update_tab_cfg` → `State.update_tab_cfg_schema`。dialog/writeback 模式為 local draft，只在 Apply 邊界寫回 `ModuleLibrary` / `WritebackItem`
- `Controller.start_run(tab_id)` 從 `State.cfg_schema` 讀；不接受外部傳入 schema。Form `is_valid()` / `first_invalid_reason()` 仍負責 pre-check
- `Controller.update_tab_cfg` 不發 `TAB_INTERACTION_CHANGED`（cfg 變更不影響 `TabInteractionState`），避免每次 keystroke 觸發全量 snapshot 重建；validity refresh 走 `CfgFormWidget.validity_changed`
- `ExpContext` 欄位：`md, ml, soc, soccfg, chip_name, qub_name, res_name, result_dir, database_path, active_label, predictor, readiness`
- `ContextReadiness.EMPTY/DRAFT/ACTIVE`：分離未建立、僅可編輯的 startup context、可執行/儲存的 file-backed context；`ExpContext.readiness` 為唯一 SSOT，`State` 不再 mirror 此資訊
- `State` 是 shared live context 的 SSOT；`TabState` 承接每個 tab 的 `cfg_schema`、run/analyze result、figure、analyze param instance、save path state、busy flags
- `State.devices: dict[str, DeviceState]` 是 device 狀態的 SSOT（含 `remember`）；`DeviceService` 只持 live driver/progress（execution 經 `BackgroundRunner`，見上「Device Lifecycle」）。device mutator 與 tab mutator 同樣只在主線寫、語義寫入 bump `device:<name>`
- `State.add_tab()` 只接受完整 `TabState`；`TabService` 負責建立 adapter 與 default cfg
- 只有 `run` 使用全域 `running_tab_id` lock；`analyze` 與 `save data` 是 per-tab busy state
- Run/Stop 按鈕只受全域 run lock 與本 tab run state 影響；analyze/save/writeback 只受本 tab busy 與資料可用性影響
- `plot reset` 只能跟真正的 `run start` 綁定
- Form state 必須回寫 State：tab 切換、event refresh、context 切換後 UI 能從 State 重新 hydrate；hydrate 路徑不得 emit state-change signal
- Save path state：每個 tab 只保存使用者 `save_path_overrides`；未覆寫時由 `TabService.get_tab_save_paths()` pure 計算 suggestion，實際 reservation 與 `mkdir` 只在 `SaveService` command boundary 執行
- Tab render state 由 `TabViewService.get_snapshot()` 一次組裝；snapshot query 不初始化 analyze state 或觸發 IO/event
- Analyze parameter instance 在 run success boundary 建立，`MainWindow` 只 render 已準備的 snapshot

### Persistence

- **Memento + Caretaker（ADR-0015）**：持久化集中成一個 **Memento**（`AppPersistedState{version, startup, session}`，`persistence_types.py` 的 **pydantic v2 frozen** model，單一 `APP_STATE_VERSION`）存**單檔 `gui_state_v1.json`**。`model_validate`/`model_dump` 取代手刻驗證；壞檔/舊版 → default + 報錯（檔層失敗），個別 tab 還原失敗 → `RestoreReport`（內容層失敗）。
- **PersistenceCaretaker = Driven Adapter**（`services/caretaker.py`，非 service、不與 `RemoteControlAdapter` 同級）：`run_app` 建立、`ctrl.attach_caretaker` 注入，Controller 經窄 `PersistOriginatorPort` 持有。只做單檔 disk I/O + load/flush 時機，不訂 event / 不碰 UI/State/cfg。`flush()` 觸發點刻意不寫死 lifecycle（未來可加 timer/button/event/RPC）。
- **關閉才寫**：無 runtime 即時寫盤（無 `DEVICE_CHANGED` 訂閱、無 splitter/apply/connect 即寫）；mid-session 不動磁碟，close 時 flush 一次。crash 丟失可接受。
- **Controller = 單一 Originator**：`capture_persisted_state` 從 `State.startup_prefs`（可變 dataclass，與 active `ExpContext` 分開）+ device 投影 + View `current_left_panel_width()` + 現有 tabs 組出 memento；`restore_persisted_state` 派發回各 service。薄 façade `persist_all`/`restore_all`。
- **各 service 提供 capture/restore**：Workspace `capture_session`/`apply_session`、Startup `capture_startup`/`restore_startup`；序列化內化於各自（tabs raw↔live codec 在 `session_codec.py`，是 Workspace 內部實作，Caretaker 只見不透明 `cfg_raw`）。
  - **codec 選 ref shape 靠 value 自帶 discriminator**：`session_codec._select_allowed_spec` 對 ref（編/解碼兩向）選 allowed shape——`<Custom:label>` 按 label、單一 allowed 直接取、**多型 LINKED（庫名 key）按 value/raw 的 `type`(module)/`style`(waveform) discriminator 配對**，無 match 即 Fast-Fail。**不** silent 取 `allowed[0]`（曾致 readout LINKED 到庫 pulse-readout 卻配 `allowed[0]` direct-readout → 崩在缺 `ro_ch`）。codec 是 pure（無 ml），故不用 `find_allowed_spec`（後者需 ml 走 library round-trip）。**向後相容**：restore 從完整 `make_default_value` 起、只覆蓋 raw 內有的 key，故舊檔缺的 optional 欄位（如 `mixer_freq`）自動留 `DirectValue(None)`，不需 bump `APP_STATE_VERSION`。
- **restore 不自動套 active context**：還原 startup 偏好（project/connection/device 預填）但不自動連線、不自動套 file-backed context（等 user 在 setup dialog 套用）。`SetupDialog` 不建立 `MetaDict` / `ModuleLibrary`；`StartupService` 回歸無狀態。

### Hardware Operation Gate

- **`OperationRunner` = 唯一 kind-agnostic 生命週期機制（ADR-0026 §1）**：抽出 run/analyze/device/SoC-connect 各自重抄的生命週期骨架（ensure_can_start → `create` 開 handle（註冊 cancel hook）→ register exclusion lease → mint per-op progress factory → `bg.submit` → 終局：discard progress → settle handle → release lease）。每個 op 不再自己編排，而是把一份 `OperationSpec`（exclusion/owner_id/wants_progress/cancel_hook/work/run_in_pool/on_terminal）交給 runner；不可化約的領域 policy（run 的 cancel-partial 判讀、analyze 的 set_tab_analyzing、device rollback、writeback compute）留在各 op 的 `on_terminal` callback。runner 只認 port（`ExclusionGate`/`ProgressHub`/`BackgroundExecutor`/`OperationHandles`），不認行為——可注 fake 單元測試。`work` 只收 runner 注入的 progress factory；figure_container / stop_event 是 op policy 的 closure 細節，不進 spec（runner 真正 figure/stop-agnostic）。
- `Controller` 經 `app_services.build_app_services` 建單一 `OperationGate`（Exclusion）+ 單一 `OperationHandles`（Handle）+ 單一 `OperationRunner`，注入需要的 service。`SoCConnectionService`/`RunService`/`DeviceService` 是 runner client（拿 exclusion + handle）；`AnalyzeService`/`PostAnalyzeService` 也是 runner client 但 exclusion=None（handle-only）。`active_operation_count`/`begin_shutdown` 走 `OperationHandles`（`live_count`/`cancel_all`/`poll`，含 analyze）；conflict 判定走 `OperationGate`
- 方案 A：`RUN` 排斥 SoC connect / 全部 device mutations；任意 device mutation 全域互斥
- terminal path（success / expected failure / unexpected failure / cancel）exactly-once release lease
- UI enable/disable 僅 mirror snapshot，不作為安全 guard
- MainWindow 關閉時經 `Controller.begin_shutdown` → Qt-free `ShutdownCoordinator`（`services/shutdown.py`）對**全部** in-flight operation（run+device+connect）cancel-all + 輪詢等停（timeout 強關）；QTimer 包在 driven adapter `QtShutdownDriver`（`adapters/`）。user close 先確認框，RPC 路徑不彈框（ADR-0003）

### Device Lifecycle 與 Snapshot（State 為 SSOT，見 `docs/adr/0007`）

- **`State` 是 device 狀態的 SSOT**：`State.devices: dict[str, DeviceState]`（frozen `DeviceState` =
  name/type_name/address/status/**remember**/info/error）。`remember` 是持久狀態（非瞬時 request 屬性）；
  `info` 是 `BaseDeviceInfo` value snapshot。每個語義寫入經 State mutator bump `device:<name>`；
  `refresh_device_info_cache`（讀取的快取刷新）**不 bump**。
- `DeviceService` 是 connect/reconnect/disconnect/setup worker：live driver（`GlobalDeviceManager`）+
  in-flight 暫態（`_active_lease`/`_active_name`/`_active_prior`）的 owner——**不再**持 device 狀態 dict，
  **也不再持 progress model**（progress 統一在 `ProgressService`，setup 經 `make_factory(token, owner_id=name)`）。
- `DeviceSnapshot` 是 **read-time projection**（`_project(DeviceState)`，保留原形狀），純由 State 欄位組成、
  不 splice progress（progress 經 `operation.progress(operation_id)` 另拉）；不被儲存。
- typed requests：`ConnectDeviceRequest`、`DisconnectDeviceRequest`、`SetupDeviceRequest`
- `DeviceStatus`（定義在 `state.py`，device 模塊 re-export）：`MEMORY_ONLY / CONNECTING / CONNECTED / DISCONNECTING / SETTING_UP`
- **無獨立 set_value**：設輸出值走 setup（`info.value` 由 driver 的 `setup` ramp，有 progress / 可 cancel）。`start_connect_device` / `start_disconnect_device` / `start_setup_device` 三者都回 operation token（`_active_token`），供 `OperationHandles.await_outcome` 等待——三條 device mutation 的 operation-handle 機制對稱
- driver construction 由 service-owned worker 執行；registration 前/後任何 failure 若 driver 已 constructed 但未成為 live entry，必須 `close()`
- **driver 實現者契約**（structural Protocol，pyright 查不到的語義）寫在 `DeviceProtocol` docstring：`setup` 在 worker thread 跑且**必須 poll `stop_event`**（否則 cancel 變 no-op）、`get_info` 不可 mutate、`close` 應冪等
- disconnect worker owns `close()` 與 registry removal；成功後依 `remember` 經 `set_device_status(MEMORY_ONLY)` 或 `remove_device`
- 跨物件不變式：device 在 `GlobalDeviceManager` ⟺ State status 是 live 狀態（CONNECTED/SETTING_*）
- render queries `get_device_snapshot()` / `list_device_snapshots()` 永遠不對 driver 執行 IO
- explicit live read（`get_device_info`、`get_device_value`、`get_device_value_for_new_context`）在同 device mutation active 時 raise `OperationConflictError`；`get_device_info` 的 driver 回值經 `refresh_device_info_cache` 寫回 State —— **值不變則靜默（不 bump）**，**值真的變了（pydantic `!=`）則 bump `device:<name>` 並 emit `DEVICE_CHANGED`**（外部硬體變化是真實狀態變化，理應推進版本 + 通知 requery）
- pyvisa resource 由 `DeviceService` 在 driver factory 內部建立；外部不直接建立 driver
- **device persistence 是 close-time 投影**：persist 不再訂 `DEVICE_CHANGED` 即時落地；`capture_startup`
  於 capture（關閉）時從 `State.devices` 讀 `remember=True` 集投影進 memento（見上「Persistence」/ADR-0015）。
  Controller 的 device handler 為 UI-only

### Connection Lifecycle

- **`ConnectionService` 已拆成兩個獨立職責（ADR-0026 §5）**：`SoCConnectionService`（SoC 連線 op）+ `PredictorService`（predictor load/predict 純計算）。兩者都住共用層 `gui/session/services/`。
- `SoCConnectionService.start_connect(req)`：單一進入點，是 `OperationRunner` 的 client（exclusion=SOC_CONNECT、cancel_hook=None——connect 無 cancellation point）。mock 與 remote **都**經 `bg.submit` off-main（舊 `_ConnectWorker(QThread)` 已刪），結果在下個 event-loop tick 經 signal 回報。
- 對 View 永遠以 `connection_finished` / `connection_failed` 回報；View 經 `Controller.bind_connection_outcome()` 綁定 signal，不取得 concrete service
- `PredictorService`（純 class、非 QObject）：`LoadPredictorRequest` / `PredictFreqRequest` 同步 API + 批次曲線（`predict_freq_curve` / `predict_matrix_element_curve`）；擁有 `exp_context.predictor` 寫 seam（set_context + `PredictorChangedPayload`）。IO 失敗轉 `PredictorLoadError`，未載入轉 `PredictorNotLoaded`。**不**進 runner / channel（非硬體 op）。

### Context Readiness 語義

| Operation | `EMPTY` | `DRAFT` | `ACTIVE` |
| --- | --- | --- | --- |
| Edit cfg / ML / MD | reject | allow | allow |
| Predictor / offline analysis | reject | allow | allow |
| Real hardware run | reject | reject | allow |
| Writeback to current context | reject | allow | allow |
| Save data / image / both | reject | reject | allow |
| Save path suggestion | None | None | suggestion |

- `set_startup_context()` 設 `readiness=DRAFT` 並清空 `active_label`
- `use_context()` / `new_context()` 設 `readiness=ACTIVE` 並注入 label
- `Controller.has_context()` 保留「可編輯 context」查詢；`has_active_context()` 是 real run 與 save 的強制 guard

### Error Handling

- `install_global_exception_hook()` 在 `app.py` 啟動時註冊，捕捉 main thread (`sys.excepthook`) 與 worker thread (`threading.excepthook`) 的未捕獲例外
- 同步 expected failure（`PredictorLoadError`、`PredictorNotLoaded`、`MlEntryValidationError`、`MdValueError`、`SessionPersistenceError`、`StartupPersistenceError`、`OperationConflictError`、`DeviceRegistrationError`、`SoCConnectionError`）由對應 View 捕捉並轉 dialog
- async expected device/connection failure 透過 terminal outcome 由 application coordination 顯示
- 不允許 broad `except Exception`；programming error 由 Global Hook 統一彈出 QMessageBox

### EventBus 契約

- `BaseEventBus`（共用 `gui/event_bus`，**payload-type-key**，三 app 統一，#6 退役自製 enum bus）；measure 的 experiment 事件依 domain 分兩模塊（ADR-0021）：`events/tab.py`（`TabEvent`+4 payloads）、`events/run.py`（`RunEvent`+2 payloads）；每 Payload 帶 `EVENT: ClassVar[<DomainEvent>]`，wire name=enum value；**session-core 事件**（md/ml/context/soc/predictor/device）住共用層 `gui/session/events.py`：`SessionEvent` enum + `SessionPayload(BasePayload)` base，wire 字串與舊相同
- `subscribe(PayloadType, cb)` 自動推 payload 型別（無 @overload）；`emit(payload)` 由 `type(payload)` 分派（payload type 即 key、配不錯，無 runtime 驗證）；**subscriber 例外吞+log、不 re-raise**（一個壞 subscriber 不破壞 publisher；取代舊「emit re-raise」——device/connection 的 lease 在真 terminal 釋放、不靠 subscriber 失敗回滾）
- subscribers 在 teardown 必須 unsubscribe；`CfgFormWidget` 與 dialog 在 modal `finished` 呼叫 `clear()`

### Sweep Editing

- `SweepCfg` 固定四欄（`start/stop/expts/step`）同步編輯
- `SweepEditor` 擁有 single-axis canonical arithmetic；`SweepLiveField` 接受 intent 並套用 invariant，`SweepWidget` 僅 dispatch/render
- resolved numeric bounds 下以 `expts` 為 canonical intent 並 exact derive `step`；unresolved `EvalValue` 不猜測 numeric update
- `SweepValue` 對 `expts < 1` Fast Fail
- **`SweepValue` 構造即自洽（Phase 130，`auto_norm: InitVar[bool]=True`）**：純 numeric 邊界時 `__post_init__` 依 start/stop/expts derive `step`（step 是 expts 的視圖、非獨立輸入）——任何直接構造（~16 adapter default / session_codec / inheritance）都一致，不再「committed step 髒、只 live 層規範化」（曾致 list_paths vs cfg_summary step 不一致）。`SweepEditor`（規範化權威，含 step→expts 反算）構造時傳 `auto_norm=False` opt-out，避免雙重規範化 / 覆寫用戶輸入的 step；EvalValue 邊界 auto_norm 跳過、交 SweepEditor 的 resolved 處理。
- `CfgSchema.to_raw_dict` 對 `SweepSpec` 走 expts-based lowering（`make_sweep(start, stop, expts=...)`）
- `step` 是同步輔助欄位，不承擔 mode sentinel
- `start/stop` 可承載 `EvalValue`（可只帶 `expr`）；lowering 以 `resolved` snapshot 或 md 求值為準，無 snapshot 又無 md 才 fail-fast

### Optional ModuleRef / WaveformRef

- `ModuleRefSpec(optional=True)` / `WaveformRefSpec(optional=True)`：combo box 多一個「None」（sentinel `_NONE_KEY = "<None>"`）
- `is_enabled` 瞬態（widget↔field 互動 flag）；初始值由 parent `SectionLiveField.__init__` 由 `initial_val is None` 控制
- **停用態 = value 樹裡的 `None`（ADR-0010，取代舊 `DisabledRefValue`）**。value 樹**永遠完整**（每個 spec 欄位都有 entry，無缺 key）；停用 optional ref 的 entry 是裸 `None`，由 value 自述、不靠旁路 flag、不反推 spec。`ModuleRefLiveField.get_value()` 停用回 `None`、`set_value(None)` 設停用（與 `__init__` 對稱）；`SectionLiveField.get_value()` 無條件收集子層自述（不再 reach-in 子層 `is_enabled`、不省略 key）。與「選 None Reset」（實驗層真 reset 選擇、lower 成 reset/none cfg）語義**不同**，勿混。
- **「停用→消失」只在 lowering**：`_section_to_dict_inner` 遇 `node_val is None` + `optional` → skip（run/save 出口才省略；persist 忠實序列化停用 ref 為 `{"__kind":"disabled"}`，還原回 `None`）

### Plotting Substrate（**共用層** `zcu_tools.gui.plotting`，Phase 133 步驟 B+ 抽出）

plot substrate 已抽到頂層共用套件 `lib/zcu_tools/gui/plotting/`（measure + fluxdep 共用），不再各自有 `mpl_backend`/`plot_host`/`plot_routing`：

| 模塊 | 責任 |
| --- | --- |
| `plotting/backend.py`（client） | 攔截 `plt.figure()` / `plt.subplots()`，轉發給 host |
| `plotting/routing.py` | task-local `ContextVar` 決定 active `FigureContainer` |
| `plotting/host.py` | 單一主線程 bridge QObject（**訊號在 host，不在 Container**）+ figure registry + canvas lifecycle |
| `plotting/container.py` | `FigureContainer`（純 QStackedWidget wrapper，無訊號） |
| `plotting/setup.py` | `BACKEND_NAME` + `configure_matplotlib_backend()`（import-clean） |

- GUI backend 是 process 專用設定（`module://zcu_tools.gui.plotting.backend`），不與 Jupyter notebook backend 混用；入口腳本（run_gui/run_fluxdep_gui）呼 `configure_matplotlib_backend()` 於任何 pyplot import 前
- worker 在入口設定 routing scope + 註冊 `QtLivePlotBackend`（`adapters/qt_liveplot_backend.py`）；依賴方向 gui → liveplot，liveplot 對 GUI 零認知
- **統一渲染路徑**：run liveplot、裸 `plt.subplots()`、analysis figure 全走 `plt.* → GuiFigureManager → attach 進 FigureContainer`。`draw_idle` 由 `GuiFigureCanvas` 覆寫：主線程直接畫、worker fire-and-forget marshalling；用 `host.is_main_thread()` 判斷
- `plt.show()` 只 activate 既有 canvas（**未 attach 則 raise**，Fast-Fail 統一）；不接管 QApplication event loop
- host QObject 必須先在 GUI thread 初始化（`FigureContainer.__init__` 呼 `host.ensure_host()`）；worker 先初始化會落到錯誤 thread affinity
- **container 解析鐵則**：refresh/activate/close 從 registry（id(fig)→container）解析,只有 attach 用 routing ContextVar（防並發 worker 串圖）
- 行為保證寫在各 module docstring（`plotting/{backend,host,routing,container,setup}` 開頭）；本表是索引

### Registry Boundaries

| Registry | Owner Service | View API |
| --- | --- | --- |
| `GlobalDeviceManager`（經 `DeviceRegistryPort`，ADR-0026 §6） | `DeviceService` | `list_device_names()`、`list_devices()`、`get_device_snapshot()`、`get_device_unit()`、`get_device_value_for_new_context()` |
| `ModuleLibrary` (live) | `ContextService` | `get_current_ml()`、`set_ml_module_from_schema`、`set_ml_waveform_from_schema`、`apply_writes`、`del_ml_module`、`del_ml_waveform`（ADR-0006 唯一寫入權威） |
| `MetaDict` (live) | `ContextService` | `get_current_md()`、`set_md_attr`、`del_md_attr`、`coerce_md_value` |
| `ExperimentManager` (file IO) | `ContextService` via `IOManager` | `use_context`、`new_context`、`setup_project`、`get_context_labels`、`get_active_context_label` |
| `ArbWaveformDatabase` (qubit-scoped `.npz` assets) | `ArbWaveformService` | `list_arb_waveforms()`、`get_arb_waveform_preview()`、`set_arb_waveform()`、`delete_arb_waveform()`、`rename_arb_waveform()` |

- LiveFields 透過 `ControllerProtocol` 取得 reactive environment，不直接 import singleton
- `DeviceService` 不直接用 `GlobalDeviceManager` singleton，改經 `DeviceRegistryPort`（5 個 instance method 鏡像 classmethod 面）；production 預設 `GlobalDeviceRegistryAdapter`，測試注 in-memory fake（ADR-0026 §6）
- `style:"arb"` waveform entry 仍由 Inspect 的 normal ML create/modify flow 建立；其 `data` scalar 透過 `choices_source="arb_waveforms"` 顯示目前 qubit-scoped asset keys，且允許空字串作為 Inspect 建立新 Arb waveform 的初始值；底層 `ArbWaveform` runtime 仍在空字串或 missing asset 時 fail-fast。delete/rename asset 不掃描 ML references，使用端碰到 missing asset 時 fail-fast。

### MetaDict Text Coercion

- `ContextService.coerce_md_value(key, text)` 取代 `ast.literal_eval`
- 已存在 key：強制依現有型別轉換；新 key：只接受 scalar（int/float/bool/str），複合 literal 拒絕
- 失敗轉 `MdValueError`

### Plot Panel Lifecycle

`ExpTabWidget._plot_stack`（`QStackedWidget`）管理兩種 canvas：

- **Liveplot canvas**：由 `adapters/qt_liveplot_backend.py`（GUI 註冊進 liveplot 的 backend）經 `plt.subplots` → mpl_backend attach 嵌入（`auto_close=False`）
- **Analysis canvas**：由 `show_analysis_figure()` 透過 `plot_host.attach_existing_figure_to_container()` 掛入

生命週期：

1. Run 開始 → `reset_plot()` 切回 placeholder；`RunService.start_run` 開頭 `State.clear_tab_results`（清 run/analyze/figure/writeback，Phase 129）→ run **進行中/失敗**時 tab 無 result。**cancel 例外**：worker 在 stop_event 前已產出的 partial result 由 `_on_run_cancelled` 保留（`has_run_result=true`、可 analyze）；完全沒產出才無
2. Run / Analyze 中 → worker 入口建立 routing scope，liveplot helper 與 GUI custom backend 從當前 routing context 取得宿主
3. Run 結束 → `update_tab_result()` 寫新 result；liveplot canvas 留在 stack
4. Analyze 後 → 若 figure 已由 GUI custom backend 建好 canvas，`show_analysis_figure()` 直接重用

### Canonical Result Load

- 使用者先開對應 adapter tab，再在 Analysis tab 用 `Load Data...` 載入 canonical result；file browser 以目前 context 的 `database_path` 最近存在目錄作為起點；remote/MCP 對應 `tab.load_data` / `gui_tab_load_data`。第一版不 auto-detect adapter。
- load 只要求 context not empty，不要求 active file-backed context 或 SoC；成功後把 loaded object 安裝為 `run_result`、記錄 `result_source_path`、清掉舊 analyze/post/writeback/figure，並對 analysis adapter 重新初始化 analyze params，因此可直接 Analyze。
- load 不把 `result.cfg_snapshot` 反填 Config tab，也不 bump `tab:<id>:cfg` / save path / SoC/device/context。未來 backfill seam 是 stateless `cfg_snapshot_backfill.py`，目前 load path 不呼叫。

### 固定尺寸出圖（`gui/figure_export.py`）

save_image 與 figure-screenshot 出圖**與視窗無關**：兩者都用嵌進 tab canvas 的 live figure，而該 figure 的 `size_inches` 隨 Qt canvas（即視窗）變。`figure_export` 統一收口 `set_size_inches(固定)→savefig→restore（try/finally，還原 on-screen 尺寸）`，但**兩條路徑用不同的固定尺寸**：`save_figure_to_path(fig,path)`（存檔，全品質）用 `SAVE_FIGSIZE=(8,5)`/`SAVE_DPI=150`；`render_figure_png(fig)->bytes`（agent 截圖，省 token）用較小的 `SCREENSHOT_FIGSIZE=(6.4,4.8)`/`SCREENSHOT_DPI=100`（~640×480）。screenshot（`MainWindow.take_figure_screenshot`，被 `tab.get_current_figure` 唯一看圖路徑使用）走 `render_figure_png`（**不 `canvas.grab()`**，後者抓 widget 像素=隨視窗）。皆主線程操作（live figure 為 GUI-owned）。

### Run Cancellation

- **scope wiring 由 run policy 的 work thunk closure 負責（ADR-0026 §2，取代舊 `OffMainScopes`）**：run policy 建的 thunk 套 `figure_ambient(live_container)`（app 層 routing + liveplot）→ `progress_ambient(pbar_factory)`（session 層 pbar）→ `ActiveTask(stop_event)`（op 專屬：把取消訊號傳給 `experiment/v2.runner.run_task()`）→ `adapter.run(request, schema)`（experiment adapter 簽名不動）。`ActiveTask`（runner library 語義）只留在 run policy 一處，不再外洩進泛用執行器。bg 只回 done/failed，**cancel 判讀（finished vs cancelled+partial）在 `RunService` 的 `on_terminal`**（持 stop_event 者自判，ADR-0019/0026）；cancel 觸發走 channel 的 cancel hook = `stop_event.set`
- 真實 experiment 內呼叫 `ModularProgramV2.acquire()` / `acquire_decimated()` 時必須把 `ctx.is_stop` 傳入 `stop_checkers`

### Progress 子系統（Phase 111，GUI v7）

四個角色，Qt 被擠到單一 driven adapter：
- **`ProgressBar`（worker 側，pbar_host.py，Qt-free）**：純轉發 raw（label/total/n）+ 30fps 節流，emit `ProgressEvent(operation_id, handle_id, kind=create/update/close)`。worker 經 `_pbar_factory` ContextVar 拿 `BoundProgressFactory`（綁固定 operation_id），故 worker 不選 container、不見 transport/Qt。
- **`ProgressTransport` port（services/ports.py）+ `QtProgressTransport`（adapters/）**：port 只抽象 worker→主線程 marshal；契約 = emit 線程安全 + receiver 在消費線程被呼叫。Qt 實作用 queued-connection 兌現。測試用同步 `DirectProgressTransport`。
- **`ProgressService`（services/progress.py，Qt-free、無鎖、只主線程）**：唯一 progress owner，`dict[operation_id, ProgressContainer]` + `dict[owner_id, live operation_id]`。`make_factory(op, owner)` 建空 container + 記 owner；`_on_event` 主線程 deliver。**不持 OperationGate**（決策2：兩者只共享 operation_id 整數，零 import/呼叫/生死同步）。兩個讀法：View 走 `bars_for_owner(owner_id)`（owner→live op→container，attach 跟隨輪替）；agent 走 `bars_for_operation(operation_id)`（直查 container，Phase 129 經 `operation.progress` RPC 折進 poll，run+device 通用）。
- **`ProgressContainer`**：一 operation 一個，`dict[handle_id, ProgressBarModel]`（一 worker 多 pbar）。生死綁 operation：`make_factory` 建、`discard_operation`（run `_release_lease` / device `_finish_operation`）銷——leave=True 的 bar 不發 close，靠 discard 清。
- **View attach**：tab widget / device dialog 用**自己的 owner_id**（tab_id / device_name）`ctrl.attach_progress(owner, listener)` 一次，listener 經 `ctrl.progress_bars(owner)` 重查渲染 `ProgressStack`。owner→live op 映射讓 attach 一次即跟隨 operation 輪替；關 dialog 只 detach、container 不滅。
- ContextVar 仍是 `_pbar_factory`（`use_pbar_factory()` set/reset）；每 QThread worker 繼承父 context snapshot 避免 race。`ProgressStack` 移除的 bar 維持 hidden child 重用（visible widget 不可 `setParent(None)`）。

### QThread Outcome Lifecycle

- background worker 的 success/failure/cancel outcome signal 僅在 Qt 原生 `QThread.finished` 已觸發後發布
- `deleteLater()` 僅連到原生 lifecycle signal，不連到 operation-specific signal
- Service 持有 worker 引用直到 terminal state，避免 Python GC 中途回收

## Adapter 預設值規則（對齊 single_qubit notebook）

### Default factory 無長參數列

per-role `make_<role>_default`（`defaults/` 每角色一檔）只收 `ctx`，內建一組 hardcoded 預設；adapter 端用 OO `.with_field(path, value)` 逐欄位覆寫（取代舊的 `gain=`/`ro_length=`/`trig_*=` 長參數列）。`make_trig_offset(ctx, *, trig_expr, trig_fallback)`：md 有 timeFly → `EvalValue(trig_expr)`（resolved 交給 lowering）、否則 `DirectValue(trig_fallback)`。

readout / reset 等多形狀 role 另有形狀明確 factory（`make_pulse_readout_default` / `make_direct_readout_default` / `make_pulse_reset_default` …）；`make_readout_default` / `make_reset_default` 為 thin alias → pulse 形狀。挑形狀的 adapter 呼叫明確版。

### CfgBuilder（value 樹組裝入口，`shared/cfg_builder.py`，ADR-0012）

`CfgBuilder(ctx, spec)` 是 adapter `make_default_value` 的高層組裝工具（領域層）：flat-path fluent、起手 = L1 blank 骨架（完整性 by construction），逐項覆寫。三層：L1 `make_default_value(spec)`（框架層 blank）/ L2 `make_<role>_default`（領域層每角色預設）/ L3 CfgBuilder（組裝）。動詞：`.scalars(**kw)`（頂層 scalar，純顯式無內建表）、`.role(path, role_id, *, optional=, prefer_blank=)`（經 `ROLE_FACTORIES` 表調 L2 掛 ref；預設走 ref 查庫、`prefer_blank` 強制 inline blank、`optional` 走 ref optional 查無回 None）、`.set(path, value)`（path scalar 覆寫，複用 `with_field`）、`.sweep(path, start, stop, expts)`（**只收字面 float**，EvalValue 邊界 raise）、`.set_sweep(path, SweepValue)`（逃生口，邊界可含 EvalValue）、`.build()`（回 value 樹，一次性）。**零鎖定宣告但自動填鎖定值**：鎖的宣告歸 `cfg_spec().lock_literal`，但 `build()` 遍歷 spec 把每個 `LiteralSpec` leaf 對齊 `spec.value`（穿透已掛 ref 的 chosen shape，復用 `find_allowed_spec`）——`.role` 掛的 L2 value 不懂 ref shape 內的鎖，build 修正；`.set` 碰 locked path 直接 raise（adapter 不該手設，消掉與 spec 鎖值的重複，C-raise）。value in-place、**不 validate**（留 make_default_cfg 邊界）；全動詞 spec-aware Fast-Fail。掛整節點靠 Builder 私有 `_mount_node`（框架層 `with_field` 保持 scalar-only）。映射舊手拼：`make_<role>_default`→`.role(prefer_blank=True)`、`make_<role>_ref_default(ctx)`→`.role()`、`(optional=True)`→`.role(optional=True)`、無條件禁用 optional ref→不寫（L1 blank 已給 None）。`ROLE_FACTORIES`（`defaults/role_factories.py`）是 `{role_id: (blank, ref)}` 單一 source，registry 的 RoleCatalog 與 Builder 共用（Builder **不**走 RoleCatalog——adapter 夠不著 + catalog factory 永不回 None）。**全 20 個 adapter 的 `make_default_value` 已統一走 CfgBuilder**（含 fake stub；舊手拼產同樣 `CfgSectionValue`、仍合法，新 adapter 照 builder 寫）。

### build_exp_spec（spec 樹組裝入口，`shared/spec_helpers.py`）

`CfgBuilder` 的 spec-side 對應：`build_exp_spec(*, modules, sweep=None, sweep_label="Sweep", dev=None, extra=None, relax_delay=True, reps=..., rounds=...)` 是 adapter `cfg_spec()` 的高層組裝工具——adapter 不再手寫 `CfgSectionSpec(fields={...})` 或手排頂層欄位。**固定吐出 canonical 順序** `modules, [dev], relax_delay, sweep, [extra], reps, rounds`，並自動補標準 `relax_delay`(FloatSpec decimals=3)/`reps`/`rounds`(IntSpec)。**頂層欄位順序與三個標準 scalar 的唯一所有者**（adapter 手排不一致的 bug 結構上不可能，Phase 142→143）。`modules`/`sweep`/`dev` 收 `dict[str, CfgNodeSpec]`，內部用 `declare_modules_spec`/`declare_sweep_spec`/`declare_dev_spec` 包成「Modules」/「Sweep」/「Device」section。變化點：`sweep=None`(無 sweep,lookback)、`sweep_label`(覆寫標頭,auto「Search bounds (min–max)」)、`dev`(flux_dep)、`extra`(run-only knob earlystop_snr/num_points,夾 sweep 與 reps 間)、`relax_delay=False`(無 relax 欄,fake/freq)、`reps=LiteralSpec(1)`(鎖定,lookback)。標量欄位用 **`IntSpec(label,...)`/`FloatSpec(label, decimals=,...)`**（`adapter/types.py` 的 `ScalarSpec` 顯式 sugar；**刻意不給 ScalarSpec 預設 type**——int/float 用量約五五波,靜默預設違最小驚訝）。全 18 個實驗 adapter 走 build_exp_spec（fake/stub 無實驗對應、仍手寫）。

### Adapter 撰寫 helper（`shared/ctx_helpers.py`）

對齊 notebook 高頻 idiom，降低 adapter 樣板：

- `proper_relax(ctx, factor=5.0, fallback=100.0)`：md 有 t1 → `EvalValue(f"{factor} * t1")`，否則 `DirectValue(fallback)`。對齊 notebook 的 `relax_delay = 5 * md.t1`。time-domain adapter 的 relax_delay 預設用它。
- `md_eval_scaled(ctx, key, factor, fallback)` → `Union[float, EvalValue]`：`factor * <key>` 的**掃描端點**（非整個 SweepValue），md 有 key → `EvalValue(f"{factor} * {key}")` 保留 md 聯動，否則 `factor * fallback` 純 float（sweep edge 無 per-edge type、不是 scalar field）。time-domain/rabi 的 `SweepValue.stop` 用它（t1 `5*t1`、t2echo/t2ramsey `4*t2e/t2r`、amp_rabi `2*pi_amp`、len_rabi `4*pi_len`）——避免 `md_get_float` 寫死數值丟掉表達式。
- `proper_res_freq_range(ctx, expts, span_factor=1.5)` / `proper_qub_freq_range(...)`：對稱 `center ± span_factor*width` 的 `SweepValue`（md 有 center+width → EvalValue edges 保留表達式，否則 scalar fallback ±30 MHz）。res 用 r_f/rf_w、qub 用 q_f/qf_w。`span_factor=1.0` 省略係數（`r_f - rf_w` 而非 `r_f - 1.0 * rf_w`）。收斂 spectroscopy adapter 的 `r_f/q_f ± span` EvalValue-or-fallback 大段。
- `proper_flux_range(ctx, expts)`：繞 calibrated flux 點外推 10% 的一週期掃描（md 有 flx_half+flx_int → EvalValue `1.1*flx_int-0.1*flx_half ~ 1.1*flx_half-0.1*flx_int`，否則 fallback `[-4e-3, 4e-3]`）。
- 上述 helper 產生的 `EvalValue` 一律**只帶 `expr`、不預先算 `resolved`**——解析由 lowering 負責（見「Eval scalar value」）。
- `md_scalar_float/int`、`md_get_float/int`、`md_has_key`：default 生成時「md 有可識別屬性就用、否則 fallback」的型別守衛讀取。

### Adapter 預設要點

| Adapter | 關鍵預設 | 理由 |
| --- | --- | --- |
| Lookback | gain=1.0, ro_length=1.4, trig_expr="timeFly - 0.1", trig_fallback=0.4 | 用途是校準 timeFly，trig 必須略早於觸發 |
| OneToneFreq | gain=0.05, ro_length=EvalValue("res_probe_len - 0.1") | 低功率探測保護共振器線型 |
| OneToneFluxDep | gain=0.005, reps=1000, rounds=1 | 低功率避免 saturation；高 reps 對齊 notebook |
| OneTonePowerDep | rounds=10 | notebook 慣例 |
| T1/T2Ramsey/T2Echo | relax_delay=EvalValue("5 * t1", fallback=100.0) | 5×T1 規則；t1 未知保守 100 us |
| T2Ramsey/T2Echo pi2 | preferred_names=["pi2_amp", "pi2_len", "pi_amp", "pi_len"] | π/2 脈衝優先 |
| LenRabi | relax_delay=10.5, sweep start=0.03 | 充分弛豫；避免零長度脈衝 |
| AmpRabi | relax_delay=10.5, sweep start=-0.3 | 含負 gain 探索 Rabi 奇偶性 |

### FluxDep `dev` section

`OneToneFluxDepAdapter` 的 `dev` 改為 `{label_key: DeviceRefSpec}`：GUI spec → lowering → `build_exp_cfg` 轉 `{device_name: {"label": label_key}}` → `make_cfg`。`make_cfg` 從 `GlobalDeviceManager.get_all_info()` 取完整 device info（含 `mode`）。

## Important Stubs

`main_window.py`：`show_plot`（Phase 11+ — Qt matplotlib liveplot backend）

## References

- 設計契約：`.agent_state/plans/gui/architecture.md`
- Phase 70 / 71 實作細節：`.agent_state/plans/gui/refractor_plan.md`
- 階段記錄：`.agent_state/plans/gui/task_plan.md`
- Phase 61 findings：`.agent_state/plans/gui/findings.md`
- Phase 69 findings：`.agent_state/plans/gui/review_gui.md`
