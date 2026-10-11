**Last updated:** 2026-10-11，獨立 predictor 查詢與模擬環境分界

# gui/session/ — 量測 session core

`value_lookup.name_from_type()`是`int`/`float`/`str`/`bool` scalar type name的公開唯一owner；
app保留帶領域context的錯誤文字與unsupported-target轉譯。

measure-gui 的「量測 session core」（context 系統 + SoC 連線 + 多 device + setup/device/inspect/predictor dialog）抽成共用層。對標 `gui/remote`、`gui/plotting`。每個 measurement-session app 注入自己的 app-local infra（gate + background）複用這層；session 模組**永不**反向 import `gui.app.*`。

`ContextService` 私下持有 `ResultEntry` 與 point/library handles。開啟 entry 發布空的記憶體 DRAFT，選取或建立 point 後發布 ACTIVE。它先完成資源與 asset capability 準備，再替換 session binding。準備失敗保留原 binding；point 已建立但安裝失敗時回報 `PointBindingError`，保留合法 point。

三個 target setter 經 resources owner 驗證值與來源。DRAFT 只替換記憶體，SavedPoint 使用單檔交易。目前 target 成功寫入後先安裝固定投影與版本，再發一個 MD／ML fact，保留焦點與 binding ID。同 entry 的其他 point 可獨立寫入，不改目前投影；舊 capture 保持隔離。

## Context、session environment 與 Run snapshot

- **Entry 與 point** 的持久資源由 [`resources.entry`](../../resources/entry/README.md) 管理，module library 由 [`resources.context`](../../resources/context/README.md) 管理。Session 不重建文件版面，也不在清除選取時刪除資源。
- **ParameterContext** 是隔離的 point/library/storage 投影，`EntryDraft` 與 `SavedPoint` 標識來源。`bind_aliases` 固定當次有序 rows。GUI callers 使用空 declarations，Notebook 保留自己的 role kind 契約。`$alias.path` 只沿這份表解析，bare concrete paths 使用獨立 namespace。未綁 alias 可留在 capture；實際讀取才指名失敗，不代用同名元件。讀取不 refresh、不讀 YAML，也不跟隨後續全域修改。Run／storage 的 names 投影包含表中所有已綁 rows，也包含自訂 alias。空列略過。歷史 fixed names 即使沒有 declarations，也必須指向來源 point 中存在的元件。
- **Alias 瀏覽投影** 使用 `ParameterContext.resolved_aliases()`，回傳有序 `ResolvedAliasRow(alias, value, resolved, error)`。它保留 authored softlinks，對固定 point 使用既有 resolver，不沿用 consumer kind 限制。只有單列 `RoleResolutionError` 變成 resolved=None 與 error；source/readability 失敗仍拒絕整個讀取，不從 live 表或檔案补值。
- **全域 alias 表** 由 `ContextService` 持有，只留在 session memory。`AliasTable` 原子驗證整批語法與重名，拒絕引用自身或後列，讀取才解析元件和 kind。App composition 宣告初值，第一份非空元件 snapshot 以 registry 的唯一候選填 primary row。此初值只初始化一次；明確提交表後不再覆蓋。`set_alias_rows` 安裝後才發布變更 fact。公開 `read_context(aliases=None)` 每次固定 point/library/storage 與全表，GUI 不要求 consumer declarations 或 kind 檢查。`aliases={"qubit": "Q2"}` 只覆寫該次 capture 的具名列，保留 dependent authored links，不改全域 rows、版本或 selection。EMPTY 沿 `no_context` 失敗，缺或未綁定的 `$alias` 在讀取時指名報錯。歷史 `read_target(target, roles=...)` 使用已固定的 names，不展開目前全表。GUI adapter 不宣告角色，Tab、Run 與 Load 直接使用 alias-aware capture。App 的 EMPTY 無 override 讀取回 None，保留 context-free 與 deferred 路徑；非空 overrides 仍拒絕 no_context。省略 overrides、None 與 {} 等價。Notebook 的 role resolver 與 via 保留原契約。
- **SessionEnv** 是已發布的 session environment。EMPTY 的 `parameters` 是 None，DRAFT／ACTIVE 持有無 consumer roles 的固定 `ParameterContext`。它不提供 md/ml 欄位；SoC 與 cached value sources 保持各自 owner。
- **Run snapshot** 是依該次 Run 契約捕捉的輸入，接受後不因後續編輯或 refresh 改用新的 cfg。持有 live `SessionEnv` 不等於持有這份不可變證據；SoC、driver 等 live handles 也不作為普通資料整包凍結。

## 結構

`persistence.py` 是 app-neutral、強型別的 single-file caretaker；app 注入 codec 並保留
schema/version/trigger policy，shared 層不依賴 Pydantic、Qt、cfg 或 app module。

```
session/
├── types.py            — SocProtocol/SocCfgProtocol/SocHandle/SocCfgHandle/ContextReadiness/SessionEnv（P-a 值型別）
├── aliases.py          — session-only 有序表；原子替換、lazy component links 與單次 override
├── events.py           — SessionEvent enum + SessionPayload base + 11 payload（Md/Ml/ContextSwitched/Soc/Device{Changed,SetupStarted,SetupFinished} + GateChanged/ConnectionFinished/DeviceOperationFinished）
├── state.py            — SessionState（session_env+devices/DeviceState+preferences/SessionPreferences+shared VersionTable+device mutators）；DeviceStatus
├── ports.py            — session service 依賴的 driven-adapter/seam ports：ExclusionGate(+OperationKind/OperationConflictError)、OwnerScheduler（owner-loop post/call/probe）、BackgroundExecutor（純 off-main 執行器，`submit(work,*,run_in_pool,on_done,on_error)` 無 scopes 參數）、ProgressHub、ProgressEvent/Kind/Transport、DriverFactoryPort、RememberedDevicePort、DeviceRegistryPort（production 使用 session-owned DeviceManager，測試注入 in-memory fake，ADR-0066）、ProjectIOPort、ContextReadPort、ProjectContextPort
├── hardware_gate.py    — RunBlocksHardwareGate：量測硬體互斥矩陣（RUN 擋 RUN/SoC/device mutation；device mutation 依 resource id 互斥；SoC connect 擋 RUN/SoC）；每個lease另保存captured origin、human note與monotonic start，register/release emit `GateChangedPayload`，`snapshot()`只投影active duration、不曝露raw epoch
├── operation_handles.py— OperationHandles（async Handle/Cancel facet，零 kind）+ per-op OperationChannel（單一有序事件 FIFO Settled/Stop，取代舊 FeedbackInbox + poll-loop，ADR-0066）+ captured EventOrigin operation record；`create(cancel_hook=, origin=)`要求caller顯式capture，`event_origin(token)`在live/retained-done record上投影string operation id；`has_cancel_hook`/channel.`can_cancel` 提供一般取消能力查詢；cancel hook例外只log，Stop事件仍入列且cancel_all繼續處理其它operation
├── operation_runner.py — OperationRunner（唯一 kind-agnostic operation 生命週期機制，ADR-0066：ensure_can_start→create→register→progress factory→submit→終局 settle）+ OperationSpec（各 op 把領域 policy 交給 runner）；run/analyze/device/SoC-connect 都是它的 client，runner 只認 port 不認行為，並隔離 terminal callback / cleanup 例外，確保 handle settle 與 exclusion release 仍 best-effort 執行；duplicate settle 是 logged no-op，不重複 cleanup 但會暴露 policy bug
├── scopes.py           — progress_ambient（session 層：pbar ContextVar，無 Qt；ADR-0066，取代舊 executor `_entered`/OffMainScopes 的 pbar 欄位）。Figure 使用 operation-owned Plots，不透過 ambient scope
├── notify_handles.py   — NotifyChannel/NotifyHandles（agent→user prompt 的跨線程 channel，事件集 Reply/Dismiss/Timeout，獨立於 operation 的 Settled/Stop；鏡像 OperationChannel 四不變式，ADR-0066）
├── device_control.py   — DeviceControlPort + DeviceControlFacet：device lifecycle/query/progress/event-subscription 的窄 control facet（DeviceService + ProgressService + BaseEventBus），供 shared DeviceDialog、PredictorDialog 與 main remote device handler 使用；cached numeric device-value query 只讀 State，不做 live hardware read；`poll_device_info()` 是可能更新 cache 的 live refresh，`try_poll_device_info() -> bool` 讓 guarded app / UI 明確取得 refresh 是否送出；不含 context `get_current_md`
├── predictor_control.py — Read-only explicit-parameter queries and concrete-qubit prefill for dialog and remote callers
├── progress_control.py — ProgressControlPort + ProgressControlFacet：owner-keyed progress subscribe/query 的窄 control facet（ProgressService），供 app-local run UI 使用；operation-id progress 查詢仍留在 app controller/remote operation handler
├── context_control.py  — ContextControlPort + ContextControlFacet：固定 parameter read、entry/point cached queries、全域 alias 表讀写、focus 與 value-source 的窄 facade，轉送 ContextService。Alias 表可在未開 entry 時編輯，提交不寫檔。
├── setup_control.py    — SetupControlPort + SetupControlFacet：entry open/new、point use/new、cached setup preferences 與獨立 SoC commands；直接公開既有 devices facet 供 cached label 建議，不依賴 app Controller
├── controller_mixin.py — SessionControllerMixin：measure 的 session forwards；shared dialog 使用窄 control facet，entry open/new 直接交 SetupControlPort，不經 project identity 或 app Controller
├── expression.py       — simpleeval 共用數值求值引擎，封裝函式／算子白名單、AST／數值大小限制及 MetaDict 名稱查詢。`evaluate_scalar_expr` 保留 int/float/complex，`evaluate_numeric_expr` 在 real-only 邊界拒絕 complex；`**` 為冪次，`^` 沿用原生 XOR。函式為 sin/cos/tan/sqrt/exp/log/log10/abs，常數 pi/e；math 函式不自動轉 cmath。引用到的保留名稱若與來源衝突則拒絕。EvalRef 是 apply-time resolve marker，不持久化。
├── value_lookup.py     — read-only value source lookup（`ValueLookup`/`ValueRegistry`/owner-scoped replace/unregister/`ValueRef` resolve-once helpers）；純 session leaf，provider 來源由 service binder 負責，使用端只見 lookup
├── pbar_host.py        — ProgressBar(worker)/ProgressBarModel(主線程 SSOT)，Qt-free；worker API 同時支援 incremental `update(delta)` 與 absolute `set_progress(n)`
├── adapters/
│   ├── manual_owner_scheduler.py — ManualOwnerScheduler（headless/test owner queue，由建立它的owner thread明確pump）
│   ├── thread_pool_background.py — ThreadPoolBackgroundExecutor（純threading worker，terminal delivery經OwnerScheduler回owner；quiesce等待worker與delivery ack）
│   ├── qt_owner_scheduler.py — QtOwnerScheduler（queued-signal owner-loop adapter）
│   ├── qt_background.py — BackgroundRunner（Qt runtime background adapter）
│   ├── qt_progress_transport.py — QtProgressTransport（worker→主線程 progress marshal，queued signal；app-agnostic）
│   └── qt_shutdown_driver.py — QtShutdownDriver（shared QTimer driving adapter，輪詢 ShutdownCoordinator，供 measure closeEvent 使用；QTimer tick/close callback 例外被 adapter 邊界 log 並 guarded close）
├── services/
│   ├── connection.py   — SoCConnectionService（SoC connect op；硬體 facet、OperationRunner client、cancel_hook=None 無 cancellation point；終局經typed `ConnectionFinishedPayload`，無Qt）
│   ├── predictor.py — Independent simulator queries; params_for is injected by app composition. params.json is read-only prefill.
│   ├── context.py      — ContextService：owner-thread entry/point binding、固定 read/role capture、focus、source facts 與目前 entry 的 ledger capability
│   ├── device.py       — DeviceService（connect/disconnect/setup off-main，**全 port 注入**：gate/bg/progress 必傳；connect/disconnect終局用`DeviceOperationFinishedPayload`，setup重用`DeviceSetupFinishedPayload`，無Qt）；`get_device_unit` 讀有效 YOKO／Fake cached Info 的 `value_unit`，分別為 A／V 與 native；缺 Info 的 UI／snapshot 標 none，不按 type 推測單位；strict binding 缺 Info 拋指名 device 的 `DeviceRegistrationError`，這些查詢不讀硬體、不改 connected／offline 狀態；`poll_device_info(name)` = best-effort off-main live-read（worker 純讀 driver、on_done 主線做 cache 比對+bump+DEVICE_CHANGED；memory-only/connect/disconnect skip，setup/ramp 允許 current-value refresh；單次讀失敗吞掉，不寫 State 於 worker）
│   ├── project_settings.py — ProjectSettingsService：以明確 result/Database roots 呼叫 ResultEntry.open/create，再交 ContextService.install_entry；記住的 entry/connection/device preferences 不證明來源已開啟
│   ├── progress.py     — ProgressService（per-operation pbar 容器 + bound factory，吃 ProgressTransport；**共用**，非 app-held；owner listener 例外逐一 log 並隔離，壞 listener 會 detach，不阻斷其他 listener 或 terminal cleanup）
│   ├── shutdown.py     — ShutdownCoordinator（Qt-free cancel-all + poll state machine，吃 OperationHandles，ADR-0066；QTimer driver 在 adapters/）
│   ├── io_manager.py   — IOManager（ContextManager 包裝，實作 ProjectIOPort；**共用**；把 project `result_dir` 映射到 `result_dir/exps` 作為 context root）
│   ├── simulated_environment.py — Use Simulate Env coordinator：經 DeviceService 斷開真實裝置，註冊／初始化 FakeDevice，再綁定 MockSoc 並由 SoC owner 發布；不安裝 predictor
│   ├── value_sources.py — ValueSourceBinder：訂閱 Context/Device event，把 live SessionState 投影註冊進 `ValueRegistry`；provider 只讀 cached state（device value 只看 `DeviceState.info`，不 poll hardware），並用 owner-level replace/unregister 維護 lifecycle
│   └── build.py        — SessionServices bundle 與 build_session_services：組裝 owning services/facets，從 project_root 注入 result/Database roots，明確注入 entry-scoped asset binder
└── ui/                 — 共用 dialog（setup 吃 SetupControlPort；inspect 吃 ContextControlPort；device 吃 DeviceControlPort + md_provider；predictor 吃 PredictorControlPort）
    ├── progress_bar.py   — LightweightProgressBar（native-free QWidget；`setValue` 只更新狀態並排程 repaint，避免高頻 progress 卡在 `QProgressBar.setValue`）
    ├── progress_stack.py — ProgressStack widget（唯一拉 Qt 的 progress 件；高頻 render 只對變更過的 maximum/format/value 寫入 LightweightProgressBar；可注入 profiling callback 拆量 derive / value writes）
    ├── setup_dialog.py — Result entry、Working points 與 Connection 的 shared dialog。New entry／Open 與 point Switch／Create 都交既有 owner。Working points 清單顯示 description，Create 明確提交 Label、Clone from 與 Description。Track device 只用 cached value＋unit 建議可修改的 Label，不讀硬體、不綁裝置。`(empty)` 使用 setup seed 與空 library。Reopen 使用 cached queries，不切 binding、不連硬體
    ├── device_dialog.py  — 設備管理；吃 `DeviceControlPort` + 獨立 `md_provider`（eval mode apply-time resolve 不污染 device facet）；per-device panel lazy-import zcu_tools.device.*；host 可切 `set_read_only(True)` 進 inspection mode：Add/Drop/Apply/Reconnect/Stop 與 panel edit controls 禁用，Refresh、selection、poll、Close 保持可用；dialog-scoped + selection-scoped QTimer（1s）對當前選取 device 呼 `try_poll_device_info`，結果經 DEVICE_CHANGED 流回並保留目前選取；timer 僅可見+有選取時跑，hide/close/無選取即停（不常駐、不全 device 輪詢）
    ├── predictor_dialog.py — Independent fluxonium or SQUID transmon trials, frequency curves (matrix curves only for models that provide them) and local Calibrate. Calibration displays actual MHz and clamped from the shared model result; it changes only dialog bias/cache, never a component. No Apply or session model. Device values and frequency expressions remain optional inputs.
    ├── predictor_canvas.py — PredictorCurveCanvas：f_ij 曲線 over device value（雙 x 軸 flux/value）+ 可拖動 flux 垂線（方案 B：marker 不折、出窗平移）；支援多條 transition 同時 highlight；host 可關閉 marker interaction 讓 live view 只由外部 value 驅動；主線程同步繪圖不需 marshal
    ├── inspect_base.py   — InspectDialogBase：md tab + ml view/rename/del；hook `_build_extra_toolbar_buttons` 讓子類在 Refresh 左側加 app-specific 入口，hook `_build_extra_ml_buttons` / `_on_ml_selection_changed` 讓子類加 app-local actions。**consumer**：measure subclass 自己擁有 embedded CfgEditor Modules presentation，不改 base。
    └── value_source_input.py — QLineEdit value-source helper：`@{prefix` 顯示分段 completion（`@{`→頂層、`dev`→`device.` 並展開下一段、`device.`→下一段），Tab/Backtab 接受預設候選時若當前候選全是 namespace 會自動補 `.` drill down，補到完整 key 才加 `}`；在 `@{full.key} ` 後輸入空格才立即 resolve 並以文字替換；token 完成或解析後明確 hide popup；依賴 `ValueSourceInputHost` port，不知道 ContextService/LiveModel
```

`Use Simulate Env` 是明確的環境組裝入口，不是 `SOC_CHANGED` 的副作用。它只在啟用時斷開真實 devices，允許之後混用。任一斷線失敗就停止並回報部分成果，不自動重連、歸零或 RF off。低層 SoC connect 不建立 FakeDevice；重複啟用保留有效 mock 的資源與數值。

Device panel 保留原 mode 顯示與 output／value／rampstep 控制，不提供獨立 `value_unit` 編輯。完整 cfg 的單位由 Info 驗證，snapshot 與 recipe 沿同一儲存事實。

## Analysis-only operations

Fluxdep search 重用 OperationRunner、OperationHandles 與 ProgressService。
Runner 的 gate／progress injection 可為 None。begin 在建立 handle 前拒絕缺少的必要 facet。
純分析不用注入硬體 gate。OperationSpec 的 optional on_opened hook 在 progress notification
與 submit 前交出 token，讓 domain owner 保留 admission 與 startup-failure activity。
Hook、registration、factory 或 submit 失敗時，runner 清理並重拋。
Measure client 保留自己的硬體 policy。

## Operation markers

`DeviceDialog` owns the shared device-list activity projection. Each row reads its
State-owned `DeviceStatus` through `DeviceControlPort` and receives the same compact
blue `●` treatment as the MainWindow Run tab while that device is `SETTING_UP`.
Setup start, every terminal outcome, device-change repaint, and dialog reopen derive
rows again from State; concurrent devices therefore clear independently and the
user's selection remains unchanged. No persisted marker or competing busy cache is
introduced.

## 關鍵設計

- **expected failure逐一opt in**：session service只讓caller可修正的concrete exceptions加入
  `ExpectedError` taxonomy；`OperationConflictError`、context validation、predictor readiness與
  value-lookup expected leaves攜帶typed category。`ValueLookupError` parent、`ProviderError`、
  SoC/device async terminal與registration failures保持unexpected，不因共同ancestry被降級
  （ADR-0068）。
- **app 注入 infra 經 port**：concrete `OperationGate`（衝突 policy，app-local）+ runtime-specific `BackgroundExecutor` 各自建；Qt composition 使用 `session/adapters/qt_background.py` 的 `BackgroundRunner`，headless composition 使用注入 `ManualOwnerScheduler` 的 `ThreadPoolBackgroundExecutor`。兩者都保留 shared-pool / dedicated-worker 語意，terminal callback 一律回 owner thread；具體 owner 可在 teardown 呼 `quiesce()` 等待 worker 與 terminal delivery。`tests/gui/session/test_headless_composition.py` 以subprocess阻擋Qt imports，穿過正式`build_session_services`與offline MockSoc success/cancel lifecycle，作為無`QApplication` composition的可執行邊界。`ProgressService`/`IOManager`/`QtProgressTransport`/`QtShutdownDriver` 是共用 session service/adapter。session service 只依賴 `ExclusionGate`/`BackgroundExecutor`（只宣告 `submit`，`quiesce` 不進 port）/`ProgressHub`/`ProjectIOPort` port。
- **ExclusionGate str-keyed**：session `OperationKind`（soc/device kinds）+ app 自己的 kind（measure `RUN`）用 wire 字串比較，故兩 enum 同一 gate。共享硬體互斥規則住在 `RunBlocksHardwareGate`，app-local gate 只保留 RUN kind 與 import boundary。
- **app 繼承/實作**：measure `State(SessionState)`；Controller 暴露 `setup_control: SetupControlPort` / `context_control: ContextControlPort` / `device_control: DeviceControlPort` / `predictor_control: PredictorControlPort` / `progress_control: ProgressControlPort` 供 shared dialog、main remote handler 與 app-local run UI 走窄 facets。
- **Inspect 元件樹**：`ComponentTreeWidget` 透過 context facet 讀取 point／setup 文件並提交元件與參數命令。`ContextService` 擁有驗證、記憶體 DRAFT 與資源交易，widget 不寫檔。Setup 範本編輯不改現有 point 或焦點。Measure Inspect 的 Parameters 使用元件樹；Modules 保留 app-owned cfg editor，dirty selection／close 仍須顯式 Apply、Discard 或 Cancel。Modules mutation caller 的新容器接線尚待整合。
- **Entry binding**：GUI Setup 和 remote entry.new/open 使用同一個 ProjectSettingsService。EntryInfo 提供 UUID、任意目錄名稱與 result／database 絕對路徑。Open 只開既有 entry，New 明確建立。成功發布空 DRAFT，不自選 point、不碰硬體。失敗保留舊 binding；New 已建資源而安裝失敗時保留 entry，以 Open 恢復，不重試 New
- **Entry 與 point**：ResultEntry 擁有 entry roots、setup 與 points。ProjectSettingsService 只負責明確 open/new 與 preferences，ContextService 安裝 binding。Point 操作作用在目前 opened entry，不從 chip／qubit／resonator 推導身分
- **operation callback 邊界**：`OperationRunner` terminal delivery 會 catch/log policy callback 例外並把 operation settle failed；`settle()` 的 progress discard / handle settle / exclusion release 各自 best-effort 執行，避免 progress UI 或 cleanup 例外讓 handle 永遠 pending。第二次呼叫同一 settle closure 會記 error 並 no-op，讓 double-terminal policy bug 可診斷但不破壞 cleanup idempotence。`ProgressService` owner listener 逐一隔離，壞 listener 不阻斷其他 view 或 terminal cleanup。
- **import-clean leaf**（不得拉 Qt/matplotlib/gui.app.*，`tests/gui/test_shared_layer.py` 守）：types/events/operation_handles/ports/state/pbar_host/setup_control/context_control/device_control/predictor_control/progress_control。`adapters/` + `ui/*` + `services/*` 是 Qt/重，不列。
- **wire name 來源**：`SessionEvent.X` 的字串值即 wire event name；measure-gui 的 wire-name lock 測試（`test_remote_event_dialog_view.py`）鎖全集，搬移/改名 payload 不得動字串值。
- **`SessionEnv.values`**：只攜帶 read-only `ValueLookup` facade，供 default generation / resolve-once 讀目前 session 投影；`ContextService.list_value_sources/read_value_source` 是 GUI/remote 共用查詢面，mutable `ValueRegistry` 只在 session composition root / source binder 內使用。`ValueSourceBinder` 以 owner-scoped replace/unregister 維護 `context.*` / `project.*` / `device.<name>.*`；named device source 包含 `device.<name>.name` 與 cached info（如 `value`/`status`），device provider 只讀 cached `DeviceState.info`、不 poll hardware，也不推導 active/flux 語義。`value_source_input.py` 把此查詢面包成可注入的 GUI token helper；它只替換輸入文字，不寫 cfg/md。`SessionEnv` 仍是 live environment facade，不是 snapshot。

跨模組設計見 ADR-0067（shared session、event ownership、frontend 邊界、md／ml 寫入權威）、ADR-0065（cfg 編輯與 lowering）、ADR-0002（version table 與 off-main handler）、ADR-0066（operation handle、執行、取消、關閉與跨線程互動 channel）。

## 獨立 predictor

`simulate.fluxonium.predict.FluxoniumParams` 與 `simulate.transmon.TransmonParams` 保存可未知的具名物理輸入，missing() 指出缺項。
齊全且有效才建立中性 query model。Service 不持有 installed model，不讀 dialog。
App 注入使用者的 params_for，把當前 capture 中的具體元件轉成參數。
Framework 不猜元件欄位，也不 import 使用者套件。查詢與 Calibrate 不寫元件或版本。
Dialog 的元件選單初始選 alias qubit；Load component 才經注入的 params_for 預填，選取或切換工作點不改 trial。欄位勾選代表已知，取消勾選保留 None；缺項會明示，完整且有效才計算。
模型選單有 Fluxonium 與 SQUID transmon，各自顯示自己的能量欄位，flux 欄位共用。Fluxonium 顯示頻率與 |n|、|phi| 曲線。Transmon 沒有矩陣元素能力，只顯示頻率曲線。Load component 依元件 kind 選模型；params.json 只預填 fluxonium。
Measure registry 啟用獨立 predictor.json user cache。Dialog 持有 versioned PredictorSnapshot 與 SingleFileCaretaker，保存 model 與 nullable params；每次編輯與關閉保存，重建 dialog 還原。Standalone callers 以 enable_parameter_cache(cache_dir=...) 明確啟用與隔離 cache，啟用時立即還原。Cache decode failure 回空 trial 並顯示錯誤，不讀寫 component DocumentStore。
