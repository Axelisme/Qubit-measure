**Last updated:** 2026-10-18，Inspect 完整表單草稿的 unsaved 判斷

# gui/session/ — 量測 session core

`value_lookup.name_from_type()`是`int`/`float`/`str`/`bool` scalar type name的公開唯一owner；
app保留帶領域context的錯誤文字與unsupported-target轉譯。

measure-gui 的「量測 session core」（context 系統 + SoC 連線 + 多 device + setup/device/inspect/predictor dialog）抽成共用層。對標 `gui/remote`、`gui/plotting`。每個 measurement-session app 注入自己的 app-local infra（gate + background）複用這層；session 模組**永不**反向 import `gui.app.*`。

`ContextService` 私下持有 `ResultEntry` 與 point/library handles。開啟或建立 entry 只發布 metadata 和 point list，不自選 point。此時 readiness=EMPTY、parameters=None，但 entry metadata、point list 與 setup source binding 仍可讀。未開 entry 時 metadata 與 source binding 才是 None，point list 為空。SoC、device 與獨立 predictor 保留各自能力。明確選取或建立 point 後才發布 ACTIVE。它先完成資源與 asset capability 準備，再替換 session binding。準備失敗保留原 binding；point 已建立但安裝失敗時回報 `PointBindingError`，保留合法 point。

三個 target setter 都經 resources owner 驗證。`set_parameter` 驗 value schema 與 metadata 的 YAML 保存表示；metadata 是 caller 的 opaque mapping，setter 不解讀 key，也不查來源。`set_module`／`set_waveform` 驗 library spec，並由 library owner 對該 entry 的 ledger 驗證來源 Provenance。SavedPoint 是唯一 target，使用單檔交易。目前 target 成功寫入後先安裝固定投影與版本，再發一個 MD／ML fact，保留焦點與 binding ID。同 entry 的其他 point 可獨立寫入，不改目前投影；舊 capture 保持隔離。

`ContextService.apply_writeback` 是底層 accept 入口，先驗 captured SavedPoint 的 entry、point label、存在性與文件，再綁定資源交共用 executor。Executor 先 append accepted event，再沿項目順序呼叫 setters。Parameter／library setter 要求目前已有選取的 point，sample-only 不要求 selection，直接追加 entry JSONL 並記 captured point label。直接呼叫底層入口時，未選 point 的 parameter／library 項目可在 append 後失敗；成功前綴與 accepted event 保留，首錯停止，不 rollback 或 retry。Application 的 `GuardService.check_writeback` 才在未選工作點時以 no_context 提前拒絕，提示先選或建立工作點。同 entry 現選 B 只允許開始命令，不把 captured SavedPoint(E, A) 改成 B；跨 entry 在 append 前拒絕。入口不寫記憶體、不自動建點。成功前綴依每項實際提交發布。樣本 preview／receipt 經 sample owner 讀 entry JSONL，不維護另一份樣本或延後 flush。

## Context、session environment 與 Run snapshot

- **Entry 與 point** 的持久資源由 [`resources.entry`](../../resources/entry/README.md) 管理，module library 由 [`resources.context`](../../resources/context/README.md) 管理。Session 不重建文件版面，也不在清除選取時刪除資源。
- **ParameterContext** 是隔離的 point/library/storage 投影，`SavedPoint` 標識來源。`bind_aliases` 固定當次有序 rows。GUI callers 使用空 declarations，Notebook 保留自己的 role kind 契約。裸 `$alias` 回傳解析後的名稱字串。`$alias.path` 只沿這份表解析，bare concrete paths 使用獨立 namespace。名稱不參與數值算術或插值。未綁 alias 可留在 capture；實際讀取才指名失敗，不代用同名元件。讀取不 refresh、不讀 YAML，也不跟隨後續全域修改。Run／storage 的 names 投影包含表中所有已綁 rows，也包含自訂 alias。空列略過。Device 名稱不必在 `point.components` 或裝置表中。歷史 fixed names 不重新展開 alias；元件欄位的有效性仍由其讀取 owner 檢查。
- **Module slot 錯誤** 由 `parameters.ModuleSlotError(AttributeError)` 區分 unset／dangling slot 與 source 故障。`ParameterContext.library_key` 與沿 slot 的 leaf lookup 保留 slot 路徑和原因。`has_library_key(path)` 對固定來源查詢 root 或 slot 是否存在，不 materialize；缺 root、unset／dangling slot 或缺／未綁 alias 回 False，語法、其他 alias 錯誤與來源故障仍 raise。Form factory 的 `ctx.has(ref)` 共用這個 owner 查詢。Expression 只轉譯這個明確型別為 `ExpressionError`。Tab 保留 Invalid cfg 並停用 Run，ordinary `AttributeError` 與其他 source 故障仍向外傳遞。
- **Alias 瀏覽投影** 使用 `ParameterContext.resolved_aliases()`，回傳有序 `ResolvedAliasRow(alias, value, resolved, error)`。它保留 authored softlinks，對固定 point 使用既有 resolver，不沿用 consumer kind 限制。只有單列 `RoleResolutionError` 變成 resolved=None 與 error；source/readability 失敗仍拒絕整個讀取，不從 live 表或檔案補值。
- 全域 alias 表由 `ContextService` 持有。`AliasTable` 原子驗證整批語法與重名，拒絕引用自身或後列。元件與 device 使用同一張表，GUI 不做 kind 或裝置可用性檢查。Notebook 的顯式元件 consumer 仍可要求 kind。
  解析失敗的 `RoleResolutionError` 訊息以 `$alias` 指名：未定義時請使用者在 Aliases 新增，未綁定時請使用者綁定，link 鏈停在空列時指名該列。Link 欄位缺失、目標不是名稱與 kind 不符也各自說明原因。`reason` 維持 `Unknown alias`、`Alias is not bound` 等短分類，供 caller 比對；只有訊息文字給使用者看。Expression 的裸 `$alias` 讀取直接沿用這段訊息，不再加前綴。
  `set_alias_rows` 只改 memory。Inspect Aliases 頁的 Apply 呼叫 `apply_alias_rows`，有 opened entry 時由 resources 原子保存 `aliases.yaml`，未開 entry 時只改 memory。Service 先驗整表，再保存檔案，成功後才安裝 memory 並發布 fact。Storage 失敗保留舊表、舊檔與頁面 draft，顯錯後可 retry。關閉 Inspect 時未存 draft 走 Save／Discard／Cancel；Discard 不撤已 Apply 的表。
  Caller 以 `AliasPresets` 提供名稱到有序表的資料。兩 app 的 composition 保留完整 catalog，只將 default rows 交 `build_session_services(alias_rows=...)` 與 ContextService。`alias_presets` 省略或 None 建立空表，顯式 mapping 必須含 default，default 的空 tuple 合法。Session 不宣告領域 preset，也不從裝置或 cache 填列。每次開 entry 有檔就載入該表，空表與空列不自填。無檔就重設 caller preset，該 entry 第一份非空 point snapshot 以 registry 的唯一候選填 primary 空列。切 point 不重讀檔或換表。明確提交表後不再初始化，直到下次開 entry。格式、結構與 alias 語法錯誤保原 binding，不退回 preset。
  公開 `read_context(aliases=None)` 每次固定 point/library/storage 與全表，GUI 不要求 consumer declarations 或 kind 檢查。`aliases={"qubit": "Q2"}` 只覆寫該次 capture 的具名列，保留 dependent authored links，不改全域 rows、版本或 selection。EMPTY 沿 `no_context` 失敗，缺或未綁定的 `$alias` 在讀取時指名報錯。
  歷史 `read_target(target, roles=...)` 使用已固定的 names，不展開目前全表。GUI adapter 不宣告角色，Tab、Run 與 Load 直接使用 alias-aware capture。App 的 EMPTY 無 override 讀取回 None，保留 context-free 與 deferred 路徑。非空 overrides 仍拒絕 no_context。省略 overrides、None 與 {} 等價。Notebook 的 role resolver 與 via 保留原契約。
- **SessionEnv** 是已發布的 session environment。EMPTY 的 `parameters` 是 None，ACTIVE 持有無 consumer roles 的固定 `ParameterContext`。它不提供 md/ml 欄位；SoC 與 cached value sources 保持各自 owner。
- **Run snapshot** 是依該次 Run 契約捕捉的輸入，接受後不因後續編輯或 refresh 改用新的 cfg。持有 live `SessionEnv` 不等於持有這份不可變證據；SoC、driver 等 live handles 也不作為普通資料整包凍結。

DeviceState.info 與 ValueSourceBinder 的 device sources 是已發布 cache，供 dialog、defaults 與 resolve-once 輸入使用，不是 Run 的控制命令或硬體 snapshot。GUI Run 框架在 hardware lease 後、adapter.run 前讀一次全部已連線 device 的 get_info。任一讀取失敗拒絕 Run，不以 cache 補值；無 device 記 `{}`。這份實驗前 snapshot 隨 RunMetadata 保存，後續 sweep 或 Save 不重讀、不改寫。實驗層 helpers 操作選定 driver，RunContext 不提供 setters。DeviceService 的獨立手動 setup 與 best-effort poll 保留各自責任，不代替 Run snapshot。Run 終局後，host 另請 DeviceService 在背景讀一次已連線儀器。成功讀取由 State owner 更新 cache，失敗台保留原 cache，不改寫固定 snapshot、run 終局或 lease。

## 結構

`persistence.py` 是 app-neutral、強型別的 single-file caretaker；app 注入 codec 並保留
schema/version/trigger policy，shared 層不依賴 Pydantic、Qt、cfg 或 app module。

```
session/
├── types.py            — SocProtocol/SocCfgProtocol/SocHandle/SocCfgHandle/ContextReadiness/SessionEnv（P-a 值型別）
├── aliases.py          — session-only 有序表；原子替換、lazy 名稱 links 與單次 override
├── events.py           — SessionEvent enum + SessionPayload base + 11 payload（Md/Ml/ContextSwitched/Soc/Device{Changed,SetupStarted,SetupFinished} + GateChanged/ConnectionFinished/DeviceOperationFinished）
├── state.py            — SessionState（session_env+devices/DeviceState+preferences/SessionPreferences+shared VersionTable+device mutators）；DeviceStatus
├── ports.py            — session service 依賴的 driven-adapter/seam ports：ExclusionGate(+OperationKind/OperationConflictError)、OwnerScheduler（owner-loop post/call/probe）、BackgroundExecutor（純 off-main 執行器，`submit(work,*,run_in_pool,on_done,on_error)` 無 scopes 參數）、ProgressHub、ProgressEvent/Kind/Transport、DriverFactoryPort、RememberedDevicePort、DeviceRegistryPort（production 使用 session-owned DeviceManager，測試注入 in-memory fake，ADR-0066）、ContextReadPort、ProjectContextPort
├── hardware_gate.py    — RunBlocksHardwareGate：量測硬體互斥矩陣（RUN 擋 RUN/SoC/device mutation；device mutation 依 resource id 互斥；SoC connect 擋 RUN/SoC）；每個lease另保存captured origin、human note與monotonic start，register/release emit `GateChangedPayload`，`snapshot()`只投影active duration、不曝露raw epoch
├── operation_handles.py— OperationHandles（async Handle/Cancel facet，零 kind）+ per-op OperationChannel（單一有序事件 FIFO Settled/Stop，取代舊 FeedbackInbox + poll-loop，ADR-0066）+ captured EventOrigin operation record；`create(cancel_hook=, origin=)`要求caller顯式capture，`event_origin(token)`在live/retained-done record上投影string operation id；`has_cancel_hook`/channel.`can_cancel` 提供一般取消能力查詢；cancel hook例外只log，Stop事件仍入列且cancel_all繼續處理其它operation
├── operation_runner.py — OperationRunner（唯一 kind-agnostic operation 生命週期機制，ADR-0066：ensure_can_start→create→register→progress factory→submit→終局 settle）+ OperationSpec（各 op 把領域 policy 交給 runner）；run/analyze/device/SoC-connect 都是它的 client，runner 只認 port 不認行為，並隔離 terminal callback / cleanup 例外，確保 handle settle 與 exclusion release 仍 best-effort 執行；duplicate settle 是 logged no-op，不重複 cleanup 但會暴露 policy bug
├── scopes.py           — progress_ambient（session 層：pbar ContextVar，無 Qt；ADR-0066，取代舊 executor `_entered`/OffMainScopes 的 pbar 欄位）。Figure 使用 operation-owned Plots，不透過 ambient scope
├── device_control.py   — DeviceControlPort + DeviceControlFacet：device lifecycle/query/progress/event-subscription 的窄 control facet（DeviceService + ProgressService + BaseEventBus），供 shared DeviceDialog、PredictorDialog 與 main remote device handler 使用；cached numeric device-value query 只讀 State，不做 live hardware read；`poll_device_info()` 是可能更新 cache 的 live refresh，`try_poll_device_info() -> bool` 讓 guarded app / UI 明確取得 refresh 是否送出；不提供參數容器讀取
├── predictor_control.py — Read-only explicit-parameter queries and concrete-qubit prefill for dialog and remote callers
├── progress_control.py — ProgressControlPort + ProgressControlFacet：owner-keyed progress subscribe/query 的窄 control facet（ProgressService），供 app-local run UI 使用；operation-id progress 查詢仍留在 app controller/remote operation handler
├── context_control.py  — ContextControlPort + ContextControlFacet：固定 parameter read、entry/point cached queries、全域 alias 表讀寫與 value-source 的窄 facade，轉送 ContextService。Alias 表可在未開 entry 時編輯；set 只改 memory，Apply 才保存 opened entry。
├── library_entry.py    — LibraryEntryForm（Module／Waveform 兩個 family）、open_library_entry 與 new_library_entry：以共用 Form 生成器從持久 Spec 建表單，開啟時固定 FormEnv 與 host 給的 arb 名稱選項；子項 library key 保持 key，文字欄位 expression 在 result 時求成數字；put／replace 產生 LibraryChange，交 apply_library_change 寫入開啟時的 target。不依賴 app，writeback 也用同一建構方式。範本新建經公開 template_input，保留其 specialized input model 建新 State，Expr 與 Slot 解析由生成器負責
├── arb_waveform.py     — ArbWaveformPort／ArbWaveformKeys：qubit-scoped arb 資產的 host 中立 port 與 preview 結果型別；session service 實作，共用 ARB 對話框、Inspect 與各 host 的 remote 面使用
├── setup_control.py    — SetupControlPort + SetupControlFacet：entry open/new、point use/new、cached setup preferences 與獨立 SoC commands；直接公開既有 devices facet 供 cached label 建議，不依賴 app Controller
├── controller_mixin.py — SessionControllerMixin：measure 的 session forwards；shared dialog 使用窄 control facet，entry open/new 直接交 SetupControlPort，不經 project identity 或 app Controller
├── expression.py       — simpleeval 共用求值引擎，封裝函式／算子白名單、AST／數值大小限制及固定 ParameterContext 查詢。裸 `$alias` 求值為名稱 str，不接受字串算術或插值。數值 `evaluate_scalar_expr` 保留 int/float/complex，`evaluate_numeric_expr` 在 real-only 邊界拒絕 complex；`**` 為冪次，`^` 沿用原生 XOR。函式為 sin/cos/tan/sqrt/exp/log/log10/abs，常數 pi/e；math 函式不自動轉 cmath。引用到的保留名稱若與來源衝突則拒絕。EvalRef 是 apply-time resolve marker，不持久化。
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
│   ├── context.py      — ContextService：owner-thread entry/point binding、固定 parameter capture、全域 alias、source facts 與目前 entry 的 ledger capability
│   ├── device.py       — DeviceService（connect/disconnect/setup off-main，**全 port 注入**：gate/bg/progress 必傳；connect/disconnect終局用`DeviceOperationFinishedPayload`，setup重用`DeviceSetupFinishedPayload`，無Qt）；`get_device_unit` 讀有效 YOKO／Fake cached Info 的 `value_unit`，分別為 A／V 與 native；缺 Info 的 UI／snapshot 標 none，不按 type 推測單位；strict binding 缺 Info 拋指名 device 的 `DeviceRegistrationError`，這些查詢不讀硬體、不改 connected／offline 狀態；`poll_device_info(name)` = best-effort off-main live-read（worker 純讀 driver、on_done 主線做 cache 比對+bump+DEVICE_CHANGED；memory-only/connect/disconnect skip，setup/ramp 允許 current-value refresh；單次讀失敗吞掉，不寫 State 於 worker）
│   ├── project_settings.py — ProjectSettingsService：以明確 result/Database roots 呼叫 ResultEntry.open/create，再交 ContextService.install_entry；記住的 entry/connection/device preferences 不證明來源已開啟
│   ├── progress.py     — ProgressService（per-operation pbar 容器 + bound factory，吃 ProgressTransport；**共用**，非 app-held；owner listener 例外逐一 log 並隔離，壞 listener 會 detach，不阻斷其他 listener 或 terminal cleanup）
│   ├── shutdown.py     — ShutdownCoordinator（Qt-free cancel-all + poll state machine，吃 OperationHandles，ADR-0066；QTimer driver 在 adapters/）
│   ├── arb_waveform.py — ArbWaveformService：只依賴 SessionState，資產根目錄依 session_env 的 database path 與 chip／qubit 名稱；每次成功寫入 bump 共用 `arb_waveforms` version 一次；未設 project 時 list 回空，其他操作拒絕 no_project
│   ├── value_sources.py — ValueSourceBinder：訂閱 Context/Device event，把 live SessionState 投影註冊進 `ValueRegistry`；provider 只讀 cached state（device value 只看 `DeviceState.info`，不 poll hardware），並用 owner-level replace/unregister 維護 lifecycle
│   └── build.py        — SessionServices bundle 與 build_session_services：組裝 owning services/facets（含 arb_waveform），從 project_root 注入 result/Database roots，明確注入 entry-scoped asset binder
└── ui/                 — 共用 dialog，依賴各 domain 的窄 control facets
    ├── progress_bar.py   — LightweightProgressBar（native-free QWidget；`setValue` 只更新狀態並排程 repaint，避免高頻 progress 卡在 `QProgressBar.setValue`）
    ├── progress_stack.py — ProgressStack widget（唯一拉 Qt 的 progress 件；高頻 render 只對變更過的 maximum/format/value 寫入 LightweightProgressBar；可注入 profiling callback 拆量 derive / value writes）
    ├── alias_panel.py — AliasPanel：Inspect 的 Aliases 頁。編輯有序 authored rows，透過 ContextControlPort 原子提交；唯讀解析欄投影已安裝的表。Load preset 選單列出 host 傳入的具名 preset，選一份整張取代草稿、不合併、不提交，Apply 才保存；None 表示無 catalog，選單停用。唯讀時同樣停用。只回報 Unsaved，不自己詢問
    ├── component_tree.py — ComponentTreeWidget：Field／Value／Unit／Metadata 四欄樹，Metadata 以單行 flow YAML 與多行 tooltip 顯示 caller 的 opaque mapping，不解讀 key；依 `component_field_specs` 列出全部欄位（未設定顯示 `(unset)`），Value 就地編輯；父欄位未設定時寫入最淺一層未設定祖先的巢狀 mapping。寫入走 owner，UI 不直接寫文件
    ├── setup_dialog.py — Result entry、Working points 與 Connection 的 shared dialog。New entry／Open 與 point Switch／Create 都交既有 owner。Working points 清單顯示 description，Create 明確提交 Label、Clone from 與 Description。Track device 只用 cached value＋unit 建議可修改的 Label，不讀硬體、不綁裝置。Clone from 的 `(empty)` 表示新 point 使用 setup seed 與空 library。Reopen 使用 cached queries，不切 binding、不連硬體
    ├── device_dialog.py  — 設備管理；吃 `DeviceControlPort` + 獨立 parameter provider，eval mode apply-time resolve 不污染 device facet；per-device panel lazy-import zcu_tools.device.*；host 可切 `set_read_only(read_only=True)` 進 inspection mode：Add/Drop/Apply/Reconnect/Stop 與 panel edit controls 禁用，Refresh、selection、poll、Close 保持可用；dialog-scoped + selection-scoped QTimer（1s）對當前選取 device 呼 `try_poll_device_info`，結果經 DEVICE_CHANGED 流回並保留目前選取；timer 僅可見+有選取時跑，hide/close/無選取即停（不常駐、不全 device 輪詢）
    ├── predictor_dialog.py — Independent fluxonium or SQUID transmon trials, frequency curves (matrix curves only for models that provide them) and local Calibrate. Calibration displays actual MHz and clamped from the shared model result; it changes only dialog bias/cache, never a component. No Apply or session model. Device values and frequency expressions remain optional inputs.
    ├── predictor_canvas.py — PredictorCurveCanvas：f_ij 曲線 over device value（雙 x 軸 flux/value）+ 可拖動 flux 垂線（方案 B：marker 不折、出窗平移）；支援多條 transition 同時 highlight；host 可關閉 marker interaction 讓 live view 只由外部 value 驅動；主線程同步繪圖不需 marshal
    ├── inspect_dialog.py — InspectDialog：各 host 共用的唯一 Inspect。必填協作者為 ContextControlPort、event bus、範本目錄、arb 名稱來源、host 的 ARB 開啟入口、DialogPresenter 與具名 alias preset catalog（None 表示無 catalog），沒有可選能力或依 host 隱藏的按鈕。Parameters 是元件樹；Modules 選取條目時以 library entry 表單在固定 capture 開啟，右側 Name、Saved／Unsaved、Revert、Apply 與 FormEditor（包在 scroll area）。Unsaved 由 Name、FormState.is_dirty 與 editor 未解析文字決定。is_dirty 包含 selection-only 差異，不只比較 raw data；切換、刪除或關閉時以 presenter 的 Save／Discard／Cancel 詢問。Aliases 頁是 AliasPanel，開啟時讀一次已安裝的表，關閉時未存 draft 同樣三選一，Save 被拒時留在 Aliases 頁。唯讀只鎖輸入，表單與 alias 表仍可讀、捲動與折疊
    ├── new_entry_dialog.py — NewEntryDialog：Inspect 的 New。每次選範本讀一次 capture，以 new_library_entry 建新 State 與 FormEditor，欄位、折疊與捲動回範本初始；只保留使用者打過的 Name，未打時建議 default_name 並在同 family 內加 _2、_3 去重。Create 經 apply_library_change put，被拒時經 presenter 報錯並留在對話框
    └── arb_waveform_dialog.py — ArbWaveformDialog：arb 資產管理對話框，只收 ArbWaveformPort。Measure 經具名 `arb_waveform` dialog 開，workflows 直接開同一類別
```

共用 Setup、Inspect、Devices 與 ARB 對話框由宿主呼叫 `set_read_only` 決定是否可編輯，不讀控制權或 run 狀態。Setup 唯讀時保留 point 選取、description、paths、查詢刷新與 Close，停用 entry／point 提交及 Connect。連線完成與事件刷新不會解除宿主的唯讀狀態。ARB 唯讀時停用編輯與 Save，保留鎖前草稿；asset 選取、Refresh、預覽與 Close 照常可用。解鎖後 Save 仍受驗證結果限制。Inspect 唯讀時「Arb Waveforms…」入口仍可按，ARB 依宿主規則以唯讀開啟。

Simulation is selected at measure startup with `--simulate <environment.py>`.
Composition injects the returned manager and calls `DeviceService.publish_initial_devices`
before preference restore. Live rows win over same-name remembered rows; other
remembered rows stay MEMORY_ONLY without connecting. `install_startup_soc` publishes
the bound SoC/config through the connection owner and fixes that pair for this process.
Session has no environment builder or dynamic simulation coordinator. The host drains
work and closes the same manager; Controller close alone does not close drivers.

Device panel 保留原 mode 顯示與 output／value／rampstep 控制，不提供獨立 `value_unit` 編輯。完整 cfg 的單位由 Info 驗證，snapshot 與 recipe 沿同一儲存事實。

## Analysis-only operations

Fluxdep search 重用 OperationRunner、OperationHandles 與 ProgressService。
Runner 的 gate／progress injection 可為 None。begin 在建立 handle 前拒絕缺少的必要 facet。
純分析不用注入硬體 gate。OperationSpec 的 optional on_opened hook 在 lease registration、progress notification
與 submit 前交出 token，讓 domain owner 先提交 admission。帶 exclusion 的 hook 只提交 owner
state，不通知 listeners 或呼叫 reentrant UI；無 exclusion 的分析 owner 可發布 activity。
Hook、registration、factory 或 submit 失敗時，runner 清理並重拋。
Measure client 保留自己的硬體 policy。

## Same-operation continuation

Domain owner 在 producer delivery 後可保留未 settle 的 operation，再以 `continue_operation`
提交下一個 producer。Runner 重用同一 token、原始 event origin 與 settle closure，不新增
handle、lease、cancel hook 或 progress factory。Owner 負責 producer exclusivity 與 closure
歸屬；Runner 不另建 producer registry。同步 continuation submit 失敗保留 ownership，讓 owner
重試或終結。Worker failure 交給 terminal policy，escaping policy exception 則 settle failed。
最終清理仍經原來的 call-once closure，只 discard progress、settle handle、release lease 一次。

## Operation markers

`DeviceDialog` reads named driver activity through `DeviceControlPort` on the GUI
thread while visible. Each row uses the compact blue `●` during an operation and
for about one second after its last finish. Repeated short operations keep the
marker steady. Activity reads do not access hardware, rebuild selection, or decide
admission. Apply, experiments, remote callers and workflows share the same driver
observations. State-owned setup status still controls Stop and editing, separately
from these markers. The selected-device value poller keeps its independent lifetime.

## 關鍵設計

- **Add input lifetime**：ComponentTree 的 read-only mode 停用 Add。Local Add dialog 不接 control authority；modal 期間 remote adapter 拒絕 Local 接管。Finished 退休輸入，晚到 submit 不會重新寫入。

- **expected failure逐一opt in**：session service只讓caller可修正的concrete exceptions加入
  `ExpectedError` taxonomy；`OperationConflictError`、context validation、predictor readiness與
  value-lookup expected leaves攜帶typed category。`ValueLookupError` parent、`ProviderError`、
  SoC/device async terminal與registration failures保持unexpected，不因共同ancestry被降級
  （ADR-0068）。
- **app 注入 infra 經 port**：concrete `OperationGate`（衝突 policy，app-local）+ runtime-specific `BackgroundExecutor` 各自建；Qt composition 使用 `session/adapters/qt_background.py` 的 `BackgroundRunner`，headless composition 使用注入 `ManualOwnerScheduler` 的 `ThreadPoolBackgroundExecutor`。兩者都保留 shared-pool / dedicated-worker 語意，terminal callback 一律回 owner thread；具體 owner 可在 teardown 呼 `quiesce()` 等待 worker 與 terminal delivery。
  `SessionServices.runner: OperationRunner` 與 `SessionServices.progress: ProgressService`
  保留 build 輸入的同一 instances。共用 session 的 operation caller 從這份 bundle
  取用，不另傳 runner／progress。Composition root 仍建立基礎物件並注入 build。
  Operation policy、exclusion 與 owner-thread 契約不變。
  `tests/gui/session/test_headless_composition.py` 在同一程序穿過正式 build 與
  offline MockSoc success/cancel lifecycle，驗證無 `QApplication` 的 owner-thread、
  progress 與 exclusion 行為。`ProgressService`、`QtProgressTransport` 與
  `QtShutdownDriver` 是共用 session service/adapter。
  Session service 只依賴
  `ExclusionGate`、`BackgroundExecutor`、`ProgressHub` ports。
  `BackgroundExecutor` 只宣告 `submit`，`quiesce` 留在 concrete adapter。
- **ExclusionGate str-keyed**：session `OperationKind`（soc/device kinds）+ app 自己的 kind（measure `RUN`）用 wire 字串比較，故兩 enum 同一 gate。共享硬體互斥規則住在 `RunBlocksHardwareGate`，app-local gate 只保留 RUN kind 與 import boundary。
- **app 繼承/實作**：measure `State(SessionState)`；Controller 暴露 `setup_control: SetupControlPort` / `context_control: ContextControlPort` / `device_control: DeviceControlPort` / `predictor_control: PredictorControlPort` / `progress_control: ProgressControlPort` 供 shared dialog、main remote handler 與 app-local run UI 走窄 facets。
- **Inspect 元件樹**：`ComponentTreeWidget` 透過 context facet 讀取 point／setup 文件並提交元件與參數命令。`ContextService` 擁有驗證與資源交易，widget 不寫檔。Setup 範本編輯不改現有 point 或 alias 表。共用 Inspect 的 Modules 以 library entry 表單編輯；子項 library key 保持 key，Apply 以一個 replace 寫 Name 與 Spec，被拒時保留 draft。Library structure change 使用開啟表單時的固定 capture，透過 ContextService 提交；Delete 先固定 capture 再確認。
- **Entry binding**：GUI Setup 和 remote entry.new/open 使用同一個 ProjectSettingsService。EntryInfo 提供 UUID、任意目錄名稱與 result／database 絕對路徑。Open 只開既有 entry，New 明確建立。成功發布 entry metadata，不自選 point、不碰硬體。失敗保留舊 binding；New 已建資源而安裝失敗時保留 entry，以 Open 恢復，不重試 New
- **Entry 與 point**：ResultEntry 擁有 entry roots、setup 與 points。ProjectSettingsService 只負責明確 open/new 與 preferences，ContextService 安裝 binding。Point 操作作用在目前 opened entry，不從 chip／qubit／resonator 推導身分
- **operation callback 邊界**：`OperationRunner` terminal delivery 會 catch/log policy callback 例外並把 operation settle failed；`settle()` 的 progress discard / handle settle / exclusion release 各自 best-effort 執行，避免 progress UI 或 cleanup 例外讓 handle 永遠 pending。第二次呼叫同一 settle closure 會記 error 並 no-op，讓 double-terminal policy bug 可診斷但不破壞 cleanup idempotence。`ProgressService` owner listener 逐一隔離，壞 listener 不阻斷其他 view 或 terminal cleanup。
- **共用層邊界**：types、events、operation_handles、ports、state、pbar_host 與 control facets 不依賴 Qt widgets。UI 和 Qt adapters 承接 Qt；headless composition 的行為由 `tests/gui/session/test_headless_composition.py` 驗證。Session 不反向依賴 app，由 `.importlinter` 的 `session-no-app` contract（C2）檢查，不把 composition 測試當成所有 leaf 的 import 禁令。
- **wire name 來源**：`SessionEvent.X` 的字串值即 wire event name；measure-gui 的 wire-name lock 測試（`test_remote_event_dialog_view.py`）鎖全集，搬移/改名 payload 不得動字串值。
- **`SessionEnv.values`**：只攜帶 read-only `ValueLookup` facade，供 default generation / resolve-once 讀目前 session 投影；`ContextService.list_value_sources/read_value_source` 是 GUI/remote 共用查詢面，mutable `ValueRegistry` 只在 session composition root / source binder 內使用。`ValueSourceBinder` 以 owner-scoped replace/unregister 維護 `context.*` / `project.*` / `device.<name>.*`；named device source 包含 `device.<name>.name` 與 cached info（如 `value`/`status`），device provider 只讀 cached `DeviceState.info`、不 poll hardware，也不推導 active/flux 語義。內部 `ValueRef`／`resolve_value_ref` 保留立即讀值的型別檢查，舊 GUI token helper 已退休。`SessionEnv` 仍是 live environment facade，不是 snapshot。

跨模組設計見 ADR-0067（shared session、event ownership、frontend 邊界、參數容器／library 寫入權威）、ADR-0065（cfg 編輯與 lowering）、ADR-0002（version table 與 off-main handler）、ADR-0066（operation handle、執行、取消、關閉與跨線程互動 channel）。

Run、WB 與目前 point 的 Inspect Set 都要求先選取或建立工作點。GUI 停用操作並提示下一步，application 入口以 failed precondition 拒絕。WB 的 selection 只允許開始命令，不替換候選的 captured SavedPoint 目的地。同 entry 選取另一個 point 後仍寫原 point；跨 entry 仍拒絕。Setup template 編輯獨立於目前 point，不修改既有 point。

## 獨立 predictor

`simulate.fluxonium.predict.FluxoniumParams` 與 `simulate.transmon.TransmonParams` 保存可未知的具名物理輸入，missing() 指出缺項。
齊全且有效才建立中性 query model。Service 不持有 installed model，不讀 dialog。
App 注入使用者的 params_for，把當前 capture 中的具體元件轉成參數。
Framework 不猜元件欄位，也不 import 使用者套件。查詢與 Calibrate 不寫元件或版本。
Dialog 的元件選單初始選 alias qubit；Load component 才經注入的 params_for 預填，選取或切換工作點不改 trial。欄位勾選代表已知，取消勾選保留 None；缺項會明示，完整且有效才計算。
模型選單有 Fluxonium 與 SQUID transmon，各自顯示自己的能量欄位，flux 欄位共用。Fluxonium 顯示頻率與 |n|、|phi| 曲線。Transmon 沒有矩陣元素能力，只顯示頻率曲線。Load component 依元件 kind 選模型；params.json 只預填 fluxonium。
Measure registry 啟用獨立 predictor.json user cache。Dialog 持有 versioned PredictorSnapshot 與 SingleFileCaretaker，保存 model 與 nullable params；每次編輯與關閉保存，重建 dialog 還原。Standalone callers 以 enable_parameter_cache(cache_dir=...) 明確啟用與隔離 cache，啟用時立即還原。Cache decode failure 回空 trial 並顯示錯誤，不讀寫 component DocumentStore。
