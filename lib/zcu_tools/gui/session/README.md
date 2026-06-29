**Last updated:** 2026-06-29 — setup dialog project layout

# gui/session/ — 量測 session core（measure + autofluxdep 共用）

把 measure-gui 的「量測 session core」（context 系統 + SoC 連線 + 多 device + setup/device/inspect/predictor dialog）提到的共用層。對標 `gui/remote`、`gui/plotting`。每個 measurement-session app 注入自己的 app-local infra（gate + background）複用這層；session 模組**永不**反向 import `gui.app.*`。measure 與 autofluxdep **都已完整複用**這層。

## 結構

```
session/
├── types.py            — SocProtocol/SocCfgProtocol/SocHandle/SocCfgHandle/ContextReadiness/ExpContext（P-a 值型別）
├── events.py           — SessionEvent enum + SessionPayload base + 8 payload（Md/Ml/ContextSwitched/Soc/Predictor/Device{Changed,SetupStarted,SetupFinished}）
├── state.py            — SessionState（exp_context+devices/DeviceState+startup_prefs/StartupPrefs+shared VersionTable+device mutators）；DeviceStatus
├── ports.py            — session service 依賴的 driven-adapter/seam ports：ExclusionGate(+OperationKind/OperationConflictError)、BackgroundExecutor（純 off-main 執行器，`submit(work,*,run_in_pool,on_done,on_error)` 無 scopes 參數）、ProgressHub、ProgressEvent/Kind/Transport、DriverFactoryPort、RememberedDevicePort、DeviceRegistryPort（GlobalDeviceManager classmethod 面的 instance 化，DeviceService 依契約存取、測試注 in-memory fake，ADR-0026 §6）、ProjectIOPort、ContextReadPort、StartupContextPort
├── operation_handles.py— OperationHandles（async Handle/Cancel facet，零 kind）+ per-op OperationChannel（單一有序事件 FIFO Settled/Message/Stop，取代舊 FeedbackInbox + poll-loop，ADR-0025）+ OperationOutcome/OperationStatus/AwaitResult/CancelHook；`create(cancel_hook=)`、`has_cancel_hook`/channel.`can_cancel`（gate 'Send & Stop' 鈕，無 op-kind 知識）
├── operation_runner.py — OperationRunner（唯一 kind-agnostic operation 生命週期機制，ADR-0026 §1：ensure_can_start→create→register→progress factory→submit→終局 settle）+ OperationSpec（各 op 把領域 policy 交給 runner）；run/analyze/device/SoC-connect 都是它的 client，runner 只認 port 不認行為
├── scopes.py           — progress_ambient（session 層：pbar ContextVar，無 Qt；ADR-0026 §2，取代舊 executor `_entered`/OffMainScopes 的 pbar 欄位）。figure_ambient（Qt）住 app 層 `gui/app/main/services/scopes.py`
├── notify_handles.py   — NotifyChannel/NotifyHandles（agent→user prompt 的跨線程 channel，事件集 Reply/Dismiss/Timeout，獨立於 operation 的 Settled/Message/Stop；鏡像 OperationChannel 四不變式，ADR-0025）
├── controller_port.py  — SessionControllerPort：共用 dialog 依賴的窄 Controller 面（setup/context/value lookup/connection + device lifecycle/queries/progress + predictor load/set_model_params/clear/predict/curve + inspect md/ml）；回傳宣告對 BaseEventBus 故 app EventBus covariant 滿足
├── controller_mixin.py — SessionControllerMixin：SessionControllerPort 的「共用 body」——兩 app 逐字相同的 port forward（讀 6 個 abstract service accessor `_soc_svc`/`_pred_svc`/`_ctx_svc`/`_dev_svc`/`_startup_svc`/`_progress_svc`，以 annotation-only 宣告型別由 concrete Controller 供應同名 attr，**不**用 @property 以免 data-descriptor `__set__` 撞既有 `self._x_svc=` 賦值）。app 各自只留 body 真正分歧的 override：`apply_startup_project`（measure 回 resolved dict/WIRE-48、autofluxdep 回 bool）、`get_project_root`（讀 app `_project_root`）、`get_bus`（回 app EventBus subtype）。import-clean（service/request 型別全 TYPE_CHECKING-only），列入 test_shared_layer 守
├── expression.py       — 安全 numeric expression evaluator（evaluate_numeric_expr + coerce_eval_result，純函式吃 MetaDict）+ EvalRef（frozen dataclass：eval 模式欄位的 read_raw() marker，apply 時 resolve，不持久化）；import-clean leaf
├── value_lookup.py     — read-only value source lookup（`ValueLookup`/`ValueRegistry`/owner-scoped replace/unregister/`ValueRef` resolve-once helpers）；純 session leaf，provider 來源由 service binder 負責，使用端只見 lookup
├── pbar_host.py        — ProgressBar(worker)/ProgressBarModel(主線程 SSOT)，Qt-free
├── adapters/
│   └── qt_progress_transport.py — QtProgressTransport（worker→主線程 progress marshal，queued signal；app-agnostic）
├── services/
│   ├── connection.py   — SoCConnectionService（SoC connect op；硬體 facet、OperationRunner client、cancel_hook=None 無 cancellation point；終局經 connection_finished/connection_failed signal，typed requests/failures）
│   ├── predictor.py    — PredictorService（predictor load/set_model_params〔typed EJ/EC/EL→FluxoniumPredictor，走 install_predictor in-memory seam〕/clear/install + predict_freq + 批次曲線計算 predict_freq_curve/predict_matrix_element_curve〔委派 simulate `FluxoniumPrediction` engine 的 affine/array paths〕；get_predictor_info 含 EJ/EC/EL；純函式 read_fluxdep_fit_params〔params.json→typed model query〕；純計算，無 Qt signal/runner/gate，擁有 exp_context.predictor 寫 seam，ADR-0026 §5 自 connection.py 拆出）
│   ├── context.py      — ContextService（context-switch + md ops + ml del/rename + 單一寫入 primitive apply_ml_writes，CfgSchema lowering 經 callback 注入；MdValueError/MlEntryValidationError）
│   ├── device.py       — DeviceService（connect/disconnect/setup off-main，**全 port 注入**：gate/bg/progress 必傳）；`_mode_dependent_unit(dev)` module-level helper 集中 YOKOGS200 voltage/current→V/A 判斷（`get_device_unit` + `get_device_unit_strict` 共用，消除逐字重複）；`poll_device_info(name)` = best-effort off-main live-read（worker 純讀 driver、on_done 主線做 cache 比對+bump+DEVICE_CHANGED；memory-only/mutating skip、單次讀失敗吞掉，不寫 State 於 worker）
│   ├── startup.py      — StartupService + PersistedStartup/PersistedDeviceEntry memento + requests；委派 `gui/result_scope.py` 做 result-scope discovery / generated path / params.json identity migration
│   ├── progress.py     — ProgressService（per-operation pbar 容器 + bound factory，吃 ProgressTransport；**共用**，非 app-held）
│   ├── io_manager.py   — IOManager（ExperimentManager 包裝，實作 ProjectIOPort；**共用**）
│   ├── mock_flux.py    — MockFluxProvisioner（FLUX-AWARE-MOCK：訂閱 SOC_CHANGED，mock connect 時 ① 綁定/provision fake_flux 源 ② 從 mock soc 自身 SimParams 建 FluxoniumPredictor 經 connection seam 安裝——不蓋使用者已載入的 predictor）
│   ├── predictor_from_sim.py — build_predictor_from_simparams(SimParams)→FluxoniumPredictor（純函式；橋接兩個彼此獨立的 lib leaf，放 session 層避免在 program/simulate 間造新跨依賴；測試共用）
│   ├── value_sources.py — ValueSourceBinder：訂閱 Context/Predictor/Device event，把 live SessionState 投影註冊進 `ValueRegistry`；provider 只讀 cached state（device value 只看 `DeviceState.info`，不 poll hardware），並用 owner-level replace/unregister 維護 lifecycle
│   └── build.py        — SessionServices bundle（soc_connection/predictor/context/device/startup）+ build_session_services(state,bus,gate,handles,background,progress,io_manager,runner,driver_factory?,device_registry?)
└── ui/                 — 共用 dialog（吃 SessionControllerPort）
    ├── progress_stack.py — ProgressStack widget（唯一拉 Qt 的 progress 件）
    ├── setup_dialog.py   — Project + Context + Connection 合併（QSplitter）；Project group 先選 Result scope，再填 chip/qubit/res，並在同一 group 提供 Apply startup context
    ├── device_dialog.py  — 設備管理；per-device panel lazy-import zcu_tools.device.*；dialog-scoped + selection-scoped QTimer（1s）對當前選取 device 呼 `poll_device_info`，結果經 DEVICE_CHANGED 流回（repaint 由 Phase 1 保留選取路徑接住）；timer 僅可見+有選取時跑，hide/close/無選取即停（不常駐、不全 device 輪詢）
    ├── predictor_dialog.py — FluxoniumPredictor 載入/定義 + 頻率預測（左控制 / 右繪圖兩欄；load/clear/predict + 可編輯 EJ/EC/EL/flux_half/flux_period 欄位 + Apply〔set_predictor_model_params 建+裝〕+「Load params.json→fields」填欄位〔不自動裝〕+ active-model 讀回；PREDICTOR_CHANGED 訂閱；host 可用 `persistent_on_close=True` 讓 Close/視窗 X 只 hide、不 emit finished，保留已算曲線）；右欄 PredictorCurveCanvas 與 Flux value spinbox 雙向連動（debounce）
    ├── predictor_canvas.py — PredictorCurveCanvas：f_ij 曲線 over device value（雙 x 軸 flux/value）+ 可拖動 flux 垂線（方案 B：marker 不折、出窗平移）；主線程同步繪圖不需 marshal
    ├── inspect_base.py   — InspectDialogBase：md tab + ml view/rename/del；hook `_build_extra_toolbar_buttons` 讓子類在 Refresh 左側加 app-specific 入口，hook `_build_extra_ml_buttons` / `_on_ml_selection_changed` 讓子類加 ml create/modify。**consumers**：measure subclass（InspectDialog，加 Arb Waveforms 入口與 CfgEditor create/modify）；autofluxdep **直接用不 subclass**（不要 create/modify）
    └── value_source_input.py — QLineEdit value-source helper：`@{prefix` 顯示分段 completion（`@{`→頂層、`dev`→`device.` 並展開下一段、`device.`→下一段），Tab/Backtab 接受預設候選時若當前候選全是 namespace 會自動補 `.` drill down，補到完整 key 才加 `}`；在 `@{full.key} ` 後輸入空格才立即 resolve 並以文字替換；token 完成或解析後明確 hide popup；依賴 `ValueSourceInputHost` port，不知道 ContextService/LiveModel
```

## 關鍵設計

- **app 注入 infra 經 port**：concrete `OperationGate`（衝突 policy，app-local）+ 共用 `BackgroundRunner`（`gui/background.py`，純 off-main 執行器，三 app 共用同一 class——owner 直接持具體 runner 才能呼 anti-segfault 的 `quiesce()`）各自建；`ProgressService`/`IOManager`/`QtProgressTransport` 已 promote 成**共用**。session service 只依賴 `ExclusionGate`/`BackgroundExecutor`（只宣告 `submit`，`quiesce` 不進 port）/`ProgressHub`/`ProjectIOPort` port。
- **ExclusionGate str-keyed**：session `OperationKind`（soc/device kinds）+ app 自己的 kind（measure/autofluxdep 各自 `RUN`）用 wire 字串比較，故兩 enum 同一 gate。
- **app 繼承/實作**：measure `State(SessionState)`、autofluxdep `AutoFluxDepState(SessionState)`；兩 Controller 結構上實作 `SessionControllerPort`（pyright 在各自 dialog call site 驗證 conformance）。
- **startup project return contract**：`SessionControllerPort.apply_startup_project` 允許 `bool | dict[str, str]`。共用 setup dialog 只看 truthiness；measure 的 remote `startup.apply` / `gui_project_apply` 回 resolved project dict（WIRE 48，含 `result_dir` / `database_path` / `params_path` / `scope_id`），autofluxdep controller 仍回 bool。
- **Result Scope vs Active Context**：上層 project scope 是 `result/<chip>/<qub>/params.json` 所在目錄；下層 active context 仍是 `ExperimentManager` 在該 `result_dir` 下管理的 context module。`ResultScopeManager` 掃描 `result/**/params.json`，並透過 `migrate_params_v0_to_v1_project_info` 原地升級 v0 params：canonical `project.{chip_name, qubit_name}` 優先；缺 project 時從 `result/` 下路徑推導，兩層=`chip/qubit`、一層=`name/name`、其它=`unknown/unknown`，`resonator_name` 補 `unknown`。`StartupService.apply_project` 未給 `scope_id` 時使用/建立 generated scope，給 `scope_id` 時只接受掃描出的既有 scope，不接受任意路徑。
- **import-clean leaf**（不得拉 Qt/matplotlib/gui.app.*，`tests/gui/test_shared_layer.py` 守）：types/events/operation_handles/ports/state/pbar_host/controller_port。`adapters/` + `ui/*` + `services/*` 是 Qt/重，不列。
- **wire name 來源**：`SessionEvent.X` 的字串值即 wire event name；measure-gui 的 wire-name lock 測試（`test_remote_event_dialog_view.py`）鎖全集，搬移/改名 payload 不得動字串值。
- **`ExpContext.values`**：只攜帶 read-only `ValueLookup` facade，供 default generation / resolve-once 讀目前 session 投影；`ContextService.list_value_sources/read_value_source` 是 GUI/remote 共用查詢面，mutable `ValueRegistry` 只在 session composition root / source binder 內使用。`ValueSourceBinder` 以 owner-scoped replace/unregister 維護 `context.*` / `project.*` / `predictor.*` / `device.<name>.*`；named device source 包含 `device.<name>.name` 與 cached info（如 `value`/`status`），device provider 只讀 cached `DeviceState.info`、不 poll hardware，也不推導 active/flux 語義。`value_source_input.py` 把此查詢面包成可注入的 GUI token helper；它只替換輸入文字，不寫 cfg/md。`ExpContext` 仍是 live environment facade，不是 snapshot。

跨模組設計見 ADR-0002/0004/0005/0006/0019/0020/0021/0025/0026（0021：event ownership——domain module 擁有 enum+payload、app 組裝；0025：跨線程互動 channel——OperationChannel/NotifyChannel；0026：operation abstraction——OperationRunner + scope-as-adapter-input + State write port + ConnectionService 拆 SoC/Predictor + DeviceRegistryPort）。**autofluxdep 已完整複用**（session-core extraction S1–S5：組 session services + 實作 SessionControllerPort + run 讀 exp_context + 用共用 setup/device/predictor dialog；見 ADR-0020 + autofluxdep/README）。
