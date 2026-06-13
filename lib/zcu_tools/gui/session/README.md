**Last updated:** 2026-06-13 — predictor dialog 互動視覺化（右側 f_ij 曲線 + 可拖動 flux 垂線）

# gui/session/ — 量測 session core（measure + autofluxdep 共用）

把 measure-gui 的「量測 session core」（context 系統 + SoC 連線 + 多 device + setup/device/inspect/predictor dialog）提到的共用層。對標 `gui/remote`、`gui/plotting`。每個 measurement-session app 注入自己的 app-local infra（gate + background）複用這層；session 模組**永不**反向 import `gui.app.*`。measure 與 autofluxdep **都已完整複用**這層。

## 結構

```
session/
├── types.py            — SocProtocol/SocCfgProtocol/SocHandle/SocCfgHandle/ContextReadiness/ExpContext（P-a 值型別）
├── events.py           — SessionEvent enum + SessionPayload base + 8 payload（Md/Ml/ContextSwitched/Soc/Predictor/Device{Changed,SetupStarted,SetupFinished}）
├── state.py            — SessionState（exp_context+devices/DeviceState+startup_prefs/StartupPrefs+shared VersionTable+device mutators）；DeviceStatus
├── ports.py            — session service 依賴的 driven-adapter/seam ports：ExclusionGate(+OperationKind/OperationConflictError)、OffMainScopes、BackgroundExecutor、ProgressHub、ProgressEvent/Kind/Transport、DriverFactoryPort、RememberedDevicePort、ProjectIOPort、ContextReadPort、StartupContextPort
├── operation_handles.py— OperationHandles/OperationOutcome/OperationStatus（async Handle/Cancel facet，零 kind）
├── controller_port.py  — SessionControllerPort：共用 dialog 依賴的窄 Controller 面（setup/context/connection + device lifecycle/queries/progress + predictor load/clear/predict/curve + inspect md/ml）；回傳宣告對 BaseEventBus 故 app EventBus covariant 滿足
├── pbar_host.py        — ProgressBar(worker)/ProgressBarModel(主線程 SSOT)，Qt-free
├── adapters/
│   └── qt_progress_transport.py — QtProgressTransport（worker→主線程 progress marshal，queued signal；app-agnostic）
├── services/
│   ├── connection.py   — ConnectionService（SoC connect worker、predictor IO + 批次曲線計算 predict_freq_curve、typed requests）
│   ├── context.py      — ContextService（context-switch + md ops + ml del/rename + 單一寫入 primitive apply_ml_writes，CfgSchema lowering 經 callback 注入；MdValueError/MlEntryValidationError）
│   ├── device.py       — DeviceService（connect/disconnect/setup off-main，**全 port 注入**：gate/bg/progress 必傳）；`_mode_dependent_unit(dev)` module-level helper 集中 YOKOGS200 voltage/current→V/A 判斷（`get_device_unit` + `get_device_unit_strict` 共用，消除逐字重複）；`poll_device_info(name)` = best-effort off-main live-read（worker 純讀 driver、on_done 主線做 cache 比對+bump+DEVICE_CHANGED；memory-only/mutating skip、單次讀失敗吞掉，不寫 State 於 worker）
│   ├── startup.py      — StartupService + PersistedStartup/PersistedDeviceEntry memento + requests + derive_project_paths
│   ├── progress.py     — ProgressService（per-operation pbar 容器 + bound factory，吃 ProgressTransport；**共用**，非 app-held）
│   ├── io_manager.py   — IOManager（ExperimentManager 包裝，實作 ProjectIOPort；**共用**）
│   ├── mock_flux.py    — MockFluxProvisioner（FLUX-AWARE-MOCK：訂閱 SOC_CHANGED，mock connect 時 ① 綁定/provision fake_flux 源 ② 從 mock soc 自身 SimParams 建 FluxoniumPredictor 經 connection seam 安裝——不蓋使用者已載入的 predictor）
│   ├── predictor_from_sim.py — build_predictor_from_simparams(SimParams)→FluxoniumPredictor（純函式；橋接兩個彼此獨立的 lib leaf，放 session 層避免在 program/simulate 間造新跨依賴；測試共用）
│   └── build.py        — SessionServices bundle + build_session_services(state,bus,gate,handles,background,progress,io_manager,driver_factory?)
└── ui/                 — 共用 dialog（吃 SessionControllerPort）
    ├── progress_stack.py — ProgressStack widget（唯一拉 Qt 的 progress 件）
    ├── setup_dialog.py   — Project + Context + Connection 合併（QSplitter）
    ├── device_dialog.py  — 設備管理；per-device panel lazy-import zcu_tools.device.*；dialog-scoped + selection-scoped QTimer（1s）對當前選取 device 呼 `poll_device_info`，結果經 DEVICE_CHANGED 流回（repaint 由 Phase 1 保留選取路徑接住）；timer 僅可見+有選取時跑，hide/close/無選取即停（不常駐、不全 device 輪詢）
    ├── predictor_dialog.py — FluxoniumPredictor 載入 + 頻率預測（左控制 / 右繪圖兩欄；load/clear/predict + PREDICTOR_CHANGED 訂閱）；右欄 PredictorCurveCanvas 與 Flux value spinbox 雙向連動（debounce）
    ├── predictor_canvas.py — PredictorCurveCanvas：f_ij 曲線 over device value（雙 x 軸 flux/value）+ 可拖動 flux 垂線（方案 B：marker 不折、出窗平移）；主線程同步繪圖不需 marshal
    └── inspect_base.py   — InspectDialogBase：md tab + ml view/rename/del；hook _build_extra_ml_buttons/_on_ml_selection_changed 讓子類加 ml create/modify。**consumers**：measure subclass（InspectDialog，加 CfgEditor create/modify）；autofluxdep **直接用不 subclass**（不要 create/modify）
```

## 關鍵設計

- **app 注入 app-local infra 經 port**：concrete `OperationGate`（衝突 policy）+ `BackgroundService`（measure 帶 QtLivePlot facet / autofluxdep 瘦版無 figure routing）各 app 自持；`ProgressService`/`IOManager`/`QtProgressTransport` 已 promote 成**共用**。session service 只依賴 `ExclusionGate`/`BackgroundExecutor`/`ProgressHub`/`ProjectIOPort` port。
- **ExclusionGate str-keyed**：session `OperationKind`（soc/device kinds）+ app 自己的 kind（measure/autofluxdep 各自 `RUN`）用 wire 字串比較，故兩 enum 同一 gate。
- **app 繼承/實作**：measure `State(SessionState)`、autofluxdep `AutoFluxDepState(SessionState)`；兩 Controller 結構上實作 `SessionControllerPort`（pyright 在各自 dialog call site 驗證 conformance）。
- **import-clean leaf**（不得拉 Qt/matplotlib/gui.app.*，`tests/gui/test_shared_layer.py` 守）：types/events/operation_handles/ports/state/pbar_host/controller_port。`adapters/` + `ui/*` + `services/*` 是 Qt/重，不列。
- **wire name 來源**：`SessionEvent.X` 的字串值即 wire event name；measure-gui 的 wire-name lock 測試（`test_remote_event_dialog_view.py`）鎖全集，搬移/改名 payload 不得動字串值。

跨模組設計見 ADR-0002/0004/0005/0006/0019/0020/0021（0021：event ownership——domain module 擁有 enum+payload、app 組裝）。**autofluxdep 已完整複用**（session-core extraction S1–S5：組 session services + 實作 SessionControllerPort + run 讀 exp_context + 用共用 setup/device/predictor dialog；見 ADR-0020 + autofluxdep/README）。
