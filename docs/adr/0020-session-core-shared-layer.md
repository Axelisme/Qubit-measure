---
status: accepted
---

# session-core 共用層：measure + autofluxdep 共享量測 session（gui/session/）

**狀態：** accepted（**已落地**：S1–S5 抽取 + autofluxdep 完整複用 + real-acquire RUN path；本檔現在式描述生效設計）。
**關聯：** session service 角色/依賴依 [[0004]]/[[0005]]（App Service / Aggregate Root / 三問依賴）；resource versioning + async operation handle 依 [[0002]]；context 單一寫入 + CfgSchema lowering 依 [[0006]]；exclusion gate / handle / background 三 facet 依 [[0019]]（本檔把那套 leaf 提到共用層 + 定 app-local vs shared 邊界）；event_bus / shared transport 依 [[0014]]；autofluxdep orchestrator 設計依 [[0018]]；worker 不畫圖故 [[0017]] marshal 不適用。

## 脈絡

measure-gui 的「量測 session core」（context 系統 MetaDict/ModuleLibrary + SoC 連線 + 多 device + setup/device/inspect/predictor dialog）原本全埋在 `gui/app/main/`。第二個 measurement-session app（autofluxdep：拿某 flux 量到的 base context，衍生其他 flux 的掃描參數，自動跑一段 flux sweep）要**整套複用**這核心，否則就是大量重抄 + 長期漂移。

## 決定

**抽 `gui/session/` 共用層**，對標 `gui/remote`/`gui/plotting`，圈出「量測 session core」：值型別（ExpContext/Soc*）、SessionState slice、session events、connection/context/device/startup 服務 + `build_session_services`、共用 dialog（setup/device/predictor/inspect_base）、`SessionControllerPort`。session 模組**永不**反向 import `gui.app.*`。

**app-local vs shared 邊界**：
- **app 自持**（policy / Qt facet 帶 app 味、各進程獨立無共享需求）：`OperationGate`（衝突 policy + app 自己的 RUN/sweep kind）與各 app 的 concrete `BackgroundRunner` instance（owner 直接呼 `quiesce()`）。
- **promote 成共用**（純機制 / app-agnostic）：`OperationHandles`、`ProgressService`、`IOManager`、`QtProgressTransport`、`BackgroundRunner` class / `BackgroundExecutor` port。
- session service 只依賴**窄 port**（`ExclusionGate`/`BackgroundExecutor`/`ProgressHub`/`ProjectIOPort`），app 經 `build_session_services` 注入具體。

**SessionControllerPort = 共用 dialog 的 Controller 契約**：dialog 不碰 concrete `gui.app.*` Controller，只依這 Protocol；各 app 的 Controller **結構上**實作它（pyright 在各自 dialog call site 驗 conformance）。回傳宣告對 `BaseEventBus`，故 app 的 richer EventBus covariant 滿足。

**app 接法**：`State(SessionState)` 子類化（零 ripple 繼承）；Controller `__init__` 組 `build_session_services` + 注入 app-local infra；run 讀 `exp_context`（SSOT），無 app 自己的 SetupResources/setup()。

**autofluxdep RUN lifecycle**：autofluxdep 的 flux-sweep RUN 使用共用 `OperationRunner` 生命週期機制，而不是 app-local `_RunWorker` / `_running` / `_stop` 平行實作。RUN 的 domain loop、workflow state 與 per-node result summary 仍屬 autofluxdep；operation facet（exclusion、handle、progress、cancel、terminal outcome）走 shared session operation interface。Stop 是帶 reason 的協作停止，保留已完成 partial results。

**autofluxdep RUN path（real acquire）**：每 node 的 `Builder.make_cfg`（**在 `produce` 跑**，snapshot 在手——決策 A + D1）lower 當前 context 的 ml/md + drive 設定頭 params → 真 cfg（經 `ml.make_cfg`），再把此 flux 點寫入 `cfg.dev[flux_device]`（by device name）→ `setup_devices` → program `.acquire`（round hook + cooperative stop + SNR early-stop）→ fit / Patch。offline 測試走 flux-aware MockSoc；無 synthetic fallback，未配置 context 由 `make_cfg` Fast Fail，orchestrator 轉成 `RunFailedPayload`。

## 理由 / 取捨

- **str-keyed ExclusionGate**：session `OperationKind`（soc/device）+ app 自己的 kind（measure/autofluxdep 各 `RUN`）用 wire 字串比較，兩 enum 同一 gate 不必合併。
- **adapter 型別 principled re-export**：`ExpContext`/`Soc*` 是 `ExpAdapterProtocol` 契約講的詞彙，從 session re-export 把 blast radius 圈在 `gui/` 內（是契約 API，非 CLAUDE.md 禁的 legacy/compat shim）。
- **ml-edit measure-only**：autofluxdep 消費 base context 的 ml（不自造），故共用 inspect = md 編輯 + ml 檢視；ml create/modify（拖整套 CfgEditor）留 measure `InspectDialog` 子類擴充。
- **predictor 共用 vs 自適應分層**：`exp_context.predictor` 存 raw `FluxoniumPredictor`（ConnectionService/共用 PredictorDialog 載）；autofluxdep `_build_tools` 每 run wrap 成自適應 `FluxoniumPredictorAdapter`（None→`SimplePredictor`）——共用「載入」，autofluxdep 自有「自適應 use + calibrate 閉環」。
- **headless 測試 Qt 化**：autofluxdep run 經 async ConnectionService（mock connect singleShot settle），測試用 package-level `qapp` autouse + `connect_mock`（QEventLoop pump），與 measure 一致；不保留同步 setup() 旁路。
