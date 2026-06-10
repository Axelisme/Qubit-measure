# 0015 — PersistenceCaretaker（Memento + Caretaker）+ 單檔 app-state + 關閉才寫

**狀態：** accepted（Phase 126 gui2，live-verified）。
**關聯：** 採 [[0005]]（Hexagonal：Caretaker = Driven Adapter；Originator = Controller façade）；用 [[0004]]（狀態進 State 兩軸：startup 偏好進 `State.startup_prefs`）；承 [[0008]]（tabs cfg raw↔live codec 留 owner 服務）；起因於用戶質疑「persist 為何 runtime 一直寫盤 / 為何兩個檔 / PersistService 為何叫 Service」。

## 背景

Phase 126 前，GUI 持久化散在 `StartupService`/`WorkspaceService`/兩個 `*PersistenceService` class，且寫入時機不一致：

- **startup 設定**（project/connection/devices/left_panel）**runtime 事件驅動即寫**：`StartupService._on_device_changed`（訂 `DEVICE_CHANGED`，每次裝置變動比對+寫，靠 diff-guard 過濾）、`save_left_panel_width`（splitter 每拖一次寫一次磁碟）、`apply_project`/`remember_connection`（套用即寫）。
- **session（tabs）** 才是關閉才寫。
- 存兩個歷史殘留檔（`startup_v2.json` / `tab_session_v1.json`），各有版本號與各自的手刻 load 驗證。

問題：持久化邏輯橫跨幾乎整個系統，但若讓一個 persistence 物件去 reach into 每個 aggregate 抓欄位，它會依賴全世界（god object）；而 `_on_device_changed` 把持久化掛在高頻 event 上、靠 diff-guard 補救，是把「導航/持久化決策」錯掛在「資料變動通知」上（見 [[feedback]] 的職責判準）。

## 決策

**Memento + Caretaker**（業界對「持久化需橫跨多 aggregate 又要保封裝」的標準解，DDD State-Snapshot）：不讓持久化物件 reach into aggregate，而是**每個 owner 自吐一個可序列化 memento，一個 Caretaker 只管收集/落盤/讀回時機**。

三層 capture/restore：

```
PersistenceCaretaker（Driven Adapter）  ← 只做單檔 disk I/O + load/flush 時機
   ↑ Controller 持有（窄 port PersistOriginatorPort，單向；run_app 建立後 attach_caretaker 注入）
Controller（單一 Originator）  ← capture 從 State+View 現組 memento / restore 派發
   ↕ 各 service capture/restore（序列化內化）
WorkspaceService.capture_session/apply_session、StartupService.capture_startup/restore_startup
```

- **Caretaker 不依賴整個 Controller**：只兩個窄方法 `capture_persisted_state()` / `restore_persisted_state(memento)`。不訂 event、不碰 UI、不碰 State、不懂 cfg。
- **單檔 `gui_state_v1.json` + 單一 `APP_STATE_VERSION`**：`AppPersistedState{version, startup, session}`，pydantic v2 frozen，`model_validate`/`model_dump` 取代手刻驗證。一次 atomic write（無「半套」狀態）；壞/舊版→default。
- **關閉才寫**：移除 `DEVICE_CHANGED` 訂閱 + diff-guard + 所有 runtime 即寫。`flush()` 在 close（`_perform_close`/`app.shutdown`），`restore_all()` 在 `run_app` 啟動。crash 丟失可接受（startup 設定多是方便性記憶）。
- **狀態進 State**：startup 偏好移入 `State.startup_prefs`（[[0004]] 兩軸：除 owner 外 setup dialog 也讀 → 進 State），`StartupService` 回歸無狀態。套用/連線時同步寫 prefs（寫入當下），capture 只讀。
- **codec 不搬家**：tabs 的 raw↔live `session_codec` 是 WorkspaceService capture/apply 的**內部實作**，Caretaker 只見不透明 `cfg_raw`。

## flush 觸發點開放擴充（不寫死 lifecycle-only）

`flush()` 是**觸發無關**的通用入口：每次呼叫重新 capture 落盤。目前唯一觸發是 lifecycle，但未來可加**定時 / 按鈕 / 事件 / RPC** 觸發而 Caretaker 不改——只需在新觸發點呼 `ctrl.persist_all()`。移除 `session.persist`/`session.restore` RPC 的理由是「**目前**語意=關閉才寫，agent 不需手動」，**非**「永遠不該有外部觸發」。**「關閉才寫」是當前唯一觸發，不是設計上限。**

## 考慮過的替代

- **PersistenceCaretaker 與 RemoteControlAdapter 同級（app 頂層、不在 Controller 內）**：否決。語意上 RemoteControlAdapter 是 Driving Adapter（驅動操作進系統），Caretaker 是 Driven（被 lifecycle 觸發做 I/O），方向相反；且 close 路徑只握 Controller，「同級」會把組裝細節洩漏到 View/adapter。改為 app 建立、Controller 經窄 port 持有。
- **保留兩個檔（獨立降級）**：否決。單一 memento↔單一檔語意一致、原子性更好、消掉「一檔壞一檔好」的不對稱降級；代價是一次性遷移丟失舊兩檔（符合不留 legacy）。
- **disk I/O 寫進 Controller（不要獨立 Caretaker）**：否決。Controller 是 Qt-free View façade，扛檔案 I/O 破壞角色、違反 [[0005]]「infra 只經 port」。
- **`StartupService._pending` in-memory 累積器**：否決。startup 每欄真相都已在 State/View（project/ip/port→State、devices→State、left_panel→View），累積器多餘；狀態進 State（[[0004]]）。
- **memento 用 dataclass / dict**：否決。dict 無 schema/驗證（違反強型別）；dataclass 強型別但反序列化要手刻——pydantic v2（cfg 層 `ConfigBase` 已用、已是依賴）兩者兼得，`model_validate` 消掉手刻驗證。

## 後果

- 刪 `SessionPersistenceService` + `StartupPersistenceService` + `StartupStorePort`/`SessionStorePort`；新增 `caretaker.py`/`persistence_types.py`/`session_codec.py`；`RestoreReport`/`RestoreIssue` 移 `ports.py`。
- Controller 多 `attach_caretaker`/`capture_persisted_state`/`restore_persisted_state`/`restore_all`/`persist_all`；移除 `restore_tabs_from_session`/`persist_tabs_session`/`restore_startup_settings`/`get/save_persisted_left_panel_width`。保留 `apply_startup_project`/`remember_startup_connection`/`get_persisted_startup`（寫改成更新 `startup_prefs`）。
- RemoteControlAdapter 移除 `session.persist`/`session.restore`（WIRE 16→17、GUI 13→14、MCP 18→19）。
- `_on_device_changed` 投影邏輯延到 `capture_startup`（flush 時即時從 `State.list_devices()` 投影 remember 集）。
