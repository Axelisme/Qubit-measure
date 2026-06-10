# 0003 — 統一 async-task cancel（Registry 持 stop_event）+ Qt-free ShutdownCoordinator 輪詢關閉

**狀態：** accepted（119a gui2:5ed92fd2 / 119b gui2:3df4890a，live-verified）。
**關聯：** 擴充 [[0002]]（operation handle / `_OperationRegistry`）；承 [[0002]]（off-main blocking handler / 主線不能阻塞）；分層採 [[0005]]（Hexagonal，Qt 經 driven adapter）；起因於 Phase 118 的 `app.shutdown`（用戶質疑「為何用 kill / 為何用屬性 flag」）。**§一「生命週期綁死」被 [[0019]] 取代**（exclusion/handle 改 opt-in facets；cancel 三動詞與 ShutdownCoordinator 不變，token 的 handle 來源改 Handle leaf）。

## 背景

GUI 關閉（user 點窗 close / agent `app.shutdown`）時，現況有兩個問題：

1. **closeEvent 只處理 device setup，不管 run/connect**——關窗時若有 run 在跑，直接 persist + 關，worker 被強行打斷（盲區）。
2. **「等 device setup 停」靠屬性 flag + 訂閱廣播事件拼湊**：`closeEvent` 設 `_shutdown_waiting_for_device_setup = True` + cancel + `a0.ignore()`，再由 `_on_bus_device_setup_finished` 讀 flag 續關。這是「用共享狀態在『發起取消』與『取消完成』之間傳遞」——用戶要求「用傳遞來共享，不用共享來傳遞」。

根因：**取消的「完成通知」分裂成兩套**。off-main 路徑（`operation.await` RPC handler 在 IO worker thread）用 `_OperationRegistry` 的 per-token Event 阻塞 await；但**主線（closeEvent）不能阻塞**（[[0002]]：卡死 event loop → 死鎖），只能退回 flag + 廣播事件。而且 **cancel 本身散在各 worker by-name**（`cancel_device_operation(name)` / `cancel_run`），`_OperationRegistry` 有 await/poll 卻**沒有 cancel**——async 三動詞不統一。

## 決策

**一、`_OperationRegistry` 成為所有 async-task 工具的統一入口：cancel / poll / await 三動詞齊備。**

- `register`（= 現有 `create`）時把 worker 的 **`stop_event`（純數據 handle，非 callback）** 一起交給 Registry。
- 新增 `cancel(token)` = `set` 該 token 的 stop_event（**異步通知式**：只發請求即返回，不等待）；`cancel_all() → list[token]` 遍歷所有 active token cancel。
- 既有 `await_outcome`（off-main 阻塞）/ `poll`（非阻塞）不變。
- **生命週期仍綁死**（一個 operation 同時是互斥的 + 可 await 的，同一 token 串 `_Exclusion` + `_Registry`，不拆成可選疊加）——只把 cancel 收進 Registry，不動互斥/handle 的綁定。〔**已被 [[0019]] 取代**：analyze/interactive 是具體的 handle-not-exclusion 場景，改 opt-in facets。〕

**二、cancel 是純信號傳遞，worker 自己把 stop 翻成 cancelled outcome（修 RunWorker 對齊 DeviceWorker）。**

`DeviceWorker._emit_outcome` 已有 `elif self._stop_event.is_set(): cancelled.emit`；`RunWorker` 沒有，故 `RunService.cancel_run` 在**外部**用 `_cancel_requested_tabs.add()` 記「用戶取消」。這是 RunWorker 的**缺陷**（不是 cancel 本質要副作用）。修法：RunWorker 加 `elif self._stop_event.is_set(): run_cancelled`，**刪 `_cancel_requested_tabs`**。如此 Registry 持純 stop_event（cancel 端零副作用、零 callback IoC）。

**三、Qt-free `ShutdownCoordinator` + 週期 poll 輪詢取代屬性 flag。**

- `ShutdownCoordinator`（純邏輯，**零 qtpy import**，可單測）：`begin()` = `gate.cancel_all()`（拿全部 token）；`tick() → WAITING/SETTLED/TIMED_OUT`（每 tick 對所有 token `gate.poll` + 比 deadline）。
- **QTimer 包在 driven adapter** `QtShutdownDriver`（`adapters/qt_shutdown_driver.py`，非 services QObject——它是 driven adapter）：`timer.timeout → coordinator.tick()`，按回傳 state 觸發「真關」(`_perform_close`)。**超時不需顯式 force-kill**：window close 後 `app.exec()` 返回、`sys.exit` 退進程，阻塞的 connect QThread 隨進程死；故 SETTLED/TIMED_OUT 都只是調 `on_closed`。
- `Controller`（Qt-free façade）暴露 `begin_shutdown(on_closed)`（懶建 driver 後純委派）+ `active_operation_count()`；`MainWindow` closeEvent / request_shutdown 調它、傳 `_perform_close` 當 `on_closed`。**消除 `_shutdown_waiting_for_device_setup` + `DEVICE_SETUP_FINISHED` 訂閱**——「在等哪些 token」是 coordinator 自己的局部執行上下文，非跨對象共享 flag。user close 保留確認框（只 user 端；文案改「取消 N 個進行中的操作」涵蓋 run+device+connect）；`_perform_close` 的二次 closeEvent 由 window 自身的 `_closing` 局部 guard 放行。

## 考慮過的替代

- **cancel 同步等待**（調用後阻塞到 operation 停）：主線會死鎖（[[0002]]），且 connect 不可中斷會永久阻塞。否決 → cancel 異步通知式。
- **主線監聽 worker 的 Qt signal**（事件驅動續關）：即時，但要屬性 flag 記「在等誰」，且各 operation 各自的 signal 不統一。否決 → 統一用 `gate.poll(token)` 週期輪詢（換來不即時，但關閉非高頻可接受）。
- **Registry 持 cancel callable**（封裝各 worker cancel 差異）：callback = IoC，被架構警惕，且 run 的「標記 cancelled」副作用仍要某處裝。否決 → 持純 stop_event + 修 RunWorker 自判（更純、符合「用傳遞不用共享」）。
- **把 exclusion / handle 拆成可選疊加**（長任務可只要 handle 不要互斥）：用戶接受生命週期綁死，無具體「不互斥長任務」場景，過度設計。否決 → 只統一 cancel 入口，不動綁定。〔**後於 [[0019]] 接受**：analyze（FIT+INTERACTIVE）正是該場景，前提失效。〕
- **抽 ClockPort/TickPort**：tick 只有 QTimer 一種實作（不像 ProgressTransport 有 Direct/Qt 兩種），抽 port 過度。否決 → coordinator `tick()` 回 state 已可單測，QTimer 接線留 QObject service。

## 後果

- **connect 不保證即停**：connect 是阻塞網絡調用、無 stop 點，`cancel(token)` 對它無效（cancel 是「請求」）。closeEvent poll 等不到它 settle → **超時強關**（force-kill）。可接受：關閉時 connect 被打斷無傷（沒有要 persist 的狀態）。
- **項目第一個週期計時器**：現況全是 `QTimer.singleShot(0,…)`（單次）。ShutdownCoordinator 的輪詢是首個 `setInterval` 週期 timer，是新模式。
- **closeEvent 行為擴大**：從「只等 device setup」變「cancel + 等所有 operation（run/device/connect）」——補了現況關窗不管 run 的盲區。
- **WIRE 不變**：`app.shutdown` RPC 已在 Phase 118 加（WIRE 13）；本 ADR 只改 GUI 內部關閉編排 + Registry cancel，不動 wire 契約。
