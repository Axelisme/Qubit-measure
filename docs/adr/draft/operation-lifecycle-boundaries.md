---
status: draft
---

# Operation 關閉與 disconnect 的未落實邊界（draft）

**狀態：** draft。Measure close 的背景 drain／逾時留窗已併入 [[0066]]，其餘保留未落實目標。舊核准未保存裁決日期與使用者原話，不把它當成 accepted 契約；本票只處理文件，不新增機制。

## Shutdown 期限與背景工作

核准目標：關閉時停止接受新工作，請求可取消工作停止，等待必要的 domain 終局與 executor cleanup。期限到達只回報未完成；共用預設不自動強制退出。若 app 提供強制退出，必須明示未保存結果、未完成寫入和資源未清理的風險。

現況證據：`gui/session/services/shutdown.py` 的 tick 可以回報 TIMED_OUT；`gui/session/adapters/qt_shutdown_driver.py` 的 `_on_tick()` 對 WAITING 之外的狀態都呼叫完成 callback，包括 TIMED_OUT 和 tick 例外。這個共用 driver 尚未改為逾時留窗。

Measure 的 `Controller.close` 已先 stop_accepting 再 quiesce，逾時 raise TimeoutError。`MainWindow._perform_close` 成功 drain 才 teardown，逾時保留視窗並詢問再等或強關，見 [[0066]] 的 2026-10-15 裁決。這是 measure caller 的保護，不表示共用 driver 已完成。

`OperationHandles.cancel_all()` 只列 live handles。Measure data save 與 artifact batch save 已同步完成，沒有 Save handle 或 Save worker。無 handle 的背景工作仍不能只靠 live handles 清單判定 quiescence。Autofluxdep 的 app-local 強關選項不能當作其他工作的共用預設。

轉正條件：共用 timeout 不觸發預設 close callback；各 app 能回報未完成狀態並區分使用者明示的強制退出；需要等待的無 handle 背景工作（例如 auto-align）有 owner／executor 的可驗證 quiescence 契約，保存與 remote／資源 teardown 在必要工作清理後才執行。不以本 draft 指示當前實作立即退出或自動恢復。

## GUI device disconnect 與 factory owner

核准目標：已註冊 driver 的 disconnect 由 registry owner 協調；建立 ResourceManager 的 owner 在其管理的 sessions 成功斷線後才關閉 factory。部分失敗保留資源和錯誤供明確處置，disconnect 不隱含硬體 output policy。

現況證據：`device/manager.py` 的 manager close APIs 已提供 identity claim 和成功後 alias cleanup；`notebook_md/single_qubit.md` 先 `close_all_devices()` 再 `resource_manager.close()`。但 `gui/session/services/device.py::_disconnect()` 仍是 `get_device(name)`、直接 `device.close()`、`drop_device(name)`，不是 manager close；建構失敗路徑也直接清理未必已註冊的 device。該 service 的 driver factory 會新建 `pyvisa.ResourceManager()`，目前找不到對應由 GUI 關閉 factory 的保證。不可因此聲稱所有 GUI caller 都採用 registry-owned close，或把未註冊的失敗清理強迫走 manager。

轉正條件：釐清 GUI 對已註冊 driver 和每個 ResourceManager 的具體 ownership；核實或實作 GUI disconnect 對 aliases、並發 close、部分失敗的契約，再將 factory teardown 接到成功 disconnect 之後。保留建構失敗時尚未註冊 session 的獨立 cleanup；不新增 RF off、歸零或自動 retry。
