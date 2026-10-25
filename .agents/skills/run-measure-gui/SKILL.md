---
name: run-measure-gui
description: GUI 操作。透過 measure-gui MCP 執行量測、保存或 writeback、核對現況，或處理 stale、中斷與使用者接手時使用。
---

# Run measure GUI

**讀者：** agent。透過 measure-gui MCP 核對 GUI 現況、執行授權內的操作並確認結果。

任務：接收或恢復量測任務、選擇下一輪、裁定結果或沉澱經驗時，先讀 [Measure task](../measure-task/SKILL.md)。任務結構與持久化時機由該 skill 管理；本 skill 只處理 GUI／MCP 操作。

## 1. 核對 GUI 現況

開始或恢復操作時，先讀 `measure-gui` MCP server instructions，取得目前 GUI 狀態與可用工具。核對實際 project、context、裝置及執行中操作。以 GUI 現況校正任務文件中的上次觀察。

讀寫目前 point 的參數、library 或 aliases 時，先用 `status()` 或 `rpc_call(method="entry.info")` 取得頂層 `files`，再依各筆絕對 path 直接讀檔。Revision 是已安裝版本，磁碟可能仍有未驗證候選，不能只靠 status 推斷檔案內容。完整流程見 [Point 檔案讀寫](../../../lib/zcu_tools/mcp/measure/README.md#point-檔案讀寫)。

硬體：有硬體副作用前，確認相關接線、裝置模式、適用限制及當次授權。沿用仍有效的已確認資訊；關鍵條件缺失時先問使用者，只推進不依賴該缺口的操作。

完成條件：已確認所連環境、操作對象與 live operation；下一個操作需要的硬體條件與授權已核對，或已暫停等待必要輸入。

## 2. 選擇 MCP 入口與 tab

契約：工具選擇、參數、回覆欄位、等待、控制與保存語意，以 server instructions 和工具描述為準。進入一種實驗時，透過 MCP 公開入口讀該 adapter guide。設定 client 等待期限，使其能容納本次呼叫。

Tab：新建或重用工作頁、再次呼叫同 adapter 的 recipe、保存不完整或清理 tab 時，讀 [Tab 生命週期](references/tab-strategy.md)。

Guard：依操作讀相關完整觀察。Tab 用 `tab.snapshot`，SoC 用 `soc.info(include_cfg=true)`，裝置用 `device.snapshot`。Cfg 寫入與 Run 前另讀 `tab.get_cfg` 的目前 input、values 與 errors。Guard 以這些完整觀察為依據。

Point 檔案編輯使用 `files` 的 path 與已安裝 revision，不以 `context.snapshot` 代替讀檔。修改前確認自己是 holder，保留已安裝 revision，直接編輯後呼叫 `rpc_call(method="context.reload", params={"path": path, "expected_revision": revision})`。`expected_revision` 必須明填，只有預期 aliases 缺檔時可用 null。檔案系統不檢查 holder，不能因為可以寫檔就省略控制權確認。依上述 Point 檔案讀寫契約執行，不另用特化讀寫工具。

沿用值：來源記錄含 `cloned_from`、準備用於量測或校準時，依 [沿用值驗證](references/tab-strategy.md#沿用值驗證) 核對來源與適用性。

完成條件：入口與 schema 已核對，tab 選擇與必要 guard 已完成；沿用值已有驗證證據，或依賴它的操作已暫停。

## 3. 執行並核對保存

按目前工具契約執行讀取與變更。步驟已確定、無需中途判讀時批次操作；下一步取決於新證據時停在判斷點。同一 live resource 的多個執行者須先安排操作順序。

保存或 writeback 前核對目標，回覆後逐項核對實際產物與寫入結果。依 Tab 生命週期處理保存缺口。把可用的操作識別、結果與 artifact 位置交回 Measure task 的當前 round，供後續裁定使用。

直接編輯 point 檔案後，核對 reload 回覆的 `document`、`path` 與 `revision`，或確認 watcher 已安裝同一候選。單有檔案寫入不代表 owner 已驗證、安裝或發布。將檔案路徑與已安裝 revision 證據交回當前 round。此流程不改既有 recipe／writeback 保存方式，也不改既有 Tab／Run 的固定 capture。Refresh／Reset 的效果見 Point 檔案讀寫契約。

完成條件：本次操作有可追溯的結果與實際保存位置；直接編輯的 point 候選已確認安裝，或已記錄失敗／未知、保存缺口及待確認事項。

## 4. 處理逾時、斷線與 stale

遇到逾時、斷線、stale 或缺少完成回覆時，先讀現況。分辨操作已完成、仍在執行、失敗或狀態未知，再決定後續動作。新 session 重新確認可用識別，舊 handle 只作上次觀察的線索。

Reload 驗證失敗時，owner 保留已安裝記憶體，磁碟仍是外部候選，先依回覆的路徑與原因修正。Revision mismatch 時先重讀 `files` 與檔案內容，watcher 可能已先安裝合法候選，核對是否為同一版本後再決定。非 holder 不重送。不要用舊 revision 自動覆寫或反覆 reload。

只有確認尚未執行或已失敗，且重試仍在授權內時，才決定是否重送。執行狀態未知時，記錄已核對的範圍與不確定性並求助，保持相關操作暫停。

完成條件：每個未結操作都有目前狀態與後續處理；未知狀態已交回任務紀錄，並指出需補哪個確認。

## 5. 處理使用者中斷與接手

中斷：使用者中斷量測時，僅保存必要狀態並等待使用者說明，後續自主執行與分析維持暫停。取得新的接續指示後，再依 Measure task 恢復流程核對現況。

接手：使用者接手 GUI 或提供修正後，重新核對設定、執行中操作與原假設的適用性。以現況與新指示選擇下一動作，保留使用者的修改。

求助：缺失硬體資訊、操作風險或授權需要使用者決定時，說明觀察、已試操作、選項與待答問題。使用者離線時按有效 policy 處理，未獲授權的操作保持暫停。

通知與背景執行只使用 runtime 已提供的能力。安排跨夜工作前確認程序存續與通知能力。

完成條件：停止位置、未結操作與接續條件已交回任務紀錄；暫停中的操作等待所需指示，接手後的下一動作以目前 GUI 狀態為依據。
