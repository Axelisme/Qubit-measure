---
name: run-measure-gui
description: GUI 操作。透過 measure-gui MCP 連線或啟動 GUI、執行量測、保存或 writeback、追蹤 execution，或處理逾時、stale、中斷與使用者接手時使用。
---

# Run measure GUI

**讀者：** agent。透過 measure-gui MCP 核對 GUI 現況、執行授權內的操作並確認結果。

任務：接收或恢復量測任務、選擇下一輪、裁定結果或沉澱經驗時，先讀 [Measure task](../measure-task/SKILL.md)。任務結構與持久化時機由該 skill 管理；本 skill 只處理 GUI／MCP 操作。

## 1. 核對 GUI 現況

開始或恢復操作時，先讀 `measure-gui` MCP server instructions，再用 `connect()` 連既有 GUI。核對 `connect.status` 的實際 project、context、裝置、控制身份及執行中操作。以 GUI 現況校正任務文件中的上次觀察。

只有當次授權涵蓋啟動 GUI 時，才用 `launch="if_missing"` 或 `launch="new"`。連線本身不連硬體，但 startup script 可能連硬體，啟動前審閱選用腳本與授權。需要換 script 或 project_root 時先關舊 GUI，再 launch new；attach 不重跑 startup，也拒絕顯式 script／project_root。選擇預設、可信 `startup(ctx)` 或 clean，以及隔離資料目錄前，讀 [Startup scripts](../../../lib/zcu_tools/mcp/measure/README.md#startup-scripts)。離線 FTF 使用 `zcu_lab/simulation_envs/FTF/startup.py`，先依該環境 README 準備 YAML。Token 視為秘密；認證或 wire 不相容時先解決連線，不送動作。

讀寫目前 point 的參數、library 或 aliases 前，先讀 [Point 檔案讀寫](../../../lib/zcu_tools/mcp/measure/README.md#point-檔案讀寫)，依該契約取得檔案與已安裝版本。

硬體：有硬體副作用前，確認相關接線、裝置模式、適用限制及當次授權。沿用仍有效的已確認資訊；關鍵條件缺失時先問使用者，只推進不依賴該缺口的操作。

完成條件：已確認所連環境、操作對象與 live operation；下一個操作需要的硬體條件與授權已核對，或已暫停等待必要輸入。

## 2. 選擇 MCP 入口與 tab

契約：工具選擇與共用 execution 生命週期以 server instructions 為準，參數與回覆以工具描述及 [Measure MCP 契約](../../../lib/zcu_tools/mcp/measure/README.md) 為準。日常量測先找符合目標的 recipe，讀 schema 與 `recipe_guide(adapter)`。已有資料先用 `tab_analyze`，互動分析用 `tab_interact`，不為分析重做量測。Client 延後載入 tools 時先搜尋 recipe 或共用分析／控制工具。其餘 setup、cfg、保存、寫回與排查先 `rpc_list`、`rpc_describe`，再 `rpc_call`；這些入口不是權限模式。

Recipe、answer 與 device_setup 最多等 300 秒，client deadline 設為大於 300 秒並預留傳輸及回覆時間。Stdio server 同步處理，同連線的下一個請求不保證在等待期間得到服務。需要較短等待或非同步裝置 setup 時，依 [Setup 工具](../../../lib/zcu_tools/mcp/measure/README.md#setup-工具) 用完整 Info cfg 與原生 RPC，不把 partial patch 直接當完整 cfg。

Tab：新建或重用工作頁、再次呼叫同 adapter 的 recipe、保存不完整或清理 tab 時，讀 [Tab 生命週期](references/tab-strategy.md)。

Guard：依操作讀相關完整觀察。Tab 用 `tab.snapshot`，SoC 用 `soc.info(include_cfg=true)`，裝置用 `device.snapshot`。Cfg 寫入與 Run 前另讀 `tab.get_cfg` 的目前 input、values 與 errors。Guard 以這些完整觀察為依據。

Point 檔案編輯前確認自己是 holder，再依 [Point 檔案讀寫](../../../lib/zcu_tools/mcp/measure/README.md#point-檔案讀寫) 讀取、編輯與 reload。檔案系統不檢查 GUI 控制權，不能因為可以寫檔就省略確認。

沿用值：來源記錄含 `cloned_from`、準備用於量測或校準時，依 [沿用值驗證](references/tab-strategy.md#沿用值驗證) 核對來源與適用性。

完成條件：入口與 schema 已核對，tab 選擇與必要 guard 已完成；沿用值已有驗證證據，或依賴它的操作已暫停。

## 3. 執行並核對保存

按目前工具契約執行讀取與變更。步驟已確定、無需中途判讀時批次操作；下一步取決於新證據時停在判斷點。同一 live resource 的多個執行者須先安排操作順序。

保存或 writeback 前核對目標，回覆後逐項核對實際產物與寫入結果。依 Tab 生命週期處理保存缺口。把 execution、op、run_id、結果與 artifact 位置交回 Measure task 的當前 round，供後續裁定使用。保留已知 execution ID；全域 status 只列活躍項目與最近 10 筆終態 recipe，不是完整歷史。用 `status(execution, detail="full")` 看捕捉的 native 證據；等待完整接續用 `wait(execution)`，不是只等 GUI operation 的 `wait(op)`。Save 要看各 output 的 status／error，run_id 或預留路徑不證明已存檔。重分析會替換同 pane 的圖，需要比較時先保存舊圖。

直接編輯 point 檔案後，依 [Point 檔案讀寫](../../../lib/zcu_tools/mcp/measure/README.md#point-檔案讀寫) 確認候選已安裝，將檔案路徑與已安裝 revision 證據交回當前 round。

完成條件：本次操作有可追溯的結果與實際保存位置；直接編輯的 point 候選已確認安裝，或已記錄失敗／未知、保存缺口及待確認事項。

## 4. 處理逾時、斷線與 stale

遇到逾時、斷線、busy、stale 或缺少完成回覆時，先讀 status、相關 snapshot 與確認保存的檔案。Client timeout 不取消操作，不自動重跑 recipe 或 mutation。分辨操作已完成、仍在執行、失敗或狀態未知，再決定後續動作。Operation handle 屬於目前 GUI 連線 generation；execution ID 只屬於目前 MCP server session。GUI 重連後從 status 重新發現 handle，不沿用舊 op。新 MCP session 的舊 execution ID 只作上次觀察的線索。

Post `result_read` 失敗時，先讀 `tab.snapshot`，比對 `post_analysis_state.source_operation_id` 與回覆的 `recovery.params.operation_id`。不同則回報 superseded，不拿新結果代替舊結果。相同才用 `rpc_call(method=recovery.rpc, params=recovery.params)` 讀回。該 route 也會拒絕比對後發生的 replacement race，不重跑分析修復讀取。

寫回回覆為 unknown 時，先查回覆中 `recovery` 的 `tab.writeback_receipt`，核對 submission_id、已確認項目與固定目的地。逾時不等於寫入失敗，不據此重送。取消只代表已請求停止，用 wait/status 確認 cleanup 終局。結束 MCP session 只會斷線，不關閉 GUI，也不保證硬體停止。

Reload 驗證失敗、revision mismatch 或 watcher 先安裝候選時，依 [Point 檔案讀寫](../../../lib/zcu_tools/mcp/measure/README.md#point-檔案讀寫) 判斷下一步；非 holder 不重送。

只有確認尚未執行或已失敗，且重試仍在授權內時，才決定是否重送。執行狀態未知時，記錄已核對的範圍與不確定性並求助，保持相關操作暫停。

完成條件：每個未結操作都有目前狀態與後續處理；未知狀態已交回任務紀錄，並指出需補哪個確認。

## 5. 處理使用者中斷與接手

中斷：使用者中斷量測時，僅保存必要狀態並等待使用者說明，後續自主執行與分析維持暫停。取得新的接續指示後，再依 Measure task 恢復流程核對現況。

接手：使用者接手 GUI 或提供修正後，重新核對設定、執行中操作與原假設的適用性。以現況與新指示選擇下一動作，保留使用者的修改。

求助：缺失硬體資訊、操作風險或授權需要使用者決定時，說明觀察、已試操作、選項與待答問題。使用者離線時按有效 policy 處理，未獲授權的操作保持暫停。

通知與背景執行只使用 runtime 已提供的能力。安排跨夜工作前確認程序存續與通知能力。

完成條件：停止位置、未結操作與接續條件已交回任務紀錄；暫停中的操作等待所需指示，接手後的下一動作以目前 GUI 狀態為依據。
