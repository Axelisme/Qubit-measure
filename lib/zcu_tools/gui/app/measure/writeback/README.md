# `gui.app.measure.writeback`

**Last updated:** 2026-10-20，sample 列呈現與 Apply 資源層驗證

本模組擁有 parameter、module 與 sample 的 proposal、draft、service、control 與 Qt review widget。

## Sample 列

Sample 的目標欄顯示 `<元件名> · sample`。未 Apply 時保留共用的 `*` 標記，成功後移除。
Current 欄顯示 `entry sample`。Proposed 欄與 tooltip 逐行列出全部 `key=value`，不將字串或巢狀 JSON 縮成摘要。
每個 key 與值的前綴可見。長值可能被欄寬裁切，tooltip 與 Copy 仍保留完整內容。
`dev_value` 後面附上 `dev_unit`，並保留獨立的 `dev_unit` 欄位。其他欄位與 `run_id` 依 proposal 呈現，不轉換單位。
Copy 將完整 JSON 放入剪貼簿。Sample 沒有 Edit 動作，也不開啟編輯對話框。

Service 從固定 Run／Load 來源補入 sample 座標。Widget 只呈現既有 proposal，不解析裝置 alias，也不改變資料型別或持久化流程。

## Apply

`WritebackService` 從 draft 的固定 Run／Load capture 解析選定提案與 module Form，
再交給 context owner。全部提案通過輸入驗證後，共用 executor 先追加一筆 accepted
ledger event，再依提案順序提交 Parameter、Module 或 Sample。
Parameter 與 Module 各列保留自己的單檔交易，Sample 各列追加樣本。
第一個失敗會停止後項，保留 accepted event 與成功前綴，並標記已成功的 draft 列。
Ledger append 失敗時不提交任何列。Apply 不提供跨列 rollback 或自動 retry。

資源層只對完整 serialization tree 驗證 YAML 表示，不逐層重驗同一子樹。
Library 的完整 provenance 驗證一次讀取 ledger，並檢查每個被引用的事件與附件。
DocumentStore 每次仍讀取磁碟 bytes，相同內容只重用已解析的 YAML 節點。
模型、清理政策與外部來源驗證照常執行，不快取 ledger 或附件的驗證結果。
這些最佳化不合併逐列交易，也不改固定目的地、錯誤分類或 applied 狀態。
Apply 維持同步執行，未增加 spinner 或背景工作。
