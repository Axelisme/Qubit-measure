# `gui.app.measure.writeback`

**Last updated:** 2026-10-22

**讀者：** framework-dev。寫回提案、review widget 與逐項 Apply 契約。

本模組擁有 parameter、module 與 sample 的 proposal、draft、service、control 與 Qt review widget。

## Initial selection and notes

Adapter 可省略不合適的候選，或以 proposal 的 `selected=False` 保留候選但不預選。
`selected` 預設為 True，`note` 預設為 None。Adapter 自行判斷實驗品質，writeback 不比較品質閾值。
Service 將 proposal 的初選與 note 複製到 draft。GUI 在目的地下面顯示 note，RPC preview 也回傳同一段文字。
人工可改勾選並 Apply。Edit、retarget 與 Apply 保留 note，不重新判斷品質；人工勾選不表示品質合格。

## Sample 列

Sample 的目標欄顯示 `<元件名> · sample`。未 Apply 時保留共用的 `*` 標記，成功後移除。
Current 欄顯示 `entry sample`。Proposed 欄與 tooltip 逐行列出全部 `key=value`，保留字串與巢狀 JSON 的完整內容。
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

資源層對完整 serialization tree 驗證一次 YAML 表示。
Library 的完整 provenance 驗證一次讀取 ledger，並檢查每個被引用的事件與附件。
DocumentStore 每次仍讀取磁碟 bytes，相同內容只重用已解析的 YAML 節點。
每次 Apply 都執行模型、清理政策與外部來源驗證，重新驗證 ledger 與附件。
各列各自提交到固定目的地，錯誤分類與 applied 狀態由逐列結果決定。
Apply 同步執行。

## Captured acceptance

每個 draft 的 revision 在建立、selection、target、parameter value 或 module Form 改動時遞增。
Module Form 的 write／command 經同一 subscription 升號。替換 draft 不重用舊 revision。
`accept_drafts` 在同一次 owner 呼叫核對全部 selected drafts 的 revision 與 IDs，再依輸入群組順序寫入。
Writeback control 固定 Primary 在 Post 前，並在提交前檢查 permission、idle 與 active draft。
任一 revision 過期就拒絕整批，錯誤列出變動群組、IDs 與前後 revision，不把 draft 還原成舊值。
提交中的失敗回 confirmed IDs 成功前綴及診斷，不 retry／rollback。
`apply_draft` 與單 pane `write_draft` 保留原有 raise 與 edit/write 契約。
