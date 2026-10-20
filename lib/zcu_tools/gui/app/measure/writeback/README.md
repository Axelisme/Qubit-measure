# `gui.app.measure.writeback`

**Last updated:** 2026-10-20，sample 列呈現

本模組擁有 parameter、module 與 sample 的 proposal、draft、service、control 與 Qt review widget。

## Sample 列

Sample 的目標欄顯示 `<元件名> · sample`。未 Apply 時保留共用的 `*` 標記，成功後移除。
Current 欄顯示 `entry sample`。Proposed 欄與 tooltip 逐行列出全部 `key=value`，不將字串或巢狀 JSON 縮成摘要。
每個 key 與值的前綴可見。長值可能被欄寬裁切，tooltip 與 Copy 仍保留完整內容。
`dev_value` 後面附上 `dev_unit`，並保留獨立的 `dev_unit` 欄位。其他欄位與 `run_id` 依 proposal 呈現，不轉換單位。
Copy 將完整 JSON 放入剪貼簿。Sample 沒有 Edit 動作，也不開啟編輯對話框。

Service 從固定 Run／Load 來源補入 sample 座標。Widget 只呈現既有 proposal，不解析裝置 alias，也不改變資料型別或持久化流程。
