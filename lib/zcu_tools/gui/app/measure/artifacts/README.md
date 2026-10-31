# Measure artifacts

**Last updated:** 2026-11-01

**讀者：** framework-dev、agent。查找量測輸出預覽、保存與狀態投影的 owner。

`SaveService.plan_layout(tab, record, saved_at)` 為預覽與 Save 計畫同一組輸出位置。
它使用注入的 Result／Database roots、歷史 metadata 的 entry／point，以及 adapter 的 stem。
`OutputPlan.binding` 是 `bind_run_outputs` 的結果，`OutputPlan.layout` 由該 binding 產生。
預覽只用 layout，不讀 `run.storage`、不建立目錄，也不改變保存狀態。
Save 使用 layout 寫檔，並以 binding 準備 ledger，不重新綁定目的地。

Composition root 將 planner 注入 Run、Load、Analyze 與 PostAnalyze。
正常 Load 完成後立即顯示預覽，分析後也顯示。
Metadata 缺 native schema 必填 key 時沒有預覽，Save 以 storage_unavailable 拒絕。
`pane_commit` 只隔離歷史欄位讀取的 `MetadataFieldError`，讓 Load 保留原始壞內容。
其他計畫錯誤照常傳出。Save 在任何輸出前完整驗證 native schema 與 experiment tag，寫檔保留 metadata 的全部 key。

`ArtifactTracker` 保存 declarations 與最新 report 的輸出狀態。
預覽是 reserved outputs，不表示已保存。
Save report 發布實際路徑與每個輸出的成功或失敗，部分失敗不回滾已寫檔案。
`ArtifactSaveCenter` 只投影這些狀態與送出保存意圖，不決定路徑或編排 writer。
保存與寫回保持獨立，由 `measure-artifacts-no-writeback` 與
`measure-writeback-no-artifacts` import contracts 檢查。

保存權責見 [ADR-0063](../../../../../../docs/adr/0063-persistence-ownership.md#run-輸出位置與-writer-權責)。
