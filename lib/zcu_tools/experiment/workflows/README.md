# `experiment.workflows` — 具名 workflow 核心

**Last updated:** 2026-10-06，期 0 的 typed contracts 與顯示 seam

此 package 擁有具名 workflow 的通用執行契約。具體 flux 清單、校準、分析與 record 語意屬於使用者 workflow，不屬於核心。架構分工見 [workflows ADR 草稿](../../../../docs/adr/draft/workflows-engine.md)。

## 能力與資料

宿主以 `EnginePorts` 注入 clock、device、plot、progress 與 detached context。宿主持有硬體連線與 worker，核心不建立執行緒、不 connect／disconnect，不依賴 Qt 或 GUI。Progress factory 沿用現有 runtime bar 契約，不能把會再次查 ContextVar 的 make_pbar 當成 concrete factory。

`Next` 配對步驟摘要與下一步 state。`Done` 不新增點 record，`Aborted` 表達明確的業務終止。`Completed` 配對 experiment 返回時的 Run.cfg 副本與 Result，saver 決定單次檔案格式。`Failed` 表達可由 workflow 處理的 Schedule 失敗，不包含部分 Result。

Live 宣告只把本次 buffer 投影到 engine-owned artists，不進 state。`assemble_rows`／`assemble_scalars` 把稀疏結果組成 detached 顯示陣列，保留缺值與 filled mask，不插值。

## 驗證 owner

`tests/experiment/workflows/` 從公開契約驗證核心結果。既有 `tests/experiment/v2/runtime/` 仍擁有 Schedule、ProgramBuilder 與 buffer 行為，不在此複製底層 runtime 測試。
