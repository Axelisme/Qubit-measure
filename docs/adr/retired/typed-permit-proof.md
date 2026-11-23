---
status: superseded
---

# Domain guard 的 typed Permit proof

> 摘自 [ADR-0001](../0001-permit-lease-typed-guard.md#現況)。以下保留歷史原文；現行規則見 [ADR-0066〈問題與決策〉](../0066-operation-lifecycle.md#問題與決策)。

## 現況

本篇的 typed Permit proof 設計由 [[0066]] 的 checks 與 RunInputs 契約取代。
`SavePermit`、`LoadPermit`、`AnalyzePermit`、`WritebackPermit` 已刪除，
`RunPermit` 改為 `RunInputs`，沒有相容別名。

Qt View 與 remote RPC 共用 application controls。Controls 在同次 owner-thread
command 先呼叫 `GuardService.check_*`，再把 tab_id 交給 operation service。
Checks 成功返回 None，失敗 raise `GuardError`，保留原 reason_code。
Run admission 由 `capture_run_inputs` 接受指定 cfg revision 並捕捉已觀察的輸入，
RunService 使用它們執行，不以資料型別證明 live state 不變。
