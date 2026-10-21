---
status: accepted
---

# ADR-0001 — Domain guard 與 OperationLease 分離

**狀態：** accepted。

## 現況

本篇的 typed Permit proof 設計由 [[0066]] 的 checks 與 RunInputs 契約取代。
`SavePermit`、`LoadPermit`、`AnalyzePermit`、`WritebackPermit` 已刪除，
`RunPermit` 改為 `RunInputs`，沒有相容別名。

Qt View 與 remote RPC 共用 application controls。Controls 在同次 owner-thread
command 先呼叫 `GuardService.check_*`，再把 tab_id 交給 operation service。
Checks 成功返回 None，失敗 raise `GuardError`，保留原 reason_code。
Run admission 由 `capture_run_inputs` 接受指定 cfg revision 並捕捉已觀察的輸入，
RunService 使用它們執行，不以資料型別證明 live state 不變。

## 分界與取捨

四個只含 tab_id 的 frozen dataclass 任何 caller 都能建構，不能強制先檢查。
直接 checks 減少 carrier 與 caller 的轉接，domain 條件仍由 GuardService 統一定義。
Analyze／Post 的 capability 檢查尚未全部集中，現況見
[measure owner](../../lib/zcu_tools/gui/app/measure/README.md)。

Domain checks 不取得 lease，也不具有 release／cleanup 語意。
Busy 與硬體互斥由 owning service 與 OperationGate 在啟動時檢查。
Lease 必須在終局釋放；handle 只追蹤 operation，不能推導硬體所有權。
執行、取消與關閉責任以 [[0066]] 為準。
