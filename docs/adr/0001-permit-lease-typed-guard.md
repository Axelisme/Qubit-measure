---
status: accepted
---

# 0001 — Domain guard 與 OperationLease 分離

## 現況

已由 ADR-0066 取代，見 [ADR-0066〈問題與決策〉](0066-operation-lifecycle.md#問題與決策)；[原文](retired/typed-permit-proof.md#現況)僅供追溯。

## 分界與取捨

四個只含 tab_id 的 frozen dataclass 任何 caller 都能建構，不能強制先檢查。
直接 checks 減少 carrier 與 caller 的轉接，domain 條件仍由 GuardService 統一定義。
Analyze／Post 的 capability 檢查尚未全部集中，現況見
[measure owner](../../lib/zcu_tools/gui/app/measure/README.md)。

Domain checks 不取得 lease，也不具有 release／cleanup 語意。
Busy 與硬體互斥由 owning service 與 OperationGate 在啟動時檢查。
Lease 必須在終局釋放；handle 只追蹤 operation，不能推導硬體所有權。
執行、取消與關閉責任以 [[0066]] 為準。
