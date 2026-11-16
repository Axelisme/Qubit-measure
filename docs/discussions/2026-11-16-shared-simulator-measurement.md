---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/README.md
  - ../../tools/simulator_measurement.py
  - ../../tools/family_pareto.py
basis:
  repo_commit: a82a482f91ad9d11cca55381d0c4411acdc5370f
  external_sources: none
---

# 共用模擬器計時與 reference 必填

## 結論

Family 量表切換到 tools 內的共用計時 owner。Session adapter 只交準備好的 invoke、inspect、clear_history 與 observer callbacks。Owner 控制 session 壽命、暖機、history 與兩個 clock，consumer 保留來源、reference preflight、誤差與結果 schema。現行責任見 [ADR-0072](../adr/0072-module-boundaries.md#開發工具的數值交接)。

Elapsed 時間採有限非負秒數。Clock 解析度可以產生 0，owner 原樣記錄，不補 epsilon、不丟樣本或重取。Process CPU 可以大於 wall。負值、NaN 與 infinity 在 sample 建構時失敗。這個規則只保證樣本有效，不保證解析度以下的速度結論可信。

測試側 timing-only estimator 退休後，量表沒有需要 reference=None 的 caller，因此 measure_family 改為 reference 必填。結果與摘要 schema 不變。

## 問題

Family 量表原先自己維護三種 history mode、暖機與 wall/process clock。成對量測也需要相同的狀態轉移。讓兩個 consumer 各維護一份，會使暖機或 history reset 被計入其中一側 timer。

量表的 EngineSession 與結果 schema 還包含自己的誤差、路徑與 observer 責任。直接把整個 consumer 搬到共用 owner，會讓來源選取、reference I/O 或場景 metadata 進入計時協定。

## 考慮過的選項與比較

採用準備好的 session callbacks。Caller 的 context 負責建構與釋放，owner 只決定何時開啟、保留或清 history。既有五樣本、兩次暖機及首筆超過 60 秒的 slow-family 規則保留。

遵循 [evolution 黑盒邊界](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution)，不操作私有 dispatcher 或 history 儲存。Inspect 與 observer 都在 clocks 停止後執行。這也保留了較晚一次 evolve 失敗時已完成樣本的通知。

接受 0 elapsed，保留 clock 的實際資訊。要求嚴格正值會拒絕未超過 clock 解析度的合法呼叫。對分母有更強需求的統計由比值計算端檢查，不由樣本 owner 猜測用途。

## 被否決的選項與理由

沒有保留兩套 mode 或 clock 實作。沒有讓 owner 讀 index、cache、場景名稱或 solver。沒有把負值改成 0，也沒有用 epsilon 製造非零速度。

沒有為退休 estimator 留下 reference=None 的 consumer 分支。通用 owner 的 observation 仍由 caller 定義，不要求它理解量表 oracle。

## 留下的問題

這次只交付共用 owner 與 family cutover，沒有交付跨 worktree controller、ABBA 統計或 telemetry 校準。Seam 測試也不代替真實成對數值驗收。兩側必須先含共同 committed 工具，才有比較基礎。

## 依據

- [Tools README](../../tools/README.md#shared-simulator-measurement) 記 caller 用法、timer 邊界與資源責任。
- [量表 cache-only 討論](2026-11-16-family-pareto-cache-only.md) 保留前一階段的來源與 reference 介面遷移。
- 原始工作紀錄未入庫，採用的責任、時間讀數規則與被否決方案已寫入本文。
