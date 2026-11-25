---
date: 2026-11-26
status: active
superseded_by:
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0074-flat-readout-module.md
  - ../../zcu_lab/v2/measure-authoring.md
  - ../../tools/README.md
basis:
  repo_commit: 56c041e074613073081697900799a54764680c13
  external_sources: none
---

# 實驗選單 key 路徑警告

## 結論

Registry key 採[目錄命名慣例](../../zcu_lab/v2/measure-authoring.md#目錄佈局)。使用者說：「S2報錯就超過了，警告可以，這是慣例而不是規則，是為了方便維護而不是提供阻礙。」

Lab 登錄成功後，路徑差異只警告。來源缺漏、重複、額外或無效也只警告，但工具會跳過整表比對，因為不完整的來源無法決定一個目錄是否需要變體段。這和真正的重複 registry key 不同，後者會使 Registry 拒絕登錄。既有 key、目錄、GUI／MCP 入口和 Native tag 不變。

## 問題

原本的 checker 用同一次成功登錄取得直接宣告模組，把來源異常和目錄差異都回傳為非零。比較 gate 因此拒絕一個已能成功登錄的 catalog。只把 checker 改成 exit 0 又會讓 gate 丟掉診斷，日常開發看不到差異。

## 考慮過的選項與比較

保留原有[選單命名方案](2026-11-24-registry-key-paths.md)的完整目錄與共享目錄變體。延續 [ADR-0074](../adr/0074-flat-readout-module.md#保留獨立-heralded-實驗的條件) 已採用的 heralded 家族及選單，不回復舊 key。沿用 [ADR-0072](../adr/0072-module-boundaries.md#lab-定義與組合根) 的直接來源與真 catalog caller；不拿 Run、guide 或 Native tag 推測宣告目錄。

比較 gate 照常呼叫 checker，把成功登錄後的診斷標成 `WARN`，再繼續其他檢查。這保留診斷的可見性，不需要新旗標或手動步驟。Lab 載入、factory 或 Registry 登錄拋錯時仍顯示 `FAIL`。現況模式只報絕對診斷筆數。輸出契約見[工具文件](../../tools/README.md#registry-key-paths)。

## 被否決的選項與理由

只放寬路徑不符，會讓僅供查核的來源回呼擋住已能成功登錄的 Lab。把 checker 的所有非零結果都當警告，則會放過真正的載入或登錄錯誤。把工具改成手動 lint，平常不執行就看不到診斷。

## 留下的問題

若日後有 runtime 呼叫者需要登錄來源，應另行裁定缺來源是否屬於 runtime 錯誤。本次沒有這樣的呼叫者。

## 依據

2026-11-26 的討論以本 repo `56c041e074613073081697900799a54764680c13` 為基準。原始工作檔未入庫。
