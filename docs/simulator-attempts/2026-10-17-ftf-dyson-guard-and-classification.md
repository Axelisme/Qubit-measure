---
date: 2026-10-17
status: adopted
commit: 11f28b048
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/experiment/ftf
discussion: ../discussions/2026-10-17-ftf-dyson-guard-and-classification.md
superseded_by: null
---

# Dyson 的輕量分類與事後檢查

**讀者：** framework-dev、agent。查 Dyson 的輕量分類與事後檢查 的做法與證據。

## 核心思路

分類不預先跑整段演化；Dyson 的結果超過偏差門檻時記錄而不重算。

## 實測

原討論的分類成本來自多輪驗收與原型，未逐筆釘量測 commit。 見[原討論](../discussions/2026-10-17-ftf-dyson-guard-and-classification.md#依據)。

## 為何採用或否決

使用者拒絕同一段數值工作算兩次。落地版的事後告警不是事前完整誤差保證。

## 留下的線索

查目前決策圖再用這份歷史證據。後續的解析分類與前綴共用另有嘗試紀錄。 另見[Dyson 輸入共用](./2026-10-18-dyson-input-reuse.md)。
