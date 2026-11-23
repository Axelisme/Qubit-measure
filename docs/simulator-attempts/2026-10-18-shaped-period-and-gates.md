---
date: 2026-10-18
status: partial
commit: e813bfa2a
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-10-18-shaped-period-and-gates.md
superseded_by: null
---

# Shaped periods、gate preparation 與 carrier reuse

**讀者：** framework-dev、agent。查找這次優化的做法、結果與證據。

## 核心思路

重用 guard 已建好的 Sambe 資料，只在窄掃頻組內對 physical blocks 插值。

## 實測

[F4 插值原型與正式版的比較](../discussions/2026-10-18-shaped-period-and-gates.md#narrow-carrier-的-physical-blocks-可以共用)分開記錄。索引列的是原型觀測值，該組受到其他程序負載干擾；原始工作檔未入庫。

## 為何採用或否決

採用窄頻插值，寬頻走原路徑。Gate preparation 的 phase nodes 由獨立 probes 檢查，見[原討論](../discussions/2026-10-18-shaped-period-and-gates.md#gate-的-phase-nodes-由驗證決定)。

## 留下的線索

若要改寬頻 guard 的成本，先看 [F15wide 為何略過插值](../discussions/2026-10-18-shaped-period-and-gates.md#被否決的選項與理由)。
