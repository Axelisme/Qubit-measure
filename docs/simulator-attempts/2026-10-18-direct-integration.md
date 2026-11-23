---
date: 2026-10-18
status: partial
commit: af446f697
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-10-18-direct-integration.md
superseded_by: null
---

# Direct integration 的 RHS、步長與 interaction picture

**讀者：** framework-dev、agent。查找這次優化的做法、結果與證據。

## 核心思路

減少 Hermitian RHS 的矩陣乘法，按有效 sample spacing 限制步長，符合條件的單 tone row 獨立改用 interaction picture。

## 實測

[F5 strong-ramp study](../discussions/2026-10-18-direct-integration.md#interaction-picture-減少需要積分的-h0-相位)含後來撤掉的 gain grouping。索引的原型數字不能當正式版的收益；原始工作檔未入庫。

## 為何採用或否決

採用逐 row 的選擇。不採用讓結果依同批 rows 與 worker 切法而異的 gain grouping，詳見[否決理由](../discussions/2026-10-18-direct-integration.md#被否決的選項與理由)。

## 留下的線索

RHS CPU、evolve wall、完整 acquire wall 各回答不同問題。比較前先對照[原討論的量測範圍](../discussions/2026-10-18-direct-integration.md#依據)。
