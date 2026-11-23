---
date: 2026-10-18
status: partial
commit: 90b6f62c2
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-10-18-constant-period-builders.md
superseded_by: null
---

# Constant-tone period builders

**讀者：** framework-dev、agent。查找這次優化的做法、結果與證據。

## 核心思路

比較 period 的建表成本，對低 carrier 按 key 選 Sambe 或 CF4。表格預測另沿 gain 軸試續算。

## 實測

[Gain 與 carrier 的同輪對照](../discussions/2026-10-18-constant-period-builders.md#gain-continuation-有效carrier-continuation-無效)把命中數和完整 acquire 計時分開。原始工作檔未入庫。

## 為何採用或否決

採用 gain anchors 與按 key 選用的 Sambe，不採用 constant carrier continuation。各版[取捨](../discussions/2026-10-18-constant-period-builders.md#被否決的選項與理由)不能視為同一組 A/B。

## 留下的線索

不要沿用 constant carrier 的負結果來否定 [shaped physical blocks 的重用](./2026-10-18-shaped-period-and-gates.md)。兩者預測的資料不同。
