---
date: 2026-10-18
status: partial
commit: 979541911
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-10-18-dyson-input-reuse.md
superseded_by: null
---

# Dyson 的解析分類與輸入共用

**讀者：** framework-dev、agent。查找這次優化的做法、結果與證據。

## 核心思路

先辨認完全相同的前綴，只演化一次。後續 Dyson rows 共用這份輸入的收縮表。

## 實測

[Sparse scan 的 R37 對照](../discussions/2026-10-18-dyson-input-reuse.md#被否決的選項與理由)包含 input capture，量的是 acquire，不是單一 kernel。原型未入庫。

## 為何採用或否決

採用輸入共用，不採用 sparse scan。[原討論的比較](../discussions/2026-10-18-dyson-input-reuse.md#不同輸入不要硬套-shared-contraction)說明不同輸入的短 plateau 何時改用 period 路徑。

## 留下的線索

[共用表與前綴的對照](../discussions/2026-10-18-dyson-input-reuse.md#相同輸入先收縮之後重用-paths)可用來分開判斷建表成本與重用收益。
