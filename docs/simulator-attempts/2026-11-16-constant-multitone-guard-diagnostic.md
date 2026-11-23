---
date: 2026-11-16
status: adopted
commit: 587eaedcd
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-16-constant-multitone-guard-diagnostic.md
superseded_by: null
---

# 常數雙 tone 的 secular guard 與唯讀診斷

**讀者：** framework-dev、agent。查找這次優化的做法、結果與證據。

## 核心思路

在 endpoint kick 之後補查長脈衝的累積誤差，讓 kernel 與唯讀診斷共用最後一次 admission 判定。

## 實測

[原討論的凍結輸入與 density 對照](../discussions/2026-11-16-constant-multitone-guard-diagnostic.md#留下的問題)沒有標明精確的量測 commit。本文的 `commit` 是實作版本，不是量測收據。A16 的一次性原始工作檔未入庫。

## 為何採用或否決

採用補充 guard，保留原本的 endpoint kick。數值取捨見[原討論](../discussions/2026-11-16-constant-multitone-guard-diagnostic.md#考慮過的選項與比較)。

## 留下的線索

這次沒有驗證日常 FTF 常數雙 tone，也沒有重新定價。若要查數值適用範圍，可從 [Evolution 的固定輸入與對照方法](../../lib/zcu_tools/simulate/evolution/README.md#constant-pair-guard-evidence)接著查。
