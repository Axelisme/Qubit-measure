---
date: 2026-11-16
status: adopted
commit: 5d2a890a9
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-16-floquet-refusal-cost.md
superseded_by: null
---

# Floquet 不重建被拒候選的 frame

**讀者：** framework-dev、agent。查 Floquet 不重建被拒候選的 frame 的做法與證據。

## 核心思路

把必拒條件提前，並在一次 evolve 內共用 exact-key frame。

## 實測

Staging 與 frame-reuse 配對沒有分別釘量測 commit，不合加或轉錄倍率。 見[原討論](../discussions/2026-11-16-floquet-refusal-cost.md#依據)。

## 為何採用或否決

兩步都不改收拒。拒一列不能代表共用 frame 的其他列都應拒收。

## 留下的線索

先處理能證明必拒的條件，再重用昂貴 frame；保留依 row 重算的 admission。
