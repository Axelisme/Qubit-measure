---
date: 2026-11-15
status: adopted
commit: 62ba326a6
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-15-common-frame-coherent-narrowing.md
superseded_by: null
---

# common_frame 收窄 coherent guard

**讀者：** framework-dev、agent。查 common_frame 收窄 coherent guard 的做法與證據。

## 核心思路

降低既有 coherent 門檻，讓超標列回到其他由價格選出的 family。

## 實測

Coherent 上限 0.03 → 0.0018（設定值）。原討論的格點計時未釘量測 commit。 見[原討論](../discussions/2026-11-15-common-frame-coherent-narrowing.md#考慮過的選項與比較)。

## 為何採用或否決

保留 bath 上限，只取捨對部分原本合格格的拒收，避免加一套新的 guard。

## 留下的線索

拒收後先選現有合格候選。不能用單格 fallback 成本判斷完整掃描退化。
