---
date: 2026-10-28
status: investigating
commit: 84e9a420a
modules:
  - lib/zcu_tools/simulate/ftf
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-10-28-ftf-fixed-dc-flux-benchmark.md
superseded_by: null
---

# FTF 固定 DC flux ramp 截斷

**讀者：** framework-dev、agent。查 FTF 固定 DC flux ramp 截斷 的做法與證據。

## 核心思路

試較小的固定 DC 子空間，並把長 hold 以常數段表示，避免依 hold 長度加密波形。

## 實測

最長 hold 仍為 11 nodes（`84e9a420a` 上的探針，合成 FTF F1 reset、M8/local4、8 ns ramp、hold 1866.200 µs 的單點 timeline）。 見[原討論](../discussions/2026-10-28-ftf-fixed-dc-flux-benchmark.md#成本與表示)。

## 為何採用或否決

較小截斷未通過精度確認，較大參考又在 phase/gauge quadrature 失敗。沒有替換演化算法或 cache。

## 留下的線索

在相同 native 座標比較 populations。最大能跑的 M 不能當收斂參考。
