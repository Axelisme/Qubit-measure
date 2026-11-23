---
date: 2026-10-18
status: adopted
commit: b314e021c
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-10-18-two-tone-floquet.md
superseded_by: null
---

# Two-tone Floquet 的 frame 與 resonant sidebands

**讀者：** framework-dev、agent。查找這次優化的做法、結果與證據。

## 核心思路

保留不能平均掉的近共振 sidebands。在跑數值 guard 前，先比較 direct 與 Floquet 的估算成本。

## 實測

索引的[五點 frame builder 驗收](../discussions/2026-10-18-two-tone-floquet.md#frame-averages-與-bath-norm)與後來的長段 sideband 驗收是不同輪。原始工作檔未入庫。

## 為何採用或否決

採用經校準的 retained-sideband 路線，不把 finite-Fourier 證書當成連續方程的誤差界。各自的[適用條件與否決理由](../discussions/2026-10-18-two-tone-floquet.md#retain-resonant-sidebands拒絕過大的-residual-certificate)留在原討論。

## 留下的線索

先分清[短段吞吐與長段 Stark peak 場景](../discussions/2026-10-18-two-tone-floquet.md#問題)，再比較 direct 與 Floquet。
