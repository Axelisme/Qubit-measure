---
date: 2026-11-18
status: adopted
commit: e86e8129d
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/simulators/ftf/environment
discussion: ../discussions/2026-11-18-ftf-gpu-dop853-backend.md
superseded_by: null
---

# dop853 的 OpenCL GPU 後端

**讀者：** framework-dev、agent。查 dop853 的 OpenCL GPU 後端 的做法與證據。

## 核心思路

把合格的獨立 row 批次送到 FP64 GPU，沒有可用裝置時保留 CPU。

## 實測

CPU／auto 中位 wall 比 5.677 倍（`544af11bf`，Intel Arc、d=32 的 `simultaneous_drive` 24 點、7 workers、三組獨占 ABBA 的整次 evolve）。 見[原討論](../discussions/2026-11-18-ftf-gpu-dop853-backend.md#結論)。

## 為何採用或否決

這個配對支持可選的 GPU 後端。Auto 路徑仍可把尾段留給 CPU，不要求整條 timeline 在一個裝置完成。

## 留下的線索

以整次 evolve 比價並包含搬運與建構。Intel Arc 的證據不能推廣到所有裝置或輸入。
