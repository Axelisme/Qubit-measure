---
date: 2026-11-17
status: partial
commit: 2514d94be
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-17-dop853-speed-route-summary.md
superseded_by: null
---

# DOP853 路線比較

**讀者：** framework-dev、agent。查 DOP853 路線比較 的做法與證據。

## 核心思路

配對比較同批次的 worker 寬度，另查載波積分、前導共用與 GPU 是否值得續做。

## 實測

8 workers 的 wall 加速中位數為 4.002307 倍（`2514d94be`，pinned DOP853 的 64 點 Gaussian 單段，對 1 worker、8 輪 warm_clear_history 的 evolve 配對）。 見[原討論](../discussions/2026-11-17-dop853-speed-route-summary.md#多-worker-的配對結果)。

## 為何採用或否決

只確認指定批次的多 worker 有收益，並未改預設。載波固定網格原型未過精度，完整 GPU 矩陣未量。

## 留下的線索

多 worker 的 wall 下降伴隨 process CPU 增加。指定取樣掃描不觸發舊前導重用，不代表改寫共享條件無益。
