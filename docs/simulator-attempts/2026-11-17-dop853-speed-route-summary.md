---
date: 2026-11-17
status: partial
commit: e399448fa
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

既有 worker 路徑在 `2514d94be` 量測。8 workers 的 wall 加速中位數為 4.002307 倍（pinned DOP853 的 64 點 Gaussian 單段，對 1 worker、8 輪 warm_clear_history 的 evolve 配對）。見[多 worker 的配對結果](../discussions/2026-11-17-dop853-speed-route-summary.md#多-worker-的配對結果)。

另以 `e399448fa` 為量測基底，比較既有 DOP853 與未入庫的載波固定網格原型。Frontmatter 記的是這次原型比較的基底，不是 worker 的量測版本。見[載波積分器的有限負結果](../discussions/2026-11-17-dop853-speed-route-summary.md#載波積分器的有限負結果)。

## 為何採用或否決

只確認指定批次的多 worker 有收益。載波固定網格原型未過精度，完整 GPU 矩陣未量。這次整理未改 production 路徑或 worker 預設。

## 留下的線索

多 worker 的 wall 下降伴隨 process CPU 增加。指定取樣掃描不觸發舊前導重用，不代表改寫共享條件無益。
